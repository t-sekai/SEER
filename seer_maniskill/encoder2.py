import torch
import torch.nn as nn

import numpy as np
import utils


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


OUT_DIM = {2: 39, 4: 35, 6: 31}
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}

LN_DIM = [31, 15, 7, 3]

class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32,output_logits=False):
        super().__init__()

        assert len(obs_shape) == 3 # (C,H,W)
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        # try 2 5x5s with strides 2x2. with samep adding, it should reduce 84 to 21, so with valid, it should be even smaller than 21.
        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=2))

        self.ln = [nn.LayerNorm([num_filters, LN_DIM[i], LN_DIM[i]], device=torch.device('cuda')) for i in range(num_layers)]
            
        out_dim = LN_DIM[-1] #OUT_DIM_64[num_layers] if obs_shape[-1] == 64 else OUT_DIM[num_layers] 
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln_fc = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()
        self.output_logits = output_logits

        self.detach_fc = False

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        if obs.max() > 1.:
            obs = obs / 255.

        self.outputs['obs'] = obs
        conv = obs
        for i in range(0, self.num_layers):
            conv = torch.relu(self.ln[i](self.convs[i](conv)))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.reshape((conv.size(0), -1))
        return h

    def forward(self, obs, detach=False):
        if obs.shape[1] == self.feature_dim: # fc feature
            self.outputs['fc'] = obs.squeeze(-1)
            h_fc = obs.squeeze(-1).detach() 
        elif obs.shape[1:] != self.obs_shape: # conv4 feature
            conv = obs
            h = conv.view(conv.size(0), -1)
            if detach:
                h = h.detach()
            h_fc = self.fc(h)
            self.outputs['fc'] = h_fc
            if self.detach_fc:
                h_fc = h_fc.detach()
        else: # raw observation
            h = self.forward_conv(obs)
            if detach:
                h = h.detach()
            h_fc = self.fc(h)
            self.outputs['fc'] = h_fc

            if self.detach_fc:
                h_fc = h_fc.detach()

        h_norm = self.ln_fc(h_fc)
        self.outputs['ln'] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        return # TODO
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters,*args):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder}


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, output_logits=False
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, output_logits
    )
