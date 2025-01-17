import numpy as np
import torch
import argparse
import os
import math
import gymnasium as gym
import sys
import random
import time
import json
import copy
import hydra
from omegaconf import OmegaConf
import multiprocessing

import utils
from common.logger import Logger
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_envs
from trainer.online_trainer import OnlineTrainer


from curl_sac import RadSacAgent
from torchvision import transforms
import data_augs as rad

def make_agent(obs_shape, action_shape, cfg, device):
    if cfg.agent == 'rad_sac':
        return RadSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=cfg.hidden_dim,
            discount=cfg.discount,
            init_temperature=cfg.init_temperature,
            alpha_lr=cfg.alpha_lr,
            alpha_beta=cfg.alpha_beta,
            actor_lr=cfg.actor_lr,
            actor_beta=cfg.actor_beta,
            actor_log_std_min=cfg.actor_log_std_min,
            actor_log_std_max=cfg.actor_log_std_max,
            actor_update_freq=cfg.actor_update_freq,
            critic_lr=cfg.critic_lr,
            critic_beta=cfg.critic_beta,
            critic_tau=cfg.critic_tau,
            critic_target_update_freq=cfg.critic_target_update_freq,
            encoder_type=cfg.encoder_type,
            encoder_feature_dim=cfg.encoder_feature_dim,
            encoder_lr=cfg.encoder_lr,
            encoder_tau=cfg.encoder_tau,
            num_layers=cfg.num_layers,
            num_filters=cfg.num_filters,
            log_interval=cfg.log_interval,
            detach_encoder=cfg.detach_encoder,
            latent_dim=cfg.latent_dim,
            data_augs=cfg.data_augs,
            steps_until_freeze=cfg.steps_until_freeze

        )
    else:
        assert 'agent is not supported: %s' % cfg.agent

@hydra.main(config_name='config', config_path='.', version_base='1.3.2')
def main(cfg: dict):
    assert torch.cuda.is_available()
    assert cfg.num_train_steps > 0, 'Must train for at least 1 step.'
    device = 'cuda'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    manager = multiprocessing.Manager()
    video_path = cfg.work_dir / 'eval_video'
    if cfg.save_video_local:
        try:
            os.makedirs(video_path)
        except:
            pass
    logger = Logger(cfg, manager)

    env = make_envs(cfg, cfg.num_envs)
    eval_env = make_envs(cfg, cfg.num_eval_envs, video_path=video_path, is_eval=True, logger=logger)

    if logger._wandb != None:
        logger._wandb.config.update(OmegaConf.to_container(cfg, resolve=True), allow_val_change=True)

    pre_transform_render_size = cfg.pre_transform_render_size if 'crop' in cfg.data_augs else cfg.render_size
 
    # project_name=cfg.domain_name+cfg.task_name
    # group_name=""+str(cfg.replay_buffer_capacity//1000)+"k"+str(cfg.steps_until_freeze//1000)+"k"+str(cfg.num_copies)

    action_shape = env.action_space.shape

    if cfg.encoder_type == 'pixel':
        action_shape = env.action_space.shape[1:]
        obs_shape = (3*cfg.frame_stack, cfg.render_size, cfg.render_size)
        pre_aug_obs_shape = (3*cfg.frame_stack,pre_transform_render_size,pre_transform_render_size)
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    replay_buffer = utils.ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=cfg.replay_buffer_capacity,
        batch_size=cfg.batch_size,
        device=device,
        image_size=cfg.render_size,
    )
    
    p = 3 * pre_aug_obs_shape[1] * pre_aug_obs_shape[2]
    l = cfg.encoder_feature_dim
    c_prime = min(cfg.num_train_steps, int(np.floor(cfg.replay_buffer_capacity * p / l / 4 / 2 / cfg.num_copies)))
    print('If frozen replay capacity will increase to ', c_prime)

    latent_buffer_critic = utils.ReplayBuffer(
        obs_shape=(cfg.encoder_feature_dim, 1),
        action_shape=action_shape,
        capacity=c_prime,
        batch_size=cfg.batch_size,
        device=device,
        is_latent=True,
        num_copies=cfg.num_copies
    )

    latent_buffer_actor = utils.ReplayBuffer(
        obs_shape=(cfg.encoder_feature_dim, 1),
        action_shape=action_shape,
        capacity=c_prime,
        batch_size=cfg.batch_size,
        device=device,
        is_latent=True,
        num_copies=cfg.num_copies
    )

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        cfg=cfg,
        device=device
    )

    trainer = OnlineTrainer(
        cfg=cfg,
		env=env,
		eval_env=eval_env,
		agent=agent,
		replay_buffer=replay_buffer,
        latent_buffer_critic=latent_buffer_critic,
        latent_buffer_actor=latent_buffer_actor,
		logger=logger,
    )

    trainer.train()
    print('\nTraining completed successfully')

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()