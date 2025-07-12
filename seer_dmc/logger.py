from collections import defaultdict
import json
import os
import shutil
import torch
import torchvision
import numpy as np
from termcolor import colored
import re
import wandb
from omegaconf import OmegaConf

FORMAT_CONFIG = {
    'rl': {
        'train': [
            ('episode', 'E', 'int'), ('step', 'S', 'int'),
            ('duration', 'D', 'time'), ('episode_reward', 'R', 'float'),
            ('batch_reward', 'BR', 'float'), ('actor_loss', 'A_LOSS', 'float'),
            ('critic_loss', 'CR_LOSS', 'float')
        ],
        'eval': [('step', 'S', 'int'), ('episode_reward', 'ER', 'float')]
    }
}


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, file_name, formating):
        self._file_name = file_name
        if os.path.exists(file_name):
            os.remove(file_name)
        self._formating = formating
        self._meters = defaultdict(AverageMeter)

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith('train'):
                key = key[len('train') + 1:]
            else:
                key = key[len('eval') + 1:]
            key = key.replace('/', '_')
            data[key] = meter.value()
        return data

    def _dump_to_file(self, data):
        with open(self._file_name, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def _format(self, key, value, ty):
        template = '%s: '
        if ty == 'int':
            template += '%d'
        elif ty == 'float':
            template += '%.04f'
        elif ty == 'time':
            template += '%.01f s'
        else:
            raise 'invalid format type: %s' % ty
        return template % (key, value)

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        pieces = ['{:5}'.format(prefix)]
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print('| %s' % (' | '.join(pieces)))

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data['step'] = step
        self._dump_to_file(data)
        self._dump_to_console(data, prefix)
        self._meters.clear()

class VideoRecorder:
    """Utility class for logging evaluation videos."""
  
    def __init__(self, wandb, fps=15):
        self._wandb = wandb
        self.fps = fps
        self.frames = []
        self.enabled = False

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self._wandb and enabled
        self.record(env)

    def record(self, env):
        if self.enabled:
            self.frames.append(env.render())

    def save(self, step, key='videos/eval_video'):
        if self.enabled and len(self.frames) > 0:
            frames = np.stack(self.frames)
            return self._wandb.log(
				{key: wandb.Video(frames.transpose(0, 3, 1, 2), fps=self.fps, format='mp4')}, step=step
			)

def cfg_to_group(args, return_list=False):
	"""
	Return a wandb-safe group name for logging.
	Optionally returns group name as list.
	"""
	lst = [f"{args.domain_name}_{args.task_name}", re.sub("[^0-9a-zA-Z]+", "-", args.exp_name)]
	return lst if return_list else "-".join(lst)
class Logger(object):
    def __init__(self, args, log_dir, config='rl'):
        self._log_dir = log_dir

        self._train_mg = MetersGroup(
            os.path.join(log_dir, 'train.log'),
            formating=FORMAT_CONFIG[config]['train']
        )
        self._eval_mg = MetersGroup(
            os.path.join(log_dir, 'eval.log'),
            formating=FORMAT_CONFIG[config]['eval']
        )

        self.wandb = args.wandb
        if not args.wandb:
            print(colored("Wandb disabled.", "blue", attrs=["bold"]))
            self._wandb = None
            self._video = None
        else:
            print(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
            wandb_tags = cfg_to_group(args, return_list=True) + [f"seed:{args.seed}"] + ["seer"]
            self._wandb = wandb.init(
				project=args.wandb_project,
				name=args.wandb_name,
				group=args.wandb_group,
				tags=wandb_tags,
				config=args,
			)
            self._video = (
                 VideoRecorder(self._wandb)
                 if self._wandb and args.save_video
                 else None
            )
    
    @property
    def video(self):
        return self._video

    def _try_wandb_log(self, key, value, step):
        if self.wandb:
            self._wandb.log(data={key: value}, step=step)

    # def _try_wandb_log_image(self, key, image, step):
    #     if self.wandb:
    #         # assert image.dim() == 3
    #         # grid = torchvision.utils.make_grid(image.unsqueeze(1))
    #         # self._sw.add_image(key, grid, step)
    #         pass

    # def _try_wandb_log_histogram(self, key, histogram, step):
    #     if self.wandb:
    #         # self._sw.add_histogram(key, histogram, step)
    #         pass

    def log(self, key, value, step, n=1):
        assert key.startswith('train') or key.startswith('eval')
        if type(value) == torch.Tensor:
            value = value.item()
        self._try_wandb_log(key, value / n, step)
        mg = self._train_mg if key.startswith('train') else self._eval_mg
        mg.log(key, value, n)

    def log_param(self, key, param, step):
        pass
        # self.log_histogram(key + '_w', param.weight.data, step)
        # if hasattr(param.weight, 'grad') and param.weight.grad is not None:
        #     self.log_histogram(key + '_w_g', param.weight.grad.data, step)
        # if hasattr(param, 'bias'):
        #     self.log_histogram(key + '_b', param.bias.data, step)
        #     if hasattr(param.bias, 'grad') and param.bias.grad is not None:
        #         self.log_histogram(key + '_b_g', param.bias.grad.data, step)

    def log_image(self, key, image, step):
        pass
        # assert key.startswith('train') or key.startswith('eval')
        # self._try_wandb_log_image(key, image, step)

    def log_histogram(self, key, histogram, step):
        pass
        # assert key.startswith('train') or key.startswith('eval')
        # self._try_wandb_log_histogram(key, histogram, step)

    def dump(self, step):
        self._train_mg.dump(step, 'train')
        self._eval_mg.dump(step, 'eval')
