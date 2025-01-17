import re
from pathlib import Path

import hydra
from omegaconf import OmegaConf

def parse_cfg(cfg: OmegaConf) -> OmegaConf:
	"""
	Parses a Hydra config. Mostly for convenience.
	"""

	# Logic
	for k in cfg.keys():
		try:
			v = cfg[k]
			if v == None:
				v = True
		except:
			pass

	# Algebraic expressions
	for k in cfg.keys():
		try:
			v = cfg[k]
			if isinstance(v, str):
				match = re.match(r"(\d+)([+\-*/])(\d+)", v)
				if match:
					cfg[k] = eval(match.group(1) + match.group(2) + match.group(3))
					if isinstance(cfg[k], float) and cfg[k].is_integer():
						cfg[k] = int(cfg[k])
		except:
			pass

	# Convenience
	cfg.work_dir = Path(hydra.utils.get_original_cwd()) / 'logs' / cfg.env_id / str(cfg.seed) / cfg.exp_name

	# Maniskill
	cfg.env_cfg.env_id = cfg.eval_env_cfg.env_id = cfg.env_id
	cfg.env_cfg.obs_mode = cfg.eval_env_cfg.obs_mode = cfg.obs # state or rgb
	cfg.env_cfg.reward_mode = cfg.eval_env_cfg.reward_mode = 'normalized_dense'
	cfg.env_cfg.num_envs = cfg.num_envs
	cfg.eval_env_cfg.num_envs = cfg.num_eval_envs
	cfg.env_cfg.sim_backend = cfg.eval_env_cfg.sim_backend = cfg.env_type
	
	cfg.eval_env_cfg.num_eval_episodes = cfg.eval_episodes_per_env * cfg.num_eval_envs
		
	# cfg.(eval_)env_cfg.control_mode is defined in maniskill.py
	# cfg.(eval_)env_cfg.env_horizon is defined in maniskill.py

	return cfg
