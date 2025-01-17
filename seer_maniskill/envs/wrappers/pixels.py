from collections import deque

import gymnasium as gym
import numpy as np
import torch


class PixelWrapper(gym.Wrapper):
	"""
	Wrapper for pixel observations. Works with Maniskill vectorized environments
	"""

	def __init__(self, cfg, env, num_envs, num_frames=3):
		super().__init__(env)
		self._vis_shape = self.unwrapped.single_observation_space['sensor_data']['base_camera']['rgb'].shape # (h, w, c)
		self.cfg = cfg
		self.env = env
		self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self._vis_shape[0], self._vis_shape[1], self._vis_shape[2], num_frames), dtype=np.uint8)
		self.single_observation_space = gym.spaces.Box(low=0, high=255, shape=(self._vis_shape[0], self._vis_shape[1], self._vis_shape[2], num_frames), dtype=np.uint8)
		self._rgb_stack = torch.zeros((num_envs, self._vis_shape[0], self._vis_shape[1], self._vis_shape[2], num_frames), dtype=torch.uint8).to(self.unwrapped.device)
		self._render_size = cfg.render_size
		self._num_frames = num_frames
		self._stack_idx = 0

	def _get_obs(self, obs):
		self._rgb_stack[..., self._stack_idx] = obs['sensor_data']['base_camera']['rgb']
		self._stack_idx = (self._stack_idx + 1) % self._num_frames
		vis = torch.cat((self._rgb_stack[..., self._stack_idx:],self._rgb_stack[..., :self._stack_idx]), dim=-1)
		vis = vis.reshape(*vis.shape[:-2], -1)
		vis = vis.permute(0, 3, 1, 2)
		return vis

	def reset(self):
		obs, info = self.env.reset()
		for _ in range(self._num_frames):
			obs_frames = self._get_obs(obs)
		return obs_frames, info

	def step(self, action):
		obs, reward, terminated, truncated, info = super().step(action)
		return self._get_obs(obs), reward, terminated, truncated, info
