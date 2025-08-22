from collections import defaultdict
from time import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from trainer.base import Trainer

class RandomShiftsAug(nn.Module):
	"""
	Random shift image augmentation. (N, C, H, W) as input.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	def __init__(self, pad=3):
		super().__init__()
		self.pad = pad

	def forward(self, x):
		x = x.float()
		n, _, h, w = x.size()
		assert h == w
		padding = tuple([self.pad] * 4)
		x = F.pad(x, padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
		shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
		shift *= 2.0 / (h + 2 * self.pad)
		grid = base_grid + shift
		return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
	
class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._start_time = time()

	def final_info_metrics(self, info):
		metrics = dict()
		if self.cfg.env_type == 'gpu':
			for k, v in info["final_info"]["episode"].items():
				metrics[k]=v.float().mean().item()
		else: # cpu
			temp = defaultdict(list)
			for final_info in info["final_info"]:
				for k, v in final_info["episode"].items():
					temp[k].append(v)
			for k, v in temp.items():
				metrics[k]=np.mean(v)
		return metrics

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		return dict(
			step=self._step,
			total_time=time() - self._start_time,
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		sample_stochastically = False
		for i in range(self.cfg.eval_episodes_per_env):
			obs, _ = self.eval_env.reset()
			done = torch.full((self.cfg.num_eval_envs, ), False, device=obs.device)
			while not done[0]:
				with utils.eval_mode(self.agent):
					if sample_stochastically:
						action = self.agent.sample_action(obs / 255.)
					else:
						action = self.agent.select_action(obs / 255.)
				obs, reward, terminated, truncated, info = self.eval_env.step(action)
				done = terminated | truncated
		# Update logger
		eval_metrics = dict()
		eval_metrics.update(self.final_info_metrics(info))
		return eval_metrics

	def train(self):
		device = 'cuda'
		episode, episode_reward, done = 0, 0, True

		self.random_crop = RandomShiftsAug()
		
		def get_cropped_obs_batch(obs, next_obs):
			obs = obs.astype(np.uint8)
			next_obs = next_obs.astype(np.uint8)
			obs_tmp = torch.as_tensor(obs, device=device).float()
			obs_tmp = self.random_crop(obs_tmp) #utils.random_crop(obs,self.cfg.render_size)
			next_obs_tmp = torch.as_tensor(next_obs, device=device).float()
			next_obs_tmp = self.random_crop(next_obs_tmp) #utils.random_crop(next_obs,self.cfg.render_size)
			return obs_tmp / 255, next_obs_tmp / 255

		def get_latent_obs(network, obses, next_obses):
			network.encoder(obses)
			conv4_obses = network.encoder.outputs['conv4']
			latent_obses = network.encoder.outputs['fc']
			network.encoder(next_obses)
			conv4_next_obses = network.encoder.outputs['conv4']
			latent_next_obses = network.encoder.outputs['fc']
			return latent_obses, latent_next_obses, conv4_obses, conv4_next_obses

		def move_ac_rew_nd(replay_buffer, buffers, num_transitions):
			for buffer in buffers:
				buffer.actions[:num_transitions] = replay_buffer.actions[:num_transitions]
				buffer.rewards[:num_transitions] = replay_buffer.rewards[:num_transitions]
				buffer.not_dones[:num_transitions] = replay_buffer.not_dones[:num_transitions]

		def move_imgs_to_latent(replay_buffer, buffers, networks, tmp_batch_size, num_transitions):
			k = 0
			# move in batches to avoid cuda out of memory
			while k * tmp_batch_size < num_transitions:
				start = k * tmp_batch_size
				end = min((k + 1) * tmp_batch_size, num_transitions)
				# repeat num_copies times along batch dimension to get different crops
				raw_obses_repeated = np.repeat(replay_buffer.obses[start:end],self.cfg.num_copies, axis=0)
				raw_next_obses_repeated = np.repeat(replay_buffer.next_obses[start:end],self.cfg.num_copies, axis=0)
				tmp_obses, tmp_next_obses = get_cropped_obs_batch(raw_obses_repeated, raw_next_obses_repeated)

				conv4_obses, conv4_next_obses = None, None
				for i in range(len(buffers)):
					network, buffer = networks[i], buffers[i]
					# for the actor network we only need to run the fc layer, so use previous conv4_obses from critic network
					# (the networks are tied at their convolutional layers)
					if conv4_obses is not None:
						latent_obses, latent_next_obses, _, _ = get_latent_obs(network, conv4_obses, conv4_next_obses)
					else:
						latent_obses, latent_next_obses, conv4_obses, conv4_next_obses = get_latent_obs(network, tmp_obses, tmp_next_obses)
					latent_obses = latent_obses.detach().cpu().numpy()
					latent_next_obses = latent_next_obses.detach().cpu().numpy()
					# storeself.cfg.num_copies random crops for each observation in the current batch
					buffer.obses[start:end] = latent_obses.reshape((end - start,self.cfg.num_copies,self.cfg.encoder_feature_dim, 1))
					buffer.next_obses[start:end] = latent_next_obses.reshape((end - start,self.cfg.num_copies,self.cfg.encoder_feature_dim, 1))
					# set buffer.idx and buffer.full appropriately (handles case where buffer.capacity > replay_buffer.capacity)
					buffer.idx = max(replay_buffer.idx, num_transitions)
					buffer.full = num_transitions >= buffer.capacity
				k += 1


		### Start of training ###
		train_metrics, time_metrics, vec_done, eval_next = {}, {}, [True], True
		rollout_times = []

		while self._step <= self.cfg.num_train_steps:
			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq < self.cfg.num_envs:
				eval_next = True

			if vec_done[0]:
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					self.logger.log_wandb_video(self._step)
					eval_next = False
				if self.cfg.save_buffer:
					buffer_dir = utils.make_dir(os.path.join(self.cfg.work_dir, 'buffer' + str(self._step)))
					self.replay_buffer.save(buffer_dir)

				if self._step > 0:
					train_metrics.update(self.final_info_metrics(vec_info))
					if self._step > self.cfg.init_steps:
						time_metrics.update(
							rollout_time=np.mean(rollout_times),
							rollout_fps=self.cfg.num_envs/np.mean(rollout_times), # self.cfg.num_envs * len(rollout_times)@steps_per_env@ /sum(rollout_times)
							update_time=update_time,
						)
						time_metrics.update(self.common_metrics())
						self.logger.log(time_metrics, 'time')
						rollout_times = []

					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')

				obs, _ = self.env.reset()
				episode_step = 0
					
			rollout_time = time()
			# sample action for data collection
			if self._step <self.cfg.init_steps:
				action = torch.from_numpy(self.env.action_space.sample()).cuda()
			else:
				with utils.eval_mode(self.agent):
					action = self.agent.sample_action(obs / 255.).cuda()

			next_obs, reward, vec_terminated, vec_truncated, vec_info = self.env.step(action)
			vec_done = vec_terminated | vec_truncated
			true_next_obs = next_obs
			if vec_done[0]: # use actual final_observation
				if self.cfg.obs == 'rgb': # RGB
					true_next_obs = vec_info["final_observation"]
				else:
					true_next_obs = vec_info["final_observation"]
					
			# allow infinit bootstrap
			done_bool = vec_terminated.float() 
			rollout_time = time() - rollout_time
			rollout_times.append(rollout_time)

			episode_reward += reward
			if self._step <=self.cfg.steps_until_freeze:
				self.replay_buffer.add(obs.cpu().numpy(), action.cpu().numpy(), reward.cpu().numpy(), true_next_obs.cpu().numpy(), done_bool.cpu().numpy())
			else: # add to latent buffers
				# similar to the "elif step ==self.cfg.steps_until_freeze" procedure
				raw_obs_repeated = np.repeat(np.expand_dims(obs.cpu().numpy(), axis=0),self.cfg.num_copies, axis=0) # TODO: weird part where obs is changed from gpu to cpu to gpu to cpu
				raw_true_next_obs_repeated = np.repeat(np.expand_dims(true_next_obs.cpu().numpy(), axis=0),self.cfg.num_copies, axis=0)
				obs_tmp, true_next_obs_tmp = [], []
				for i in range(self.cfg.num_copies):
					_obs_tmp, _true_net_obs_tmp =  get_cropped_obs_batch(raw_obs_repeated[i], raw_true_next_obs_repeated[i]) 
					obs_tmp.append(_obs_tmp)
					true_next_obs_tmp.append(_true_net_obs_tmp)
				obs_tmp = torch.stack(obs_tmp).permute(1,0,2,3,4)
				true_next_obs_tmp = torch.stack(true_next_obs_tmp).permute(1,0,2,3,4)
				networks = [self.agent.critic, self.agent.actor]
				buffers = [self.latent_buffer_critic, self.latent_buffer_actor]

				obs_tmp = obs_tmp.reshape((self.cfg.num_envs*self.cfg.num_copies, *obs_tmp.shape[2:]))
				true_next_obs_tmp = true_next_obs_tmp.reshape((self.cfg.num_envs*self.cfg.num_copies, *true_next_obs_tmp.shape[2:]))
				conv4_obs, conv4_true_next_obs = None, None
				for i in range(len(buffers)):
					network, buffer = networks[i], buffers[i]
					if conv4_obs is not None:
						latent_obs, latent_true_next_obs, _, _ = get_latent_obs(network, conv4_obs, conv4_true_next_obs)
					else:
						latent_obs, latent_true_next_obs, conv4_obs, conv4_true_next_obs = get_latent_obs(network, obs_tmp, true_next_obs_tmp)
					latent_obs = latent_obs.reshape((self.cfg.num_envs, self.cfg.num_copies, *latent_obs.shape[1:]))
					latent_true_next_obs = latent_true_next_obs.reshape((self.cfg.num_envs, self.cfg.num_copies, *latent_true_next_obs.shape[1:]))
					latent_obs = latent_obs.unsqueeze(-1).detach().cpu().numpy()
					latent_true_next_obs = latent_true_next_obs.unsqueeze(-1).detach().cpu().numpy()
					buffer.add(latent_obs, action.cpu().numpy(), reward.cpu().numpy(), latent_true_next_obs, done_bool.cpu().numpy())

			# Update agent
			if self._step >=self.cfg.init_steps:
				update_time = time()
				num_updates = max(1,int(self.cfg.utd * self.cfg.num_envs))
				for _ in range(num_updates):
					if self._step <self.cfg.steps_until_freeze:
						self.agent.update(self.replay_buffer, L=None, step=self._step, detach_fc=False, random_crop=self.random_crop)
					elif self._step ==self.cfg.steps_until_freeze:
						print("detaching fc layer")
						self.agent.critic.encoder.detach_fc = True
						self.agent.critic_target.encoder.detach_fc = True
						self.agent.actor.encoder.detach_fc = True

						num_transitions = min(self._step, self.replay_buffer.capacity)

						utils.soft_update_params(self.agent.critic, self.agent.critic_target, 1) # set critic_target params to critic params
						with torch.no_grad():
							networks = [self.agent.critic, self.agent.actor]
							buffers = [self.latent_buffer_critic, self.latent_buffer_actor]
							# move actions, rewards, and not_dones to latent buffers
							move_ac_rew_nd(self.replay_buffer, buffers, num_transitions)
							# move obs and next_obs to latent buffers
							move_imgs_to_latent(self.replay_buffer, buffers, networks, 100, num_transitions)

						self.agent.update_with_latent(self.latent_buffer_critic, self.latent_buffer_actor, L=None, step=self._step)
					else:
						self.agent.update_with_latent(self.latent_buffer_critic, self.latent_buffer_actor, L=None, step=self._step)
				update_time = time() - update_time
			
			obs = next_obs
			self._step += self.cfg.num_envs
			episode_step += 1
		
		self.logger.finish() # TODO save agent
