from curl_sac import RadSacAgent
from utils import ReplayBuffer
from common.logger import Logger

class Trainer:
	"""Base trainer class for TD-MPC2."""

	def __init__(self, cfg, env, eval_env, agent, replay_buffer, latent_buffer_critic, latent_buffer_actor, logger):
		self.cfg = cfg
		self.env = env
		self.eval_env = eval_env
		self.agent: RadSacAgent = agent
		self.replay_buffer: ReplayBuffer = replay_buffer
		self.latent_buffer_critic: ReplayBuffer = latent_buffer_critic
		self.latent_buffer_actor: ReplayBuffer = latent_buffer_actor
		self.logger: Logger = logger
		print('Actor Architecture:', self.agent.actor)
		print('Critic Architecture:', self.agent.critic)
		total_params = sum(p.numel() for p in self.agent.actor.parameters() if p.requires_grad) + sum(p.numel() for p in self.agent.critic.parameters() if p.requires_grad)
		print("Learnable parameters: {:,}".format(total_params))

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		raise NotImplementedError

	def train(self):
		"""Train a TD-MPC2 agent."""
		raise NotImplementedError
