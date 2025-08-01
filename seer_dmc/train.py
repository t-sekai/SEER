import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import argparse
import os
import math
import sys
import random
import time
import json
from envs.dmcontrol import make_env
import copy

import utils
from logger import Logger
from video import VideoRecorder

from curl_sac import RadSacAgent
from torchvision import transforms
import data_augs as rad
from tqdm import tqdm
     

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='cartpole')
    parser.add_argument('--task_name', default='swingup')
    parser.add_argument('--pre_transform_image_size', default=100, type=int)

    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    # train
    parser.add_argument('--agent', default='rad_sac', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    parser.add_argument('--eval_freq', default=1000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float) # try 0.05 or 0.1
    parser.add_argument('--critic_target_update_freq', default=2, type=int) # try to change it to 1 and retain 0.01 above
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--latent_dim', default=128, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--detach_encoder', default=False, action='store_true')

    # wandb
    parser.add_argument('--wandb', default=False, action="store_true")
    parser.add_argument('--exp_name', default='default', type=str)
    parser.add_argument('--wandb_project', default='seer_dmc', type=str)
    parser.add_argument('--wandb_name', default='', type=str)
    parser.add_argument('--wandb_group', default='', type=str)

    # data augs
    parser.add_argument('--data_augs', default='crop', type=str) # actually shift


    parser.add_argument('--log_interval', default=100, type=int)

    parser.add_argument('--steps_until_freeze', default=10000, type=int)
    parser.add_argument('--num_copies', default=1, type=int)
    args = parser.parse_args()
    return args


def evaluate(env, agent, num_episodes, L, step, args):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        for i in tqdm(range(num_episodes)):
            obs = env.reset()
            if args.save_video:
                L.video.init(env, enabled=(i == 0))
            done = False
            episode_reward = 0
            while not done:
                # center crop image
                if args.encoder_type == 'pixel' and 'crop' in args.data_augs:
                    obs = utils.center_crop_image(obs,args.image_size)
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs / 255.)
                    else:
                        action = agent.select_action(obs / 255.)
                obs, reward, done, _ = env.step(action)
                if args.save_video:
                    L.video.record(env)
                episode_reward += reward
            if args.save_video:
                L.video.save(step)
            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)
        
        L.log('eval/' + prefix + 'eval_time', time.time()-start_time , step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        std_ep_reward = np.std(all_ep_rewards)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

        filename = args.work_dir + '/' + args.domain_name + '--'+args.task_name + '-' + args.data_augs + '--s' + str(args.seed) + '--eval_scores.npy'
        key = args.domain_name + '-' + args.task_name + '-' + args.data_augs
        try:
            log_data = np.load(filename,allow_pickle=True)
            log_data = log_data.item()
        except:
            log_data = {}
            
        if key not in log_data:
            log_data[key] = {}

        log_data[key][step] = {}
        log_data[key][step]['step'] = step 
        log_data[key][step]['mean_ep_reward'] = mean_ep_reward 
        log_data[key][step]['max_ep_reward'] = best_ep_reward 
        log_data[key][step]['std_ep_reward'] = std_ep_reward 
        log_data[key][step]['env_step'] = step * args.action_repeat

        np.save(filename,log_data)
        return mean_ep_reward

    mean_ep_reward = run_eval_loop(sample_stochastically=False)
    L.dump(step)
    return mean_ep_reward



def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'rad_sac':
        return RadSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            latent_dim=args.latent_dim,
            data_augs=args.data_augs,
            steps_until_freeze=args.steps_until_freeze

        )
    else:
        assert 'agent is not supported: %s' % args.agent

def main():
    args = parse_args()
    if args.seed == -1: 
        args.__dict__["seed"] = np.random.randint(1,1000000)
    utils.set_seed_everywhere(args.seed)

    pre_transform_image_size = args.image_size

    env = make_env(args)
 
    project_name=args.domain_name+args.task_name
    group_name=""+str(args.replay_buffer_capacity//1000)+"k"+str(args.steps_until_freeze//1000)+"k"+str(args.num_copies)
    
    # make directory
    # ts = time.gmtime() 
    # ts = time.strftime("%m-%d", ts)    
    # env_name = args.domain_name + '-' + args.task_name
    # exp_name = env_name + '-' + ts + '-im' + str(args.image_size) +'-b'  \
    # + str(args.batch_size) + '-s' + str(args.seed)  + '-' + args.encoder_type
    original_work_dir = args.work_dir
    exp_name = project_name + "-" + group_name + "-s" + str(args.seed)
    args.work_dir = args.work_dir + '/'  + exp_name

    utils.make_dir(args.work_dir)
    # video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))

    # video = VideoRecorder(video_dir if args.save_video else None)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    action_shape = env.action_space.shape

    if args.encoder_type == 'pixel':
        obs_shape = (3*args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (3*args.frame_stack,pre_transform_image_size,pre_transform_image_size)
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    replay_buffer = utils.ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
    )
    
    p = 3 * pre_aug_obs_shape[1] * pre_aug_obs_shape[2]
    l = args.encoder_feature_dim
    c_prime = min(args.num_train_steps, int(np.floor(args.replay_buffer_capacity * p / l / 4 / 2 / args.num_copies)))
    print('If frozen replay capacity will increase to ', c_prime)

    latent_buffer_critic = utils.ReplayBuffer(
        obs_shape=(args.encoder_feature_dim, 1),
        action_shape=action_shape,
        capacity=c_prime,
        batch_size=args.batch_size,
        device=device,
        is_latent=True,
        num_copies=args.num_copies
    )

    latent_buffer_actor = utils.ReplayBuffer(
        obs_shape=(args.encoder_feature_dim, 1),
        action_shape=action_shape,
        capacity=c_prime,
        batch_size=args.batch_size,
        device=device,
        is_latent=True,
        num_copies=args.num_copies
    )

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )

    L = Logger(args, args.work_dir)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()

    def get_shift_obs_batch(obs, next_obs):
        obs = obs.astype(np.uint8)
        next_obs = next_obs.astype(np.uint8)
        obs_tmp = utils.random_shift(obs) #utils.random_crop(obs,self.cfg.render_size)
        obs_tmp = torch.as_tensor(obs, device=device).float()
        next_obs_tmp = utils.random_shift(next_obs) #utils.random_crop(next_obs,self.cfg.render_size)
        next_obs_tmp = torch.as_tensor(next_obs, device=device).float()
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
            raw_obses_repeated = np.repeat(replay_buffer.obses[start:end], args.num_copies, axis=0)
            raw_next_obses_repeated = np.repeat(replay_buffer.next_obses[start:end], args.num_copies, axis=0)
            tmp_obses, tmp_next_obses = get_shift_obs_batch(raw_obses_repeated, raw_next_obses_repeated)

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
                # store args.num_copies random crops for each observation in the current batch
                buffer.obses[start:end] = latent_obses.reshape((end - start, args.num_copies, args.encoder_feature_dim, 1))
                buffer.next_obses[start:end] = latent_next_obses.reshape((end - start, args.num_copies, args.encoder_feature_dim, 1))
                # set buffer.idx and buffer.full appropriately (handles case where buffer.capacity > replay_buffer.capacity)
                buffer.idx = max(replay_buffer.idx, num_transitions)
                buffer.full = num_transitions >= buffer.capacity
            k += 1

    for step in tqdm(range(args.num_train_steps)):
        # evaluate agent periodically

        if step % args.eval_freq == 0:
            L.log('eval/episode', episode, step)
            
            evaluate(env, agent, args.num_eval_episodes, L, step,args)
            if args.save_model:
                agent.save_curl(model_dir, step)
                agent.save(model_dir, step)
            if args.save_buffer:
                buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer' + str(step)))
                replay_buffer.save(buffer_dir)

        if done:
            if step > 0:
                if step % args.log_interval == 0:
                    L.log('train/duration', time.time() - start_time, step)
                    L.dump(step)
                start_time = time.time()
            if step % args.log_interval == 0:
                L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step % args.log_interval == 0:
                L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs / 255.)

        # run training update
        if step >= args.init_steps:
            num_updates = 1 
            for _ in range(num_updates):
                if step < args.steps_until_freeze:
                    agent.update(replay_buffer, L, step, detach_fc=False)
                elif step == args.steps_until_freeze:
                    print("detaching fc layer")
                    agent.critic.encoder.detach_fc = True
                    agent.critic_target.encoder.detach_fc = True
                    agent.actor.encoder.detach_fc = True

                    num_transitions = min(step, replay_buffer.capacity)

                    utils.soft_update_params(agent.critic, agent.critic_target, 1) # set critic_target params to critic params
                    with torch.no_grad():
                        networks = [agent.critic, agent.actor]
                        buffers = [latent_buffer_critic, latent_buffer_actor]
                        # move actions, rewards, and not_dones to latent buffers
                        move_ac_rew_nd(replay_buffer, buffers, num_transitions)
                        # move obs and next_obs to latent buffers
                        move_imgs_to_latent(replay_buffer, buffers, networks, 100, num_transitions)

                    agent.update_with_latent(latent_buffer_critic, latent_buffer_actor, L, step)
                else:
                    agent.update_with_latent(latent_buffer_critic, latent_buffer_actor, L, step)
        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episode_reward += reward
        if step <= args.steps_until_freeze:
            replay_buffer.add(obs, action, reward, next_obs, done_bool)
        else: # add to latent buffers
            # similar to the "elif step == args.steps_until_freeze" procedure
            raw_obs_repeated = np.repeat(np.expand_dims(obs, axis=0), args.num_copies, axis=0)
            raw_next_obs_repeated = np.repeat(np.expand_dims(next_obs, axis=0), args.num_copies, axis=0)
            obs_tmp, next_obs_tmp = get_shift_obs_batch(raw_obs_repeated, raw_next_obs_repeated)
            networks = [agent.critic, agent.actor]
            buffers = [latent_buffer_critic, latent_buffer_actor]

            conv4_obs, conv4_next_obs = None, None
            for i in range(len(buffers)):
                network, buffer = networks[i], buffers[i]
                if conv4_obs is not None:
                    latent_obs, latent_next_obs, _, _ = get_latent_obs(network, conv4_obs, conv4_next_obs)
                else:
                    latent_obs, latent_next_obs, conv4_obs, conv4_next_obs = get_latent_obs(network, obs_tmp, next_obs_tmp)
                latent_obs = latent_obs.unsqueeze(-1).detach().cpu().numpy()
                latent_next_obs = latent_next_obs.unsqueeze(-1).detach().cpu().numpy()
                buffer.add(latent_obs, action, reward, latent_next_obs, done_bool)
        obs = next_obs
        episode_step += 1

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    main()