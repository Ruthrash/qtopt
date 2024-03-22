import math
import random
import argparse

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical
from collections import namedtuple

from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display
from reacher import Reacher
np.random.seed(111)
import pickle
import os


SPARSE_REWARD=False
SCREEN_SHOT=False
    
def collect_interaction_data(qt_opt,
                            env,
                            max_episodes = 4500, 
                            max_steps = 150, 
                            frame_idx = 0,
                            use_noisy_expert = True, 
                            debug = True): 
    episode_rewards = []
    for i_episode in range (max_episodes):
        state = env.reset(SCREEN_SHOT)
        episode_reward = 0
        for step in range(max_steps):
            if use_noisy_expert:
                #goal_joint_angles = env.compute_ik(env.target_pos)
                # print(env.target_pos, env.compute_fk(env.goal_joint_angles),env.compute_fk(env.joint_angles))
                # print( env.goal_joint_angles, env.joint_angles)
                # print("state: ", state)
                action = qt_opt.get_epsilon_greedy_action(env.goal_joint_angles, 
                                                         env.joint_angles, 
                                                         epsilon = 0.3)
            else: 
                action = qt_opt.cem_optimal_action(state)
            # if debug and (step+1) % 30 == 0: 
            #     print("action:", action)
            #     
            next_state, reward, done, _ = env.step(action, SPARSE_REWARD, SCREEN_SHOT)
            episode_reward += reward
            qt_opt.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
        episode_rewards.append(episode_reward)    
        print("length of replay buffer: {} | Episode: {} | Episode reward: {}".format(len(qt_opt.replay_buffer), i_episode, episode_reward))
        # print(len(qt_opt.replay_buffer), (max_steps*20))
        if len(qt_opt.replay_buffer) % (max_steps*20) == 0: 
            print("saving buffer")
            with open(qt_opt.replay_buffer.file_name, 'wb') as file: 
                pickle.dump(qt_opt.replay_buffer.buffer, file) 
        if i_episode% 10==0:
            plot(episode_rewards)
            plot_loss(qt_opt.q_losses)     
            
def train(env, qt_opt,qnet_checkpoint_path, qnet1_model_path, qnet2_model_path,
          batch_size, 
          n_epochs = 20000, rollout_every_n_updates = 600,
          is_online_finetuning = False, is_rollout_model = False): 
    for i in range(n_epochs):
        qt_opt.update(batch_size)
        if i % rollout_every_n_updates == 0:
            qt_opt.save_checkpoint(qnet_chkpoint_path = qnet_checkpoint_path, 
                                qnet1_path = qnet1_model_path,
                                qnet2_path = qnet2_model_path)
        # if i % 5000 == 0: 
        #     qt_opt.q_optimizer.param_groups[0]['lr'] *= 2.5
        print("gradient update steps: {},length of replay buffer: {}, q_net trainable: {}".format( i, len(qt_opt.replay_buffer), qt_opt.qnet.training))
        if is_rollout_model and i % rollout_every_n_updates == 0: 
            env.render = True
            qt_opt.load_checkpoint(qnet_chkpoint_path = qnet_checkpoint_path, 
                                   qnet1_path = qnet1_model_path, 
                                   qnet2_path = qnet2_model_path,
                                   is_only_inference = not(is_online_finetuning))
            # hyper-parameters
            max_episodes  = 20
            max_steps   = 100
            frame_idx   = 0
            rollout_current_model(env = env, 
                                  qt_opt = qt_opt, 
                                  max_steps = max_steps, 
                                  max_episodes = max_episodes, 
                                  frame_idx = frame_idx, 
                                  is_collect_rollout = is_online_finetuning)
    qt_opt.save_checkpoint(qnet_chkpoint_path=qnet_checkpoint_path, 
                    qnet1_path = qnet1_model_path,
                    qnet2_path = qnet2_model_path)            

def rollout_current_model(env,
                qt_opt, 
                is_collect_rollout = False,
                max_steps = 150, 
                max_episodes = 10, 
                frame_idx = 0,
                is_plot_ep_rewards = False):
    episode_rewards = []
    for i_episode in range(max_episodes):
        state = env.reset(SCREEN_SHOT)
        episode_reward = 0
        for step in range(max_steps): 
            action = qt_opt.cem_optimal_action(state)
            next_state, reward, done, _ = env.step(action, SPARSE_REWARD, SCREEN_SHOT)
            episode_reward += reward
            if is_collect_rollout: 
                qt_opt.replay_buffer.push(state, action, reward, next_state, done)
                if (None in qt_opt.replay_buffer.buffer):
                    breakpoint()
                if len(qt_opt.replay_buffer) % (max_steps*20) == 0: 
                    print("saving buffer")
                    with open(qt_opt.replay_buffer.file_name, 'wb') as file: 
                        pickle.dump(qt_opt.replay_buffer.buffer, file)
            state = next_state 
        episode_rewards.append(episode_reward)
        if is_plot_ep_rewards:
            plot(episode_rewards)

        print('length of replay buffer: {} Episode: {}  | Reward:  {}'.format(len(qt_opt.replay_buffer), i_episode, episode_reward))
    


def plot_loss(q_losses):
    clear_output(True)
    plt.figure(figsize=(20,5))
    # plt.subplot(131)
    plt.plot(q_losses)
    plt.savefig('qt_opt_v3_loss.png')   

def plot(rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    # plt.subplot(131)
    plt.plot(rewards)
    plt.savefig('qt_opt_v3.png')
    # plt.show()