'''
QT-Opt: Q-value assisted CEM policy learning,
for reinforcement learning on robotics.

QT-Opt: https://arxiv.org/pdf/1806.10293.pdf
CEM: https://www.youtube.com/watch?v=tNAIHEse7Ms

Pytorch implementation
CEM for fitting the action directly, action is not directly dependent on state.
Actually CEM could used be fitting any part (the variable x or the variable y that parameterizes the variable x):
Q(s,a), a=w*s+b, CEM could fit 'Q' or 'a' or 'w', all possible and theoretically feasible. 
'''


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
<<<<<<< HEAD
np.random.seed(300)
import pickle
import os

from _qt_opt_funcs import train, collect_interaction_data, rollout_current_model, plot, plot_loss
=======
np.random.seed(111)
import pickle
import os

from _qt_opt_funcs import train, collect_interaction_data, rollout_current_model
>>>>>>> d421c56399a4b68cb3da44bcf3d58d0a55cf7628

# use_cuda = torch.cuda.is_available()
# device   = torch.device("cuda" if use_cuda else "cpu")
# print(device)

config = {}




random.seed(111)
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--collect_data', dest='collect_data', action='store_true', default=False)
parser.add_argument("--joint_online_finetuning", dest="joint_online_finetuning", action='store_true', default=False)

args = parser.parse_args()

GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device, torch.cuda.is_available())

    
class ReplayBuffer:
    def __init__(self, capacity, file_name = "buffer.pkl", buffer:list = None):
        self.capacity = capacity
        if buffer is None:
            self.buffer = []
            self.position = 0
        else: 
            self.buffer = buffer
            self.position = len(self.buffer)-1
        
        self.file_name = file_name
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
<<<<<<< HEAD

        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer        
        self.buffer[self.position] = (state, action, reward, next_state, done)

    
    def sample(self, batch_size):
        # print(len(self.buffer), batch_size)
        batch = random.sample(self.buffer, batch_size)
        # try:
        #     print(len(self.buffer), batch_size, len(batch))
        # except:
        #     print(self.buffer, batch_size, len(batch))
=======
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
    
    def sample(self, batch_size):
        print(len(self.buffer), batch_size)
        batch = random.sample(self.buffer, batch_size)
        try:
            print(len(self.buffer), batch_size, len(batch))
        except:
            print(self.buffer, batch_size, len(batch))
>>>>>>> d421c56399a4b68cb3da44bcf3d58d0a55cf7628
        
        try:
            state, action, reward, next_state, done = map(np.stack, zip(*batch)) # stack for each element
        except: 
<<<<<<< HEAD
            print("ITs happeninig again")
            breakpoint()
=======
            index = self.buffer.index(None)
            self.buffer.pop(index)
            print("ITs happeninig again")
            # breakpoint()
>>>>>>> d421c56399a4b68cb3da44bcf3d58d0a55cf7628
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)




class ContinuousActionLinearPolicy(object):
    def __init__(self, theta, state_dim, action_dim):
        assert len(theta) == (state_dim + 1) * action_dim
        self.W = theta[0 : state_dim * action_dim].reshape(state_dim, action_dim)
        self.b = theta[state_dim * action_dim : None].reshape(1, action_dim)
    def act(self, state):
        # a = state.dot(self.W) + self.b
        a = np.dot(state, self.W) + self.b
        return a
    def update(self, theta):
        self.W = theta[0 : state_dim * action_dim].reshape(state_dim, action_dim)
        self.b = theta[state_dim * action_dim : None].reshape(1, action_dim)

class PID(): 
    def __init__(self, kps=[0.1, 0.1], kds=[]):
        self.kps = kps
        
    def control(self, goal, current):
        outputs = np.zeros(np.shape(self.kps))
        for i in range(len(self.kps)):
            outputs[i] =   self.kps[i] * (goal[i]-current[i])

        return outputs

class CEM():
    ''' 
    cross-entropy method, as optimization of the action policy 
    '''
    def __init__(self, theta_dim, ini_mean_scale=0.0, ini_std_scale=1.0):
        self.theta_dim = theta_dim
        self.initialize(ini_mean_scale=ini_mean_scale, ini_std_scale=ini_std_scale)

    def initialize(self, ini_mean_scale=0.0, ini_std_scale=10.0):
        self.mean = ini_mean_scale*np.ones(self.theta_dim)
        self.std = ini_std_scale*np.ones(self.theta_dim)
        
    def sample(self):
        # theta = self.mean + np.random.randn(self.theta_dim) * self.std
        theta = self.mean + np.random.normal(size=self.theta_dim) * self.std
        return theta

    def sample_multi(self, n):
        theta_list=[]
        for i in range(n):
            theta_list.append(self.sample())
        return np.array(theta_list)


    def update(self, selected_samples):
        self.mean = np.mean(selected_samples, axis = 0)
        # print('mean: ', self.mean)
        self.std = np.std(selected_samples, axis = 0)  # plus the entropy offset, or else easily get 0 std
        # print('std: ', self.std)

        return self.mean, self.std


class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, init_w=3e-3):
        super(QNetwork, self).__init__()
        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1) # the dim 0 is number of samples
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        # x = torch.nn.sigmoid(x)
        return x

class QT_Opt():
    def __init__(self, replay_buffer, hidden_dim, q_lr=3e-4, cem_update_itr=2, select_num=1, num_samples=1, 
                 qnet_chkpoint_path:str = None, qnet1_path:str = None, qnet2_path:str = None):
        self.num_samples = num_samples
        self.select_num = select_num
        self.cem_update_itr = cem_update_itr
        self.replay_buffer = replay_buffer
        self.qnet = QNetwork(state_dim+action_dim, hidden_dim).to(device) # gpu
        self.target_qnet1 = QNetwork(state_dim+action_dim, hidden_dim).to(device)
        self.target_qnet2 = QNetwork(state_dim+action_dim, hidden_dim).to(device)
        self.cem = CEM(theta_dim = action_dim)  # cross-entropy method for updating
        
        self.q_optimizer = optim.Adam(self.qnet.parameters(), lr=q_lr)
        self.step_cnt = 0
        self.q_losses = []
        self.current_q_loss = 0.0
        self.pid = PID()
        if qnet_chkpoint_path is not None: 
            self.load_checkpoint(qnet_chkpoint_path = qnet_chkpoint_path, 
                                qnet1_path = qnet1_path, 
                                qnet2_path = qnet2_path)
    
    def get_epsilon_greedy_action(self, goal_state, current_state, epsilon = 0.5):
        rand_event = random.choices([True, False], weights=[epsilon, 1 - epsilon])
        if rand_event[0]: 
            return self.pid.control(goal = goal_state, 
                                    current = current_state)
        else: 
            return np.array([np.random.normal(scale = 10), np.random.normal(scale = 10)])
        
        
        
    def update(self, batch_size, gamma=0.9, soft_tau=1e-2, update_delay=3000):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        self.step_cnt+=1
        
        state_      = torch.FloatTensor(state).to(device)
        next_state_ = torch.FloatTensor(next_state).to(device)
        action     = torch.FloatTensor(action).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

<<<<<<< HEAD
        # print(state_)
=======
>>>>>>> d421c56399a4b68cb3da44bcf3d58d0a55cf7628
        predict_q = self.qnet(state_, action) # predicted Q(s,a) value

        # get argmax_a' from the CEM for the target Q(s', a')
        new_next_action = []
        for i in range(batch_size):      # batch of states, use them one by one, to prevent the lack of memory
            new_next_action.append(self.cem_optimal_action(next_state[i]))
            
        new_next_action=torch.FloatTensor(new_next_action).to(device)

        target_q_min = torch.min(self.target_qnet1(next_state_, new_next_action), self.target_qnet2(next_state_, new_next_action))
        target_q = reward + (1-done)*gamma*target_q_min

        q_loss = ((predict_q - target_q.detach())**2).mean()  # MSE loss, note that original paper uses cross-entropy loss
        # loss = torch.nn.CrossEntropyLoss()
        # q_loss = loss(predict_q, target_q)
        print("q_loss:", q_loss)
        self.q_losses.append(q_loss.cpu().detach().numpy())
        self.current_q_loss = self.q_losses[-1]
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # update the target nets, according to original paper:
        # one with Polyak averaging, another with lagged/delayed update
        self.target_qnet1=self.target_soft_update(self.qnet, self.target_qnet1, soft_tau)
        self.target_qnet2=self.target_delayed_update(self.qnet, self.target_qnet2, update_delay)
    


    def cem_optimal_action(self, state):
        ''' evaluate action wrt Q(s,a) to select the optimal using CEM '''
        cuda_states = torch.FloatTensor(np.vstack([state]*self.num_samples)).to(device)
        self.cem.initialize() # every time use a new cem, cem is only for deriving the argmax_a'
        for itr in range(self.cem_update_itr):
            actions = self.cem.sample_multi(self.num_samples)
            q_values = self.target_qnet1(cuda_states, torch.FloatTensor(actions).to(device)).detach().cpu().numpy().reshape(-1) # 2 dim to 1 dim
            # print(np.shape(q_values), np.shape(actions))
            max_idx=q_values.argsort()[-1]  # select one maximal q
            idx = q_values.argsort()[-int(self.select_num):]  # select top maximum q
            selected_actions = actions[idx]
            _,_=self.cem.update(selected_actions)
        optimal_action = actions[max_idx]
        return optimal_action
 

    def target_soft_update(self, net, target_net, soft_tau):
        ''' Soft update the target net '''
        print("soft update: ", len(list(target_net.parameters())))
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        return target_net

    def target_delayed_update(self, net, target_net, update_delay):
        ''' delayed update the target net '''
        if self.step_cnt%update_delay == 0:
            for target_param, param in zip(target_net.parameters(), net.parameters()):
                target_param.data.copy_(  # copy data value into target parameters
                    param.data 
                )

        return target_net

    # def save_model(self, qnet_path, qnet1_path, qnet2_path):
    #     torch.save(self.qnet.state_dict(),  qnet_path)
    #     torch.save(self.target_qnet1.state_dict(), qnet1_path)
    #     torch.save(self.target_qnet2.state_dict(), qnet2_path)
    
    def save_checkpoint(self, qnet_chkpoint_path, qnet1_path, qnet2_path):
        torch.save({'model_state_dict': self.qnet.state_dict(), 
                    'optimizer_state_dict': self.q_optimizer.state_dict(), 
                    'loss': self.current_q_loss}, qnet_chkpoint_path)
        torch.save(self.target_qnet1.state_dict(), qnet1_path)
        torch.save(self.target_qnet2.state_dict(), qnet2_path)
    
    def load_checkpoint(self, qnet_chkpoint_path, qnet1_path, qnet2_path, is_only_inference = True):
        checkpoint = torch.load(qnet_chkpoint_path)
        self.qnet.load_state_dict(checkpoint['model_state_dict'])
        self.q_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.target_qnet1.load_state_dict(torch.load(qnet1_path))
        self.target_qnet2.load_state_dict(torch.load(qnet2_path))        
        if is_only_inference:
            self.qnet.eval()
            self.target_qnet1.eval()
            self.target_qnet2.eval()
        else: 
            self.qnet.train()

    # def load_model(self, qnet_path, qnet1_path, qnet2_path, is_inference = True):
    #     self.qnet.load_state_dict(torch.load(qnet_path))
    #     self.target_qnet1.load_state_dict(torch.load(qnet1_path))
    #     self.target_qnet2.load_state_dict(torch.load(qnet2_path))
    #     if is_inference:
    #         self.qnet.eval()
    #         self.target_qnet1.eval()
    #         self.target_qnet2.eval()
    #     else: 
    #         self.qnet.train()
    #         self.target_qnet1.train()
    #         self.target_qnet2.train()


<<<<<<< HEAD

=======
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
>>>>>>> d421c56399a4b68cb3da44bcf3d58d0a55cf7628

if __name__ == '__main__':
    

    # replay_buffer_size = 5e5
    # replay_buffer = ReplayBuffer(replay_buffer_size, buffer = buffer)
    # print(len(replay_buffer))
    # ip = input("waiting....")

    NUM_JOINTS=2
    LINK_LENGTH=[200, 140]
    INI_JOING_ANGLES=[0.0, 0.0]
    SCREEN_SIZE=1000
    SPARSE_REWARD=False
    SCREEN_SHOT=False
    env=Reacher(screen_size=SCREEN_SIZE, num_joints=NUM_JOINTS, link_lengths = LINK_LENGTH, \
<<<<<<< HEAD
                ini_joint_angles=INI_JOING_ANGLES, render=False, change_goal = True, 
                change_goal_episodes = 10)
    
    
    action_dim = env.num_actions   # 2
    state_dim  = env.num_observations  # 6

=======
    ini_joint_angles=INI_JOING_ANGLES, target_pos = [743,530], render=False, change_goal = False)
    
    
    action_dim = env.num_actions   # 2
    state_dim  = env.num_observations  # 8
>>>>>>> d421c56399a4b68cb3da44bcf3d58d0a55cf7628
    hidden_dim = 512
    batch_size = 100
    qnet_model_path = './qt_opt_model/model'
    qnet1_model_path = './qt_opt_model/model1'
    qnet2_model_path = './qt_opt_model/model2'
<<<<<<< HEAD
    replay_buffer_size = 5e7
=======
    replay_buffer_size = 5e9
>>>>>>> d421c56399a4b68cb3da44bcf3d58d0a55cf7628
    data_buffer_file = "buffer.pkl"
    if os.path.exists(data_buffer_file):
        file = open("buffer.pkl",'rb')
        buffer = pickle.load(file)    
<<<<<<< HEAD
        try: 
            idx = buffer.index(None)
            buffer.pop(idx)
            print("None in buffer")
        except: 
            print("None not in buffer")
            
    else: 
        buffer = None

=======
    else: 
        buffer = None
    idx = buffer.index(None)
    buffer.pop(idx)
>>>>>>> d421c56399a4b68cb3da44bcf3d58d0a55cf7628
    replay_buffer = ReplayBuffer(replay_buffer_size, buffer = buffer)

    qt_opt = QT_Opt(replay_buffer, hidden_dim)

    if args.train:
        print("Is it training??", qt_opt.qnet.training)
        train(env = env, 
            qt_opt = qt_opt, 
            n_epochs = 10000, 
            is_online_finetuning = True)
        
    if args.collect_data: 
<<<<<<< HEAD
        env.render = True
        collect_interaction_data(qt_opt = qt_opt,
                                env = env,
                                max_episodes = 3500, 
=======
        env.render = False
        collect_interaction_data(replay_buffer = replay_buffer, 
                                env = env,
                                max_episodes = 12000, 
>>>>>>> d421c56399a4b68cb3da44bcf3d58d0a55cf7628
                                max_steps = 150, 
                                frame_idx = 0, 
                                use_noisy_expert = True, 
                                debug = True)
        
    if args.joint_online_finetuning: 
        print("Is it training??", qt_opt.qnet.training)
<<<<<<< HEAD
        # qt_opt.load_checkpoint(qnet_chkpoint_path = qnet_model_path, 
        #                 qnet1_path = qnet1_model_path, 
        #                 qnet2_path = qnet2_model_path, 
        #                 is_only_inference = False)
=======
>>>>>>> d421c56399a4b68cb3da44bcf3d58d0a55cf7628
        # train(env = env, 
        #     qt_opt = qt_opt, 
        #     n_epochs = 30000, 
        #     batch_size = batch_size, 
<<<<<<< HEAD
        #     rollout_every_n_updates = 1000,
        #     is_rollout_model = True,
=======
>>>>>>> d421c56399a4b68cb3da44bcf3d58d0a55cf7628
        #     is_online_finetuning = False,
        #     qnet_checkpoint_path = qnet_model_path , 
        #     qnet1_model_path = qnet1_model_path,
        #     qnet2_model_path = qnet2_model_path)
        qt_opt.load_checkpoint(qnet_chkpoint_path = qnet_model_path, 
                               qnet1_path = qnet1_model_path, 
                               qnet2_path = qnet2_model_path, 
                               is_only_inference = False)
        print("Is it training??", qt_opt.qnet.training)  
        qnet_model_path = './qt_opt_model1/model'
        qnet1_model_path = './qt_opt_model1/model1'
<<<<<<< HEAD
        qnet2_model_path = './qt_opt_model1/model2'     
=======
        qnet2_model_path = './qt_opt_model1/model2'      
>>>>>>> d421c56399a4b68cb3da44bcf3d58d0a55cf7628
        train(env = env, 
              batch_size = batch_size,
            qt_opt = qt_opt, 
            n_epochs = 90000, 
            rollout_every_n_updates = 5000,
            is_rollout_model = True,
<<<<<<< HEAD
            is_online_finetuning = False,
=======
            is_online_finetuning = True,
>>>>>>> d421c56399a4b68cb3da44bcf3d58d0a55cf7628
            qnet_checkpoint_path = qnet_model_path , 
            qnet1_model_path = qnet1_model_path,
            qnet2_model_path = qnet2_model_path)        
        
        
        
    if args.test:
        env.render = True
        qt_opt.load_checkpoint(qnet_chkpoint_path = qnet1_model_path, qnet1_path = qnet1_model_path, 
                               qnet2_path = qnet1_model_path, is_only_inference=True )
        # hyper-parameters
        max_episodes  = 10
        max_steps   = 100
        frame_idx   = 0
        episode_rewards = []

        for i_episode in range (max_episodes):
            
            state = env.reset(SCREEN_SHOT)
            episode_reward = 0

            for step in range(max_steps):
                # action = qt_opt.policy.act(state)  
                action = qt_opt.cem_optimal_action(state)
                next_state, reward, done, _ = env.step(action, SPARSE_REWARD, SCREEN_SHOT)
                episode_reward += reward
                state = next_state

            episode_rewards.append(episode_reward)

            plot(episode_rewards)

            print('Episode: {}  | Reward:  {}'.format(i_episode, episode_reward))
    

