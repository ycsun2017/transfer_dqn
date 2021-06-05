import random
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
from collections import namedtuple, deque
from itertools import count
from experiment.utils.param import Param



def rollout(agent, env, num_trajectories=3, num_steps=1000):
    rews = []
    for i in range(num_trajectories):
        o = env.reset()
        total_rew = 0
        for t in range(num_steps):
            a = int(agent.step(o))
            (o, reward, done, _info) = env.step(a)
            total_rew += reward
            if done: break
        rews.append(total_rew)
    return sum(rews)/len(rews)

def roll_out_atari(agent, env, num_trajectories=3):
    rews = []
    for episode in range(num_trajectories):
        obs = env.reset()
        for t in count():
            action = int(agent.step(obs))
            obs, reward, done, info = env.step(action)
            if 'episode' in info.keys():
                rews.append(info['episode']['r'])
                break
    return sum(rews)/len(rews)

def exponential_smoothing(lst, alpha=0.1):
    s = lst[0]
    smoothing = [s]
    for i in range(len(lst)-1):
        s = alpha*lst[i+1]+(1-alpha)*s
        smoothing.append(s)
    return smoothing

def build_mlp(
        input_size,
        output_size,
        n_layers,
        size,
        activation = nn.ReLU(),
        output_activation = nn.Identity(),
        init_method=None,
):

    layers = []
    in_size = input_size
    for _ in range(n_layers):
        curr_layer = nn.Linear(in_size, size)
        if init_method is not None:
            curr_layer.apply(init_method)
        layers.append(curr_layer)
        layers.append(activation)
        in_size = size

    last_layer = nn.Linear(in_size, output_size)
    if init_method is not None:
        last_layer.apply(init_method)

    layers.append(last_layer)
    layers.append(output_activation)
        
    return nn.Sequential(*layers)

### Architecture is the same as the one published in the original nature paper
### see https://www.nature.com/articles/nature14236
class Q_Atari(nn.Module):
    def __init__(self, in_channels=4, num_actions=18):
        super(Q_Atari, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)

    def forward(self, x):
        if len(x.shape)==3:
            x = x.unsqueeze(0)
        x = F.relu(self.conv1(x/255.))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(self.flatten(x)))
        return self.fc5(x)

### Dueling network architecture proposed in https://arxiv.org/pdf/1511.06581.pdf
class Q_Atari_Duel(nn.Module):
    def __init__(self, in_channels=4, num_actions=18):
        super(Q_Atari_Duel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)
        self.fc6 = nn.Linear(7 * 7 * 64, 512)
        self.fc7 = nn.Linear(512, 1)

    def forward(self, x):
        if len(x.shape)==3:
            x = x.unsqueeze(0)
        x = F.relu(self.conv1(x/255.))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        a = F.relu(self.fc4(self.flatten(x)))
        a = self.fc5(a)
        v = F.relu(self.fc6(self.flatten(x)))
        v = self.fc7(v)
        a_mean = torch.mean(a, dim=1, keepdim=True)
        return (a-a_mean+v)

class Q_Lunar(nn.Module):
    def __init__(self, input_dim=8, num_actions=4):
        super(Q_Lunar, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class Q_Test(nn.Module):
    def __init__(self, input_dim=2, num_actions=3):
        super(Q_Test, self).__init__()
        self.fc1 = nn.Linear(input_dim, 6)
        self.fc2 = nn.Linear(6, num_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def model_get(name, num_actions, duel=None):
    if name == 'Test':
        return lambda: Q_Test(num_actions=num_actions)
    elif name == 'Lunar':
        return lambda: Q_Lunar(num_actions=num_actions)
    elif name == 'Atari':
        if duel:
            return lambda: Q_Atari_Duel(num_actions=num_actions)
        else:
            return lambda: Q_Atari(num_actions=num_actions)
    else:
        raise Exception("Environments not supported")
    

class DQN_Agent(nn.Module):
    def __init__(self, q_func,learning_rate= 0.00025, doubleQ=False, atari=False, update_freq=None):
        super(DQN_Agent, self).__init__()
        self.Q = q_func().to(Param.device).type(Param.dtype)
        self.target_Q = q_func().to(Param.device).type(Param.dtype)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=learning_rate)
        self.doubleQ = doubleQ
        self.atari = atari
        self.update_freq = update_freq
        self.counter = 0
        
    def update(self, obs_batch, act_batch, rew_batch,\
               next_obs_batch, not_done_mask, gamma, tau, weights=None):
        current_Q_values = self.Q(obs_batch).gather(1, act_batch)
        if self.doubleQ:
            indices = self.Q(next_obs_batch).max(1)[-1].unsqueeze(1)
            next_max_q = self.target_Q(next_obs_batch)
            next_max_q = next_max_q.gather(1, indices)
        else:
            next_max_q = (self.target_Q(next_obs_batch).detach().max(1)[0]).unsqueeze(1)
        
        next_Q_values = not_done_mask * next_max_q
        target_Q_values = rew_batch + (gamma * next_Q_values)
        assert(next_Q_values.shape==target_Q_values.shape)
        
        ### Compute td error
        if weights is None:
            loss = F.smooth_l1_loss(current_Q_values, target_Q_values)
        else:
            loss = F.smooth_l1_loss(current_Q_values, target_Q_values, reduce=False)*(weights.unsqueeze(1))
            priority = loss + 1e-5
            loss = torch.mean(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        ### Update the Target network
        ### two methods: 1. soft update tau*\theta+(1-tau)\theta'
        ### 2. hard update: after a certain period, update the target network completely
        if (self.update_freq is None):
            self.soft_update(tau)
        else:
            self.counter += 1
            if (self.counter%self.update_freq==0):
                self.update_target()
        
        ### Return the new priority
        if weights is not None:
            return priority.detach()
                
    
    ### Update the target Q network
    def update_target(self):
        self.target_Q.load_state_dict(self.Q.state_dict())
    
    
    def soft_update(self, tau=1e-3):
        """Soft update model parameters"""
        for target_param, local_param in zip(self.target_Q.parameters(), self.Q.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def select_epilson_greedy_action(self, obs, t, exploration, num_actions):
        sample = random.random()
        eps_threshold = exploration.value(t)
        if sample > eps_threshold:
            obs = torch.from_numpy(obs).to(Param.device).type(Param.dtype)
            if not self.atari:
                obs = obs.unsqueeze(0)
            with torch.no_grad():
                return int(self.Q(obs).data.max(1)[1].cpu())
        else:
            with torch.no_grad():
                return random.randrange(num_actions)
   
    ### Choose the best action, happened during test time
    def step(self,obs):
        with torch.no_grad():
            if (len(obs.shape) == 1):
                obs = torch.from_numpy(obs).to(Param.device).type(Param.dtype).unsqueeze(0)
            else:
                obs = torch.from_numpy(obs).to(Param.device).type(Param.dtype)
            return self.Q(obs).data.max(1)[1].cpu()[0]
    
    def step_torch(self,obs):
        if (len(obs.shape)==1):
            obs = obs.unsqueeze(0)
        return self.Q(obs).data.max(1)[1].cpu()[0]
    
    def step_torch_batch(self,obs):
        # print("obs", obs)
        # print("q", self.Q(obs).data)
        # print("max", self.Q(obs).data.max(1)[1].unsqueeze(1))
        return self.Q(obs).data.max(1)[1].unsqueeze(1)
    
    def save(self,log_dir=os.path.join(Param.model_dir,'./dqn'), exp_name='dqn'):
        torch.save(self.state_dict(), os.path.join(log_dir, exp_name))
    

