import random
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
from collections import namedtuple, deque
from itertools import count
from utils.param import Param



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

def roll_out_atari(agent, env, num_trajectories=3, max_steps=15000):
    rews = []
    for episode in range(num_trajectories):
        obs = env.reset()
        r = 0
        for t in count():
            action = int(agent.step(obs/255.))
            #print(action)
            obs, reward, done, info = env.step(action)
            r+=reward
            if t>max_steps:
                print('Maximum {} Steps Reached'.format(max_steps))
                print(r)
                break
            if 'episode' in info.keys():
                rews.append(info['episode']['r'])
                break
    return sum(rews)/len(rews) if len(rews)>0 else np.nan

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
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(self.flatten(x)))
        return self.fc5(x)

class Q_Ram(nn.Module):
    def __init__(self, num_states, num_actions=18, feature_size=128, encoder_layers=3, hidden_units=256):
        super(Q_Ram, self).__init__()
        self.encoder = LatentEncoder(num_states, feature_size, encoder_layers, hidden_units)
        self.fc = nn.Linear(feature_size, num_actions)
        
    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x)

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
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        a = F.relu(self.fc4(self.flatten(x)))
        a = self.fc5(a)
        v = F.relu(self.fc6(self.flatten(x)))
        v = self.fc7(v)
        a_mean = torch.mean(a, dim=1, keepdim=True)
        return (a-a_mean+v)

class LatentEncoder(nn.Module):
    def __init__(self, input_dim=8, feature_size=4, num_layers=2, hidden_units=64):
        super().__init__()
        
        self.latent = build_mlp(input_dim, feature_size, num_layers, hidden_units, 
                                activation=nn.Tanh(), output_activation=nn.Tanh())
    
    def forward(self, x):
        return self.latent(x)

class DynamicModel(nn.Module):
    def __init__(self, feature_size=4, num_actions=4, hidden_units=64, num_layers=2):
        super().__init__()
        self.num_actions = num_actions
        
        self.predict_state = build_mlp(feature_size + num_actions, feature_size, num_layers, hidden_units, 
                                activation=nn.Tanh(), output_activation=nn.Tanh())

        self.predict_reward = build_mlp(feature_size + num_actions, 1, num_layers, hidden_units, 
                                activation=nn.Tanh(), output_activation=nn.Tanh())

    def forward(self, encoder, s, a, no_grad_encoder=True):
        encoding = encoder(s)
        onehot = torch.nn.functional.one_hot(a, self.num_actions).squeeze().float()
        
        state_action = torch.cat((encoding, onehot), dim=1)
        
        if no_grad_encoder:
            state_action = state_action.detach()
        predict_next = self.predict_state(state_action)
        predict_reward = self.predict_reward(state_action)
        return predict_next, predict_reward
    
class Q_Lunar(nn.Module):
    def __init__(self, input_dim=8, num_actions=4, feature_size=4, encoder_layers=2):
        super(Q_Lunar, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc3 = nn.Linear(64, num_actions)
        self.encoder = LatentEncoder(input_dim, feature_size, encoder_layers)
        self.fc = nn.Linear(feature_size, num_actions)
        
    def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)
        x = self.encoder(x)
        return self.fc(x)
    
class Q_Test(nn.Module):
    def __init__(self, input_dim=2, num_actions=3):
        super(Q_Test, self).__init__()
        self.fc1 = nn.Linear(input_dim, 6)
        self.fc2 = nn.Linear(6, num_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def model_get(name, num_states, num_actions, duel=None):
    if name == 'Test':
        return lambda: Q_Test(input_dim=num_states, num_actions=num_actions)
    elif name == 'Lunar':
        return lambda: Q_Lunar(input_dim=num_states, num_actions=num_actions)
    elif name == 'Atari':
        if duel:
            return lambda: Q_Atari_Duel(num_actions=num_actions)
        else:
            return lambda: Q_Atari(num_actions=num_actions)
    else:
        raise Exception("Environments not supported")
    

class DQN_Agent(nn.Module):
    def __init__(self, num_states, num_actions, feature_size, learning_rate=0.00025, model_lr=0.00025, 
                 doubleQ=False, atari=False, ram=False, update_freq=None, transfer=False, hidden_units=64):
        super(DQN_Agent, self).__init__()
        
        if not atari:
            self.Q = Q_Lunar(num_states, num_actions, feature_size).to(Param.device).type(Param.dtype)
            self.target_Q = Q_Lunar(num_states, num_actions, feature_size).to(Param.device).type(Param.dtype)
        elif ram:
            self.Q = Q_Ram(num_states, num_actions, feature_size).to(Param.device).type(Param.dtype)
            self.target_Q = Q_Ram(num_states, num_actions, feature_size).to(Param.device).type(Param.dtype)
        else:
            self.Q = Q_Atari(num_actions=num_actions).to(Param.device).type(Param.dtype)
            self.target_Q = Q_Atari(num_actions=num_actions).to(Param.device).type(Param.dtype)
        
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=learning_rate)
        self.doubleQ = doubleQ
        self.atari = atari
        self.update_freq = update_freq
        self.counter = 0
        
        # dynamic model
        self.dynamic_model = DynamicModel(feature_size, num_actions, hidden_units).to(Param.device).type(Param.dtype)
        self.model_optimizer = torch.optim.Adam(self.dynamic_model.parameters(), lr=model_lr)
        self.transfer = transfer
    
    def get_model_loss(self, obs_batch, act_batch, rew_batch, next_obs_batch):
        loss_fn = torch.nn.MSELoss()
        encoded_next = self.Q.encoder(next_obs_batch)
        if self.transfer:
            encoded_next = encoded_next.detach()
        pred_next, pred_rew = self.dynamic_model(self.Q.encoder, obs_batch, act_batch, no_grad_encoder=not self.transfer)
        
#         print("pred next", pred_next)
#         print("next", encoded_next)
#         print("pred rew", pred_rew)
#         print("rew", rew_batch)

        return loss_fn(encoded_next, pred_next) + loss_fn(torch.clamp(rew_batch, -1, 1), pred_rew)
        
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
        
        ### source task
        if not self.transfer:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
            self.model_optimizer.zero_grad()
            m_loss = self.get_model_loss(obs_batch, act_batch, rew_batch, next_obs_batch)
#             print(m_loss.item())
            m_loss.backward()
            self.model_optimizer.step()
        ### target task
        else:
            m_loss = self.get_model_loss(obs_batch, act_batch, rew_batch, next_obs_batch)
            loss += 0.5 * m_loss
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
            return priority.detach(), m_loss.item()
        
        return m_loss.item()
                
    
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
    
    def save(self,log_dir=os.path.join(Param.model_dir,'dqn'), exp_name='dqn'):
        torch.save([self.Q.state_dict(), self.dynamic_model.state_dict()], os.path.join(log_dir, exp_name))
    

