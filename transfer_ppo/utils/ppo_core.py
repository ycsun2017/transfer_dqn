import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
from torch.distributions.categorical import Categorical
from transfer_ppo.utils.dynamic_model import Dynamic_Model
from transfer_ppo.utils.param import Param
import copy
import matplotlib.pyplot as plt

def from_numpy(n_array, dtype=None):
    if dtype is None:
        return torch.from_numpy(n_array).to(Param.device).type(Param.dtype)
    else:
        return torch.from_numpy(n_array).to(Param.device).type(dtype)
    
def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

### Clip the Action
def clip(action, low, high):
    return np.clip(action, low, high)

def rollout(env, agent, num_trajectories=10, 
            max_steps=1000, render=False, reward_normalize=False):
    rews = []
    for i in range(num_trajectories):
        o = env.reset()
        total_rew = 0
        for t in range(max_steps):
            a = agent.act(torch.from_numpy(o).to(Param.device).type(Param.dtype))
            (o, reward, done, _info) = env.step(a.cpu().numpy())
            if reward_normalize:
                total_rew += env.r ### use the unnormalized reward from the wrapper
            else:
                total_rew += reward
            if render: 
                env.render()
            if done: break
        rews.append(total_rew)
    return sum(rews)/len(rews)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)




class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi,_ = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class LatentEncoder(nn.Module):
    def __init__(self, input_dim=8, feature_size=64, hidden_sizes=(64,64)):
        super().__init__()
        self.latent = mlp([input_dim]+hidden_sizes+[feature_size], activation=nn.Tanh, output_activation=nn.Identity)
        
    def forward(self, x):
        return self.latent(x)
    
class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, feature_size, 
                 activation, policy_layers=(), disable_encoder=False):
        super().__init__()
        if disable_encoder:
            self.encoder = lambda x: x
        else:
            self.encoder = LatentEncoder(obs_dim, feature_size, hidden_sizes)
        self.logits_net = mlp([feature_size]+policy_layers+[act_dim], nn.Tanh)
    def _distribution(self, obs):
        latent_vec = self.encoder(obs)
        logits = self.logits_net(latent_vec)
        return Categorical(logits=logits), latent_vec

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, 
                 feature_size, activation, policy_layers=(), disable_encoder=False):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        if disable_encoder:
            self.encoder = lambda x: x
        else:
            self.encoder = LatentEncoder(obs_dim, feature_size, hidden_sizes)
        self.mu_net = mlp([feature_size]+policy_layers+[act_dim], nn.Tanh)

    def _distribution(self, obs):
        latent_vec = self.encoder(obs)
        mu = self.mu_net(latent_vec)
        std = torch.exp(self.log_std)
        return Normal(mu, std), latent_vec

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution
    
class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape


### Architecture of the network:
###
### Latent Encoder: 
### Input_dim:    obs_dim 
### Hidden_sizes: encoder_layers (=[hidden_units]*encoder_layers in ppo.py, encoder_layers=2 by default)
### Output_dim:   feature_size
###
### Reward & Transition Dynamics Model: 
### Input_dim:    feature_size + act_dim
### Hidden_sizes: model_hidden_sizes (=[hidden_units]*model_layers in ppo.py, model_layers = 0 by default)
### Output_dim:   1 & feature_size
###
### Policy (Logits net):
### Input_dim:    feature_size
### Hidden_sizes: policy_layers
### Output_dim:   act_dim
###
### Value Function:
### Input_dim:   obs_dim
### Hidden_sizes: value_hidden_sizes (=[hidden_units]*value_layers in ppo.py, value_layers=2 by default)
### Output_dim:  act_dim
###


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, feature_size=128,
                 encoder_hidden_sizes=(64,64), value_hidden_sizes=(64,64), 
                 model_hidden_sizes=(), policy_layers=(), activation=nn.Tanh, 
                 no_grad_encoder=True, model_lr=0.001, delta=True, obs_only=False, env=None,
                disable_encoder=False):
        super().__init__()
        
        obs_dim = observation_space.shape[0] 
        act_dim = action_space.shape[0] if isinstance(action_space, Box) else action_space.n
        
        if disable_encoder: ### Since we don't have encoder, we will set obs_dim as feature size
            feature_size = obs_dim
        
        # policy builder depends on the type of action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, act_dim, encoder_hidden_sizes, 
                                       feature_size, activation, policy_layers, disable_encoder)
            self.dynamic_model = Dynamic_Model(feature_size, act_dim, model_hidden_sizes, 
                                              cont=True, no_grad_encoder=no_grad_encoder, 
                                               lr=model_lr, delta=delta, obs_only=obs_only, env=env)
        else:
            assert isinstance(action_space, Discrete)
            self.pi = MLPCategoricalActor(obs_dim, act_dim, encoder_hidden_sizes, 
                                          feature_size, activation, policy_layers, disable_encoder)
            self.dynamic_model = Dynamic_Model(feature_size, act_dim, model_hidden_sizes, 
                                              cont=False, no_grad_encoder=no_grad_encoder, 
                                               lr=model_lr, delta=delta, obs_only=obs_only, env=env)
        # build value function
        self.v  = MLPCritic(obs_dim, value_hidden_sizes, activation)
        
        ### Keep track of moving mean and std of observation for normalizatioin purpose
        self.MovingMeanStd = MovingMeanStd(observation_space.shape)
        self.moving_mean = torch.zeros(observation_space.shape).to(Param.device).type(Param.dtype)
        self.moving_std  = torch.ones(observation_space.shape).to(Param.device).type(Param.dtype)
    
    def step(self, obs, normalize=True, train=True):
        with torch.no_grad():
            if train:
                self.MovingMeanStd.push(obs)
            if normalize:
                self.moving_mean = self.MovingMeanStd.mean()
                self.moving_std  = self.MovingMeanStd.std()
            obs = self.normalize(obs)
            pi, latent_vec = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a, v, logp_a, latent_vec

    def act(self, obs, normalize=True):
        return self.step(obs, normalize, train=False)[0]
        
    def save(self, path): 
        state_dict = {
            'policy': self.pi.state_dict(),
            'value':  self.v.state_dict(),
            'model': self.dynamic_model.state_dict(),
            'moving_mean': self.moving_mean,
            'moving_std': self.moving_std
            }
        torch.save(state_dict, path)

    def load_models(self, path):
        checkpoint = torch.load(path)
        print("load from ", path)
        self.pi.load_state_dict(checkpoint['policy'])
        self.v.load_state_dict(checkpoint['value'])
        self.moving_mean = checkpoint['moving_mean']
        self.moving_std  = checkpoint['moving_std']

    
    ### Only load the dynamics, used when trained on target task
    def load_dynamics(self, path):
        print(self.dynamic_model)
        checkpoint = torch.load(path)
        print("load from ", path)
        self.dynamic_model.load_state_dict(checkpoint['model'])
        
    def normalize(self, obs):
        return (obs - self.moving_mean)/(self.moving_std+1e-6)


### Compute rolling mean and standard deviation
class MovingMeanStd:
    
    def __init__(self, shape):
        self.n = 0
        self.shape = shape   
    def push(self, x):
        self.n += 1
        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)
            self.old_m = self.new_m
            self.old_s = self.new_s
    def mean(self):
        if self.n > 0:
            return self.new_m
        else:
             return torch.zeros(self.shape).to(Param.device).type(Param.dtype)
    def variance(self):
        if self.n > 1:
            return self.new_s / (self.n - 1) 
        else:
            return torch.ones(self.shape).to(Param.device).type(Param.dtype)
        
    def std(self):
        return torch.sqrt(self.variance())
    
def calculate_mean_prediction_error(env, encoder, action_sequence, dyn_model, data_statistics):
    latent = lambda x: encoder(x).cpu().numpy() 
    true_states, pred_states, obs = [], [],  latent(env.reset())
    for a in action_sequence:
        pred_state = dyn_model(encoder, obs, a, data_statistics)[0]
        pred_states.append(pred_state)
        true_state = latent(env.step(a)[0])
        true_states.append(true_state)
    true_states = np.squeeze(true_states)
    pred_states = np.squeeze(pred_states)
    
    # compute mean prediction error
    mean_squared_error = lambda a, b: np.mean((a-b)**2)
    mpe = mean_squared_error(pred_states, true_states)
    
    return mpe, true_states, pred_states

def log_model_predictions(env, encoder, dyn_model, 
                          data_statistics,  itr, cont=False, 
                          horizon=20, k=12, log_dir='./plot'):
    # model predictions
    fig = plt.figure()
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    if cont:
        # calculate and log model prediction error
        low, high = env.action_space.low, env.action_space.high
        action_sequence = np.random.uniform(low, high,(horizon, act_dim))
    else:
        action_sequence = np.random.randint(act_dim, size=(horizon,))
    mpe, true_states, pred_states = calculate_mean_prediction_error(env, encoder, action_sequence, dyn_model, data_statistics)
    dimensions = np.random.choice(obs_dim, size=(k, ))

    # plot the predictions
    fig.clf()
    for i in dimensions:
        plt.subplot(k/2, 2, i+1)
        plt.plot(true_states[:,i], 'g')
        plt.plot(pred_states[:,i], 'r')
    fig.suptitle('MPE: ' + str(mpe))
    figname = log_dir+'/itr_'+str(itr)+'_predictions.png' if itr is not None else log_dir + '/prediction{}.png'.format(horizon)
    fig.savefig(figname, dpi=200, bbox_inches='tight')
