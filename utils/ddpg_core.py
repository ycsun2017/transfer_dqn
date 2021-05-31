import numpy as np
import scipy.signal

import torch
import torch.nn as nn
from experiment.utils.param import Param
from torch.distributions.beta import Beta
import os
    
def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def as_tensor(x):
    return torch.from_numpy(x).to(Param.device).type(Param.dtype)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class Shared_CNN:
    hidden_size = 512
    def __init__(self,num_actions=4):
        self.n_actions = num_actions
        Shared_CNN.cnn_layers = CNN_Layers(num_actions=num_actions)
    def shared_cnn_layers(self):
        return Shared_CNN.cnn_layers
    
class CNN_Layers(nn.Module):
    hidden_size=512
    def __init__(self, in_channels=1, num_actions=18):
        super(CNN_Layers, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)

    def forward(self, x):
        x = F.relu(self.conv1(x/255.))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return x

class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit, atari=False):
        super().__init__()
        self.cnn_layers = CNN_layers(num_actions=act_dim) if atari else None
        if atari:
            pi_sizes = [CNN_layers.hidden_size] + list(hidden_sizes) + [act_dim]
        else:
            pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        if self.cnn_layers is not None:
            obs = self.cnn_layers(obs)
        return self.act_limit * self.pi(obs)



class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, atari=False):
        super().__init__()
        self.cnn_layers = CNN_layers(num_actions=act_dim) if atari else None
        if atari:
            self.q = mlp([CNN_layers.hidden_size + act_dim] + list(hidden_sizes) + [1], activation)
        else:
            self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        if (self.cnn_layers is not None):
            obs = self.cnn_layers(obs)
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU, atari=False):
        super().__init__()

        obs_dim = observation_space.shape[0] if not atari else None
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit, atari=atari)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation, atari=atari)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).cpu().detach().numpy()
    
    def save(self, log_dir= os.path.join(Param.model_dir, r'./ddpg'), model_name='ddpg_policy', adv=False):
        if not adv:
            torch.save(self.state_dict(), os.path.join(log_dir,model_name))
