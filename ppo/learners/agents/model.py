import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal
import numpy

def mlp(sizes, activation, output_activation=nn.Identity()):
    layers = []
    for j in range(len(sizes)-1):
        act = activation() if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act]
    
    return nn.Sequential(*layers)

def encode_mlp(
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

class LatentEncoder(nn.Module):
    def __init__(self, input_dim=8, feature_size=4, num_layers=2, hidden_units=64):
        super().__init__()
        
        self.latent = encode_mlp(input_dim, feature_size, num_layers, hidden_units, 
                                activation=nn.Tanh(), output_activation=nn.Tanh())
    
    def forward(self, x):
        return self.latent(x)

class DynamicModel(nn.Module):
    def __init__(self, feature_size=4, num_actions=4, hidden_units=64, num_layers=2, cont=False):
        super().__init__()
        self.num_actions = num_actions
        self.cont = cont

        self.predict_state = encode_mlp(feature_size + num_actions, feature_size, num_layers, hidden_units, 
                                activation=nn.Tanh(), output_activation=nn.Tanh())

        self.predict_reward = encode_mlp(feature_size + num_actions, 1, num_layers, hidden_units, 
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

class EncodedActor(nn.Module):
    def __init__(self, state_dim, action_dim, feature_size, activation, 
        hidden_units=64, encoder_layers=2):
        super(EncodedActor, self).__init__()
        # if type(hidden_sizes) == int:
        #     hid = [hidden_sizes]
        # else:
        #     hid = list(hidden_sizes)
        # actor
        # self.action_layer = mlp([state_dim] + hid + [action_dim], activation, nn.Softmax(dim=-1))
        
        self.encoder = LatentEncoder(state_dim, feature_size, encoder_layers, hidden_units)
        self.latent = mlp([feature_size, action_dim], activation, nn.Softmax(dim=-1))

    def forward(self, state):
        encoding = self.encoder(state)
        return self.latent(encoding)
        
    def act(self, state, device):
        state = torch.from_numpy(state).float().to(device) 
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        return state, action, dist.log_prob(action)
    
    def act_prob(self, state, action, device):
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        
        return action_logprobs

    def get_dist(self, state, device):
        if type(state) == numpy.ndarray:
            state = torch.from_numpy(state).float().to(device) 
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        
        return dist

class ContActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes, activation, action_std, device):
        super(ContActor, self).__init__()
        if type(hidden_sizes) == int:
            hid = [hidden_sizes]
        else:
            hid = list(hidden_sizes)
        # actor
        self.action_layer = mlp([state_dim] + hid + [action_dim], activation, nn.Tanh())
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
        
        
    def act(self, state, device):
        if type(state) == numpy.ndarray:
            state = torch.from_numpy(state).float().to(device) 
        action_mean = self.action_layer(state)
        cov_mat = torch.diag(self.action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return state, action.detach(), action_logprob

    def act_prob(self, state, action, device):
        action_mean = self.action_layer(state)
        cov_mat = torch.diag(self.action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)
        
        return action_logprobs
    
    def get_dist(self, state, device):
        if type(state) == numpy.ndarray:
            state = torch.from_numpy(state).float().to(device) 
        action_mean = self.action_layer(state)
#        print("model action mean", action_mean)
        cov_mat = torch.diag(self.action_var).to(device)
#        print("model cov", cov_mat)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        return dist


class Value(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        if type(hidden_sizes) == int:
            hid = [hidden_sizes]
        else:
            hid = list(hidden_sizes)
        self.v_net = mlp([obs_dim] + hid + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.
    
class QValue(nn.Module):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        if type(hidden_sizes) == int:
            hid = [hidden_sizes]
        else:
            hid = list(hidden_sizes)
        self.q = mlp([obs_dim + act_dim] + hid + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes, activation):
        super(ActorCritic, self).__init__()
        if type(hidden_sizes) == int:
            hid = [hidden_sizes]
        else:
            hid = list(hidden_sizes)
        # actor
        self.action_layer = mlp([state_dim] + hid + [action_dim], activation, nn.Softmax(dim=-1))
        
        # critic
        self.value_layer = mlp([state_dim] + hid + [1], activation)
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state, device):
        state = torch.from_numpy(state).float().to(device) 
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        return state, action, dist.log_prob(action)
    
    def act_prob(self, state, action, device):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        
        return action_logprobs
    
    def get_dist(self, state, device):
        if type(state) == numpy.ndarray:
            state = torch.from_numpy(state).float().to(device) 
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        
        return dist
    
    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.value_layer(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
    
    
class ContActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes, activation, action_std, device):
        super(ContActorCritic, self).__init__()
        if type(hidden_sizes) == int:
            hid = [hidden_sizes]
        else:
            hid = list(hidden_sizes)
            
        # action mean range -1 to 1
        self.action_layer = mlp([state_dim] + hid + [action_dim], activation, nn.Tanh())
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
        # critic
        self.value_layer = mlp([state_dim] + hid + [1], activation)
        
        self.device = device
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, device):
        state = torch.from_numpy(state).float().to(device) 
        action_mean = self.action_layer(state)
        cov_mat = torch.diag(self.action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        
        return state, action, dist.log_prob(action)
    
    def act_prob(self, state, action, device):
        action_mean = self.action_layer(state)
        cov_mat = torch.diag(self.action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)
        
        return action_logprobs
    
    def get_dist(self, state, device):
        if type(state) == numpy.ndarray:
            state = torch.from_numpy(state).float().to(device) 
        action_mean = self.action_layer(state)
        cov_mat = torch.diag(self.action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        return dist
    
    def evaluate(self, state, action):  
        action_mean = self.action_layer(state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.value_layer(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy