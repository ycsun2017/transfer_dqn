import sys
import torch  
import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from gym.spaces import Box, Discrete
from .model import EncodedActor, ContActor, DynamicModel
from .updates import vpg_update
from torch.distributions import Categorical

class VPG(nn.Module):
    def __init__(self, state_space, action_space, feature_size, hidden_units=64, encoder_layers=2,
                 activation=nn.Tanh, learning_rate=3e-4, gamma=0.9, device="cpu", action_std=0.5,
                 transfer=False):
        super(VPG, self).__init__()
        
        # deal with 1d state input
        state_dim = state_space.shape[0]
        
        self.gamma = gamma
        self.device = device
        self.transfer = transfer
        
        if isinstance(action_space, Discrete):
            self.action_dim = action_space.n
            self.policy = EncodedActor(state_dim, self.action_dim, feature_size, 
                        activation, hidden_units, encoder_layers).to(self.device)

        # elif isinstance(action_space, Box):
        #     self.action_dim = action_space.shape[0]
        #     self.policy = ContActor(state_dim, self.action_dim, hidden_sizes, activation, action_std, self.device).to(self.device)
        self.dynamic_model = DynamicModel(feature_size, self.action_dim, hidden_units, encoder_layers).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.model_optimizer = optim.Adam(self.dynamic_model.parameters(), lr=learning_rate)
    
    def act(self, state):
        return self.policy.act(state, self.device)
        
    
    def update_policy(self, memory, op_memory):
        ### source task
        if not self.transfer:
            self.optimizer.zero_grad()
            p_loss = self.policy_loss(memory)
            p_loss.backward()
            self.optimizer.step()

            self.model_optimizer.zero_grad()
            m_loss = self.model_loss(op_memory)
            m_loss.backward()
            self.model_optimizer.step()

        ### target task
        else:
            self.optimizer.zero_grad()
            p_loss = self.policy_loss(memory)
            m_loss = self.model_loss(op_memory)
            (p_loss + 2 * m_loss).backward()
            self.optimizer.step()

        return p_loss, m_loss

    def model_loss(self, memory):
        states, actions, next_states, rewards = memory.sample(256)
        state = torch.stack(states).to(self.device).detach()
        action = torch.stack(actions).to(self.device).detach()
        next_state = torch.stack(next_states).to(self.device).detach()
        reward = torch.FloatTensor(rewards).to(self.device).detach()

        loss_fn = torch.nn.MSELoss()
        encoded_next = self.policy.encoder(next_state)
        if self.transfer:
            encoded_next = encoded_next.detach()
        pred_next, pred_rew = self.dynamic_model(self.policy.encoder, state, action, 
                                no_grad_encoder=not self.transfer)
        # print(pred_next, encoded_next)
        # print(rewards, pred_rew)
        return loss_fn(encoded_next, pred_next) + loss_fn(reward, pred_rew.squeeze())

    def policy_loss(self, memory):
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        
        logprobs = self.policy.act_prob(old_states, old_actions, self.device)

        discounted_reward = []
        Gt = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                Gt = 0
            Gt = reward + (self.gamma * Gt)
            discounted_reward.insert(0, Gt)
        
    #    discounted_reward = torch.tensor(discounted_reward)
    #    discounted_reward = (discounted_reward - discounted_reward.mean()) / (discounted_reward.std() + 1e-5)
        # Normalizing the rewards:
        #        rewards = torch.tensor(rewards).to(self.device)
        #        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        policy_gradient = []
        for log_prob, Gt in zip(logprobs, discounted_reward):
            policy_gradient.append(-log_prob * Gt)
        
        return torch.stack(policy_gradient).sum()

    def save_models(self, path):
        torch.save({
            'policy': self.policy.state_dict(),
            'policy_opt': self.optimizer.state_dict(),
            'model': self.dynamic_model.state_dict(),
            'model_opt': self.model_optimizer.state_dict()
            }, path)
    
    def load_models(self, path):
        checkpoint = torch.load(path)
        print("load from ", path)
        self.set_state_dict(checkpoint['policy'], checkpoint['policy_opt'],
                            checkpoint['model'], checkpoint['model_opt'])
    
    def load_dynamics(self, path):
        checkpoint = torch.load(path)
        print("load from ", path)
        self.dynamic_model.load_state_dict(checkpoint['model'])
        for name, param in self.dynamic_model.named_parameters():
            print(name, param)
    # def get_state_dict(self):
    #     return self.policy.state_dict(), self.optimizer.state_dict()
    
    def set_state_dict(self, state_dict, optim, model, model_opt):
        self.policy.load_state_dict(state_dict)
        self.optimizer.load_state_dict(optim)
        self.dynamic_model.load_state_dict(model)
        self.model_optimizer.load_state_dict(model_opt)
        
        