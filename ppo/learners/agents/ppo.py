import sys
import torch  
import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from gym.spaces import Box, Discrete
from .model import EncodedActorCritic, ContActorCritic, DynamicModel
from .updates import ppo_update
from torch.distributions import Categorical

class PPO(nn.Module):
    def __init__(self, state_space, action_space, feature_size, hidden_units=64, encoder_layers=2, model_layers=0,
                K_epochs=4, eps_clip=0.2, activation=nn.Tanh, learning_rate=3e-4, gamma=0.9, device="cpu", 
                action_std=0.5, transfer=False, share_encoder=False, use_model="both", no_detach=False):
        super(PPO, self).__init__()
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.transfer = transfer
        self.share = share_encoder
        self.use_model = use_model
        self.no_detach = no_detach
        # deal with 1d state input
        state_dim = state_space.shape[0]
        
        if isinstance(action_space, Discrete):
            self.action_dim = action_space.n
            self.policy = EncodedActorCritic(state_dim, self.action_dim, feature_size, 
                        activation, hidden_units, encoder_layers, share_encoder=self.share).to(self.device)
        # elif isinstance(action_space, Box):
        #     self.action_dim = action_space.shape[0]
        #     self.policy = ContActorCritic(state_dim, self.action_dim, hidden_sizes, activation, action_std, self.device).to(self.device)
        self.dynamic_model = DynamicModel(feature_size, self.action_dim, hidden_units, model_layers).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.model_optimizer = optim.Adam(self.dynamic_model.parameters(), lr=learning_rate)
        
        print("policy", self.policy)
        print("model", self.dynamic_model)
        self.loss_fn = nn.MSELoss()
    
    def act(self, state):
        return self.policy.act(state, self.device)
        
    
    def update_policy(self, memory, op_memory, coeff=1.0):
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
#             print(coeff)
            self.optimizer.zero_grad()
            p_loss = self.policy_loss(memory)
            m_loss = self.model_loss(op_memory)
            (p_loss + coeff * m_loss).backward()
            self.optimizer.step()

        return p_loss, m_loss
        
    def model_loss_single(self, encoder, state, action, next_state, reward):
        loss_fn = torch.nn.MSELoss()
        encoded_next = encoder(next_state)
        if not self.no_detach and self.transfer:
            encoded_next = encoded_next.detach()
        pred_next, pred_rew = self.dynamic_model(encoder, state, action, 
                                no_grad_encoder=not self.transfer)
        
        return loss_fn(encoded_next, pred_next) + loss_fn(reward, pred_rew.squeeze())

    def model_loss(self, memory):
        states, actions, next_states, rewards = memory.sample(256)
        state = torch.stack(states).to(self.device).detach()
        action = torch.stack(actions).to(self.device).detach()
        next_state = torch.stack(next_states).to(self.device).detach()
        reward = torch.FloatTensor(rewards).to(self.device).detach()

        if self.use_model == "both":
            loss_1 = self.model_loss_single(self.policy.encoder, state, action, next_state, reward)
            loss_2 = self.model_loss_single(self.policy.critic_encoder, state, action, next_state, reward)
            return loss_1 + loss_2
        elif self.use_model == "actor":
            return self.model_loss_single(self.policy.encoder, state, action, next_state, reward)
        elif self.use_model == "critic":
            return self.model_loss_single(self.policy.critic_encoder, state, action, next_state, reward)

    def policy_loss(self, memory):
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()

        discounted_reward = []
        Gt = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                Gt = 0
            Gt = reward + (self.gamma * Gt)
            discounted_reward.insert(0, Gt)
            
        discounted_reward = torch.tensor(discounted_reward).to(self.device)
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            new_logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(new_logprobs - old_logprobs.detach())
                
            # Finding Surrogate Loss:
            advantages = discounted_reward - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.loss_fn(state_values, discounted_reward) \
                - 0.01*dist_entropy
        
        return loss.mean()

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