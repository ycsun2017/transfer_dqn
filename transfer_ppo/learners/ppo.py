import numpy as np
import torch
from torch.optim import Adam
from copy import deepcopy
import itertools
from gym.spaces import Box, Discrete
from transfer_ppo.utils.ppo_core import *

class PPO(nn.Module):
    def __init__(self, observation_space, action_space, feature_size, hidden_units=64, encoder_layers=2, 
                 value_layers = 2, model_layers=1, clip_ratio=0.2, activation=nn.Tanh, 
                 gamma=0.99, lam = 0.97, device=torch.device("cpu"),
                train_pi_iters=80, train_v_iters=80, train_dynamic_iters=5, target_kl=0.01,
                transfer=False, no_detach=False, pi_lr=3e-4, vf_lr=1e-3, model_lr=1e-3, coeff=1.0, 
                delta=True, obs_only=False, env=None):
        super(PPO, self).__init__()
        self.cont = isinstance(action_space, Box)
        self.ac = MLPActorCritic(observation_space, action_space, policy_hidden_sizes=[hidden_units]*encoder_layers, 
                                 value_hidden_sizes = [hidden_units]*value_layers,
                                 feature_size = feature_size, no_grad_encoder=(not no_detach), model_lr=model_lr, 
                                 model_hidden_sizes=[hidden_units]*model_layers, obs_only=obs_only,
                                 env=env).to(Param.device)
        self.var_counts = tuple(count_vars(module) for module in [self.ac.pi, self.ac.v])
        self.transfer = transfer
        self.gamma = gamma
        self.lam = lam
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.train_dynamic_iters = train_dynamic_iters
        self.target_kl = target_kl
        self.clip_ratio = clip_ratio
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(),  lr=vf_lr)
        self.coeff = coeff
        self.obs_only = obs_only
        
    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        obs = self.ac.normalize(obs)
        act_info = dict(act_mean=torch.mean(act), act_std=torch.std(act))
        pi, logp = self.ac.pi(obs, act)

        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = clipped.type(Param.dtype).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
        return loss_pi, pi_info, act_info

    def compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        obs = self.ac.normalize(obs)
        return ((self.ac.v(obs) - ret)**2).mean()
        
    ### Update Policy Network
    ### If Transfer is True, the loss is compputed by
    ### policy_loss + self.coeff * (reward_loss+obs_loss)
    def update_policy(self, data, transfer=False, op_memory=None):
        pi_l_old, pi_info_old, act_info = self.compute_loss_pi(data)
        pi_l_old = pi_l_old.item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info, act_info = self.compute_loss_pi(data)
            kl = pi_info['kl']
            if kl > 1.5 * self.target_kl:
                print('Early stopping at step %d due to reaching max kl.'%i)
                break
            if not transfer:
                loss_pi.backward()
            else:
                batch, data_statistics = op_memory.sample(), op_memory.get_statistics(cont=self.cont)
                if self.obs_only:
                    loss_obs = self.ac.dynamic_model.compute_loss(self.ac.pi.encoder, batch, data_statistics)
                    loss = loss_pi + self.coeff*loss_obs
                else:
                    loss_reward, loss_obs = self.ac.dynamic_model.compute_loss(self.ac.pi.encoder, batch, data_statistics)
                    loss = loss_pi + self.coeff*(loss_reward+loss_obs)
                loss.backward()
            self.pi_optimizer.step()
        stop_iter = i
        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        act_mean, act_std = act_info['act_mean'], act_info['act_std']
        return dict(LossPi=pi_l_old, KL=kl, Entropy=ent, ClipFrac=cf,
                    DeltaLossPi=(loss_pi.item() - pi_l_old),
                    StopIter=stop_iter, Act_Mean=act_mean, Act_Std=act_std)
        
    def update_v(self, data):
        v_l_old = self.compute_loss_v(data).item()
        # Value function learning
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            self.vf_optimizer.step()
        return dict(LossV = v_l_old, DeltaLossV=(loss_v.item() - v_l_old))
            
            
    def update(self, memory, op_memory):
        data = memory.get()
        
        if not self.transfer:
            pi_info = self.update_policy(data, transfer=False)  ### Source Task
        else:
            pi_info = self.update_policy(data, transfer=True, op_memory=op_memory) ### Target Task
        v_info = self.update_v(data)
        
        ### In source task, we update the environment model
        ### In target task, we only compute the loss without taking any gradient step
        batch = op_memory.sample()
        data_statistics = op_memory.get_statistics(cont=self.cont)
        if not self.transfer:
            ### Source Task, update environment models
            if not self.obs_only:
                loss_obs, loss_reward = self.ac.dynamic_model.update(self.ac.pi.encoder, batch, data_statistics, self.train_dynamic_iters)
                return pi_info, v_info, loss_obs, loss_reward
            else:
                loss_obs = self.ac.dynamic_model.update(self.ac.pi.encoder, batch, data_statistics, self.train_dynamic_iters)
                return pi_info, v_info, loss_obs
        else:
            ### Target Task, only compute the loss
            if not self.obs_only:
                loss_obs, loss_reward = self.ac.dynamic_model.compute_loss(self.ac.pi.encoder, batch, data_statistics)
                return pi_info, v_info, loss_obs, loss_reward
            else:
                loss_obs = self.ac.dynamic_model.compute_loss(self.ac.pi.encoder, batch, data_statistics)
                return pi_info, v_info, loss_obs
            
    def save(self, path):
        self.ac.save(path)
        
    def load_models(self, path):
        self.ac.load_models(path)
            
    def load_dynamics(self, path):
        self.ac.load_dynamics(path)
        
    def step(self, obs):
        return self.ac.step(obs)

    def act(self, obs):
        return self.ac.act(obs)