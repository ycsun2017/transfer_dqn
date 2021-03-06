import numpy as np
import torch
from torch.optim import Adam
from copy import deepcopy
import itertools
from gym.spaces import Box, Discrete
from collections import deque
from transfer_ppo.utils.ppo_core import *

class PPO(nn.Module):
    def __init__(self, observation_space, action_space, feature_size, hidden_units=64, encoder_layers=2, 
                 value_layers = 2, model_layers=1, policy_layers=0, clip_ratio=0.2, activation=nn.Tanh, 
                 gamma=0.99, lam = 0.97, device=torch.device("cpu"),
                train_pi_iters=80, train_v_iters=80, train_dynamic_iters=5, target_kl=0.01,
                transfer=False, no_detach=False, pi_lr=3e-4, vf_lr=1e-3, model_lr=1e-3, coeff=1.0, 
                delta=True, obs_only=False, env=None, disable_encoder=False, source_aux=False, act_encoder=False, alpha=1.0):
        super(PPO, self).__init__()
        self.cont = isinstance(action_space, Box)
        self.ac = MLPActorCritic(observation_space, action_space, encoder_hidden_sizes=[hidden_units]*encoder_layers, 
                                 value_hidden_sizes = [hidden_units]*value_layers, 
                                 policy_layers = [hidden_units]*policy_layers,
                                 feature_size = feature_size, no_grad_encoder=(not no_detach), model_lr=model_lr, 
                                 model_hidden_sizes=[hidden_units]*model_layers, obs_only=obs_only,
                                 env=env, disable_encoder=disable_encoder, source_aux=source_aux, act_encoder=act_encoder).to(Param.device)
        self.var_counts = tuple(count_vars(module) for module in [self.ac.pi, self.ac.v])
        self.feature_size = feature_size
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
        self.source_aux = source_aux
        self.alpha=alpha
        
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
    ### If transfer or source_aux is True, the loss is computed by
    ### policy_loss + self.coeff * (reward_loss+obs_loss)
    def update_policy(self, data, transfer=False, source_aux=False, op_memory=None):
        pi_l_old, pi_info_old, act_info = self.compute_loss_pi(data)
        pi_l_old = pi_l_old.item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info, act_info = self.compute_loss_pi(data)
            kl = pi_info['kl']
            if kl > 1.5 * self.target_kl:
                #print('Early stopping at step %d due to reaching max kl.'%i)
                break
            if not (transfer or source_aux):
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
        if (not self.transfer) or self.source_aux:
            pi_info = self.update_policy(data, transfer=False, source_aux=False)  ### Source Task
        else:
            pi_info = self.update_policy(data, transfer=True, source_aux=self.source_aux, op_memory=op_memory) ### Target Task
        v_info = self.update_v(data)
        
        ### In source task, we update the environment model
        ### In target task, we only compute the loss without taking any gradient step
        data_statistics = op_memory.get_statistics(cont=self.cont)
        if not self.transfer:
            ### Source Task, update environment models
            if not self.obs_only:
                results = self.ac.dynamic_model.update(self.ac.pi.encoder, op_memory, data_statistics, self.train_dynamic_iters)
                loss_obs, loss_reward = results[0], results[1]
                return pi_info, v_info, loss_obs, loss_reward
            else:
                results = self.ac.dynamic_model.update(self.ac.pi.encoder, op_memory, data_statistics, self.train_dynamic_iters)
                loss_obs = results[0]
                return pi_info, v_info, loss_obs
        else:
            data = op_memory.sample()
            ### Target Task, only compute the loss without doing any updates
            if not self.obs_only:
                results = self.ac.dynamic_model.compute_loss(self.ac.pi.encoder, data, data_statistics)
                loss_obs, loss_reward = results[0], results[1]
                return pi_info, v_info, loss_obs, loss_reward
            else:
                results  = self.ac.dynamic_model.compute_loss(self.ac.pi.encoder, data, data_statistics)
                loss_obs = results[0]
                return pi_info, v_info, loss_obs
            
    def save(self, path, state_dict=None):
        self.ac.save(path, state_dict)
        
    def load_models(self, path, load_buffer=False):
        self.ac.load_models(path, load_buffer=load_buffer)
            
    def load_dynamics(self, path, load_buffer=False):
        self.ac.load_dynamics(path, load_buffer=load_buffer)
        
    def step(self, obs):
        return self.ac.step(obs)

    def act(self, obs):
        return self.ac.act(obs)
    
    def pretrain_env_model(self, env, num_t=50, max_ep_len=1000, batch_size=256):
        self.obs     = deque(maxlen = num_t*max_ep_len)
        self.acs     = deque(maxlen = num_t*max_ep_len)
        self.n_obs   = deque(maxlen = num_t*max_ep_len)
        self.rewards = deque(maxlen = num_t*max_ep_len)
        if (Param.logger_file is not None):
            print("--Collecting Trajectories--")
        
        ### Collecting trajectories from random policies
        for t in range(num_t):
            obs, n_obs = env.reset(), None
            for step in range(max_ep_len):
                if self.cont:
                    a = np.random.uniform(env.action_space.low, env.action_space.high)
                else:
                    a = np.random.randint(env.action_space.n)
                n_obs, r, done, _ = env.step(a)
                self.obs.append(obs)
                self.acs.append(a)
                self.n_obs.append(n_obs)
                self.rewards.append(r)
                obs = n_obs
                if done:
                    break
        
        ### Training the reward and transition model
        shuffle      = np.random.permutation(len(self.obs))
        self.obs     = np.asarray(self.obs)[shuffle]
        self.n_obs   = np.asarray(self.n_obs)[shuffle]
        self.acs     = np.asarray(self.acs)[shuffle]
        self.rewards = np.asarray(self.rewards)[shuffle]
        
        data_statistics = None
        if self.cont:
            data_statistics = {
                            'obs_mean': np.mean(np.asarray(self.obs), axis=0),
                            'obs_std':  np.std(np.asarray(self.obs), axis=0),
                            'acs_mean': np.mean(np.asarray(self.acs), axis=0),
                            'acs_std':  np.std(np.asarray(self.acs), axis=0),
                            }
        else:
            data_statistics = {
                            'obs_mean': np.mean(np.asarray(self.obs), axis=0),
                            'obs_std':  np.std(np.asarray(self.obs), axis=0)
                            }
        target_buffer = Target_Buffer(self.obs, self.acs, self.n_obs, self.rewards)
        
        optimizer = Adam(self.ac.pi.encoder.parameters(), lr=0.0001)
        for i in range(500):
            losses = self.ac.dynamic_model.compute_loss(self.ac.pi.encoder, None, data_statistics, buffer=target_buffer, detach=False)
            loss    = sum(losses)
            #loss    = losses[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #if i%50 == 0:
                #print('Obs Loss:{}, Reward Loss:{}'.format(losses[1], losses[0])) 
                #total_norm = 0.
                #for p in self.ac.pi.encoder.parameters():
                    #param_norm = p.grad.detach().data.norm(2)
                    #total_norm += param_norm.item() ** 2
                #total_norm = total_norm ** (1. / 2)
                #print("Total Norm:{}".format(total_norm))
        
        self.ac.dynamic_model.pretrian_reward(self.ac.pi.encoder, target_buffer, data_statistics, 
                                              optimizer, epochs=100, alpha=self.alpha)
        
        if Param.logger_file is not None:
            if not self.obs_only:        
                print('Obs Loss:{}, Reward Loss:{}'.format(losses[1], losses[0])) 
                Param.logger_file.write('Obs Loss:{}, Reward Loss:{}\n'.format(losses[1], losses[0])) 
            else:
                Param.logger_file.write('Obs Loss:{}\n'.format(losses[0])) 
                

class Target_Buffer:
    
    ### Store a pool of encoded source states
    def __init__(self, obs, acs, n_obs, rewards):
        self.obs     = obs
        self.acs     = acs
        self.n_obs   = n_obs
        self.rewards = rewards
        
    def sample(self, batch_size):
        index = np.random.choice(self.obs.shape[0], 
                                 batch_size, replace=False)
        return self.obs[index], self.acs[index], self.n_obs[index], self.rewards[index]
    
    def __len__(self):
        return self.obs.shape[0]
       
    

            
            
            