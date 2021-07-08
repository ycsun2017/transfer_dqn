import torch
import numpy as np
import copy
from collections import deque
from transfer_ppo.utils.param import Param
from transfer_ppo.utils.ppo_core import from_numpy, MovingMeanStd
import scipy

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf  = np.zeros(combined_shape(size, obs_dim))
        self.act_buf  = np.zeros(combined_shape(size, act_dim))
        self.adv_buf  = np.zeros(size)
        self.rew_buf  = np.zeros(size)
        self.ret_buf  = np.zeros(size)
        self.val_buf  = np.zeros(size)
        self.logp_buf = torch.zeros(size).to(Param.device).type(Param.dtype)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        def discount_cumsum(x, discount):
            return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    
    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf, 0), np.std(self.adv_buf, 0)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=from_numpy(self.obs_buf), act=from_numpy(self.act_buf), ret=from_numpy(self.ret_buf),
                    adv=from_numpy(self.adv_buf), logp=self.logp_buf)
        return data

class ModelReplayBuffer:
    
    def __init__(self, obs_dim, act_dim, max_size=100000):

        self.max_size = max_size
        self.obs     = deque(maxlen = max_size)
        self.acs     = deque(maxlen = max_size)
        self.n_obs   = deque(maxlen = max_size)
        self.rewards = deque(maxlen = max_size)
        self.counter = 0
    
    def store(self, obs, acs, n_obs, rewards, cont=False, noise=False):
        if noise:
            if cont:
                acs = add_noise(acs)
            obs, n_obs = add_noise(obs), add_noise(n_obs)
        self.obs.append(obs)
        self.acs.append(acs)
        self.n_obs.append(n_obs)
        self.rewards.append(rewards)
        self.counter += 1
    
    def get_statistics(self, cont=False):
        if cont:
            return ({
                'obs_mean': np.mean(np.asarray(self.obs), axis=0),
                'obs_std': np.std(np.asarray(self.obs), axis=0),
                'acs_mean': np.mean(np.asarray(self.acs), axis=0),
                'acs_std': np.std(np.asarray(self.acs), axis=0),
            })
        else:
            return ({
                'obs_mean': np.mean(np.asarray(self.obs), axis=0),
                'obs_std': np.std(np.asarray(self.obs), axis=0),
            })
        
    def sample(self, batch_size=256):
        idx = np.random.choice(min(self.counter, self.max_size), size=batch_size)
        states, actions, nexts, rewards = [], [], [], []
        for i in idx:
            states.append(self.obs[i])
            actions.append(self.acs[i])
            nexts.append(self.n_obs[i])
            rewards.append(self.rewards[i])
        return np.asarray(states), np.asarray(actions), np.asarray(nexts), np.asarray(rewards)
    
    def __len__(self):
        return self.counter
        
def add_noise(data_inp, noiseToSignal=0.01):

    data = copy.deepcopy(data_inp) #(num data points, dim)
    mean_data = np.mean(data, axis=0)
    #if mean is 0,
    #make it 0.001 to avoid 0 issues later for dividing by std
    data[data == 0] = 0.000001
    #width of normal distribution to sample noise from
    #larger magnitude number = could have larger magnitude noise
    std_of_noise = data * noiseToSignal
    for j in range(data.shape[0]):
        data[j] = np.copy(data[j] + np.random.normal(
            0, np.absolute(std_of_noise[j])))

    return data