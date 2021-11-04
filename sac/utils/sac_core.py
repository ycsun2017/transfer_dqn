import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.optim import Adam
from utils.param import Param
import os

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SACReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: from_numpy(v) for k,v in batch.items()}
    

def from_numpy(n_array, dtype=None):
    if dtype is None:
        return torch.from_numpy(n_array).to(Param.device).type(Param.dtype)
    else:
        return torch.from_numpy(n_array).to(Param.device).type(dtype)
    
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

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

    
class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.s_encoder   = LatentEncoder(obs_dim, list(hidden_sizes[:-1]), activation)
        self.mu_layer  = mlp([hidden_sizes[-2], hidden_sizes[-1], act_dim], activation, nn.Identity)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.s_encoder(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class LatentEncoder(nn.Module):
    def __init__(self, input_dim=8, hidden_sizes=[256,256], activation=nn.ReLU, output_activation=nn.Identity):
        super().__init__()
        self.feature_size = hidden_sizes
        self.latent = mlp([input_dim]+hidden_sizes, activation=activation, output_activation=output_activation)
    
    def forward(self, x):
        latent_vec = self.latent(x)
        return latent_vec/torch.linalg.norm(latent_vec, dim=-1).unsqueeze(-1)
    
class MLPQFunction(nn.Module):

    def __init__(self, a_encoder, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.a_encoder = a_encoder
        self.s_encoder = LatentEncoder(obs_dim, list(hidden_sizes[:-1]), activation)
        self.q = mlp([hidden_sizes[-2], hidden_sizes[-1],1], activation)

    def forward(self, obs, act):
        latent_vec = self.s_encoder(obs)*self.a_encoder(act)
        q = self.q(latent_vec)
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.a_encoder = mlp([act_dim, 64, hidden_sizes[-2]], activation, nn.Tanh)
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(self.a_encoder, obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(self.a_encoder, obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.cpu().numpy()
    
    def save(self, log_dir= Param.model_dir, model_name='sac_policy'):
        state_dict = {
            "a_encoder": self.a_encoder.state_dict(),
            "pi.s_encoder": self.pi.s_encoder.state_dict(),
            "pi.mu_layer": self.pi.mu_layer.state_dict(),
            "pi.log_std_layer": self.pi.log_std_layer.state_dict(),
            "q1.s_encoder": self.q1.s_encoder.state_dict(),
            "q2.s_encoder": self.q2.s_encoder.state_dict(),
            "q1.q": self.q1.q.state_dict(),
            "q2.q": self.q2.q.state_dict()
        }
        torch.save(state_dict, os.path.join(log_dir,model_name))
       
    def load_all(self, load_dir):
        state_dict = torch.load(load_dir, map_location=Param.device)
        self.a_encoder.load_state_dict(state_dict["a_encoder"])
        self.pi.s_encoder.load_state_dict(state_dict["pi.s_encoder"]) 
        self.pi.mu_layer.load_state_dict(state_dict["pi.mu_layer"]) 
        self.pi.log_std_layer.load_state_dict(state_dict["pi.log_std_layer"]) 
        self.q1.q.load_state_dict(state_dict["q1.q"])
        self.q1.s_encoder.load_state_dict(state_dict["q1.s_encoder"])
        self.q2.q.load_state_dict(state_dict["q2.q"])
        self.q2.s_encoder.load_state_dict(state_dict["q2.s_encoder"])
        
    
    def load(self, load_dir):
        state_dict = torch.load(load_dir, map_location=Param.device)
        self.a_encoder.load_state_dict(state_dict["a_encoder"])
        self.pi.mu_layer.load_state_dict(state_dict["pi.mu_layer"]) 
        self.pi.log_std_layer.load_state_dict(state_dict["pi.log_std_layer"]) 
        self.q1.q.load_state_dict(state_dict["q1.q"])
        self.q2.q.load_state_dict(state_dict["q2.q"])
        

def compute_model_loss(buffer, transition_model, encoders, batch_size=256):
    s_encoder, a_encoder = encoders
    data = buffer.sample_batch(batch_size)
    obs, n_obs, act, reward = data["obs"], data["obs2"], data["act"], data["rew"]
    
    ### Compute Latent Encodings
    h = s_encoder(obs)
    a = a_encoder(act)
    latent_vec = h*a
    
    ### Compute Predicted Next Hidden Feature Vector
    pred_next_latent_mu, pred_next_latent_sigma = transition_model(latent_vec)
    if pred_next_latent_sigma is None:
        pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)

    next_h = s_encoder(n_obs)
    diff = (pred_next_latent_mu - next_h.detach()) / pred_next_latent_sigma
    loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma))

    pred_next_reward = transition_model.reward_prediction(latent_vec)
    reward_loss = F.mse_loss(pred_next_reward, reward.unsqueeze(-1))
    total_loss = loss + reward_loss
    loss_info = {
        "Loss P": loss,
        "Loss R": reward_loss
    }
    return total_loss, loss_info
    

def update_model(buffer, transition_model, encoders, batch_size=256, epochs=100):
    s_encoder, a_encoder = encoders
    for i in range(epochs):
        ### Sample a batch fromothe buffer
        data = buffer.sample_batch(batch_size)
        obs, n_obs, act, reward = data["obs"], data["obs2"], data["act"], data["rew"] 
        
        ### Compute Latent Encodings
        h = s_encoder(obs)
        a = a_encoder(act)
        latent_vec = h*a
        
        ### Compute Predicted Next Hidden Feature Vector
        pred_next_latent_mu, pred_next_latent_sigma = transition_model(latent_vec)

        ### Compute the loss
        next_h = s_encoder(n_obs)
        diff = (pred_next_latent_mu - next_h.detach()) / pred_next_latent_sigma
        loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma))
        
        pred_next_reward = transition_model.reward_prediction(latent_vec)
        reward_loss = F.mse_loss(pred_next_reward, reward.unsqueeze(-1))
        total_loss = loss + reward_loss
        
        transition_model.optimizer.zero_grad()
        total_loss.backward()
        transition_model.optimizer.step()
    loss_info = {
        "Loss P": loss,
        "Loss R": reward_loss
    }
    return loss_info

def pretrain_encoder(env, buffer, encoders, transition_model, num_t=10, epochs=1000):
    ### Collect Trajectories from the environment by randomly sampling actions
    for t in range(num_t):
        o = env.reset()
        while (True):
            a = env.action_space.sample()
            o2, r, d, _ = env.step(a)
            buffer.store(o, a, r, o2, d)
            o = o2
            if d:
                break
    
    ### Optimize Encoder to minimize predictino error
    optimizer = Adam(encoders[0].parameters(), lr=1e-4)
    for i in range(epochs):
        loss, loss_info = compute_model_loss(buffer, transition_model, encoders, batch_size=256)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return loss_info
    
    
    
    

        