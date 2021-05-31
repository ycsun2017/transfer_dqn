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
from experiment.utils.param import Param
import matplotlib.pyplot as plt
import copy



class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
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

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class SACActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()
    def save(self, log_dir= r'./learned_models/', model_name='sac_policy'):
        torch.save(self.state_dict(), os.path.join(log_dir,model_name))


def rollout(agent, env, num_steps=1000, render=False):
    total_rew = 0
    o = env.reset()
    for t in range(num_steps):
        a = agent.act(torch.as_tensor(o, dtype=torch.float32))
        (o, reward, done, _info) = env.step(a)
        total_rew += reward
        if render: 
            env.render()
        if done: break
    return total_rew, t+1

def perform_actions(env, actions):
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    for ac in actions:
        obs.append(ob)
        acs.append(ac)
        ob, rew, done, _ = env.step(ac)
        # add the observation after taking a step to next_obs
        next_obs.append(ob)
        rewards.append(rew)
        steps += 1
        # If the episode ended, the corresponding terminal value is 1
        # otherwise, it is 0
        if done:
            terminals.append(1)
            break
        else:
            terminals.append(0)

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)

def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}

def calculate_mean_prediction_error(env, action_sequence, model, data_statistics):
    
    # true
    true_states = perform_actions(env, action_sequence)['observation']

    # predicted
    ob = np.expand_dims(true_states[0],0)
    pred_states = []
    for ac in action_sequence:
        pred_states.append(ob)
        action = np.expand_dims(ac,0)
        #model.eval()
        ob = model.get_prediction(ob, action, data_statistics)
    pred_states = np.squeeze(pred_states)

    # mpe
    mean_squared_error = lambda a, b: np.mean((a-b)**2)
    mpe = mean_squared_error(pred_states, true_states)

    return mpe, true_states, pred_states


def log_model_predictions(env, dyn_model, data_statistics, itr, log_dir='./results', horizon=20):
    # model predictions
    fig = plt.figure()
    
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    # calculate and log model prediction error
    low, high = env.action_space.low, env.action_space.high
    action_sequence = np.random.uniform(low, high,(horizon, ac_dim))
    mpe, true_states, pred_states = calculate_mean_prediction_error(env, action_sequence, dyn_model, data_statistics)
    assert ob_dim == true_states.shape[1] == pred_states.shape[1]
    ob_dim = 2*int(ob_dim/2.0) ## skip last state for plotting when state dim is odd

    # plot the predictions
    fig.clf()
    for i in range(ob_dim):
        plt.subplot(ob_dim/2, 2, i+1)
        plt.plot(true_states[:,i], 'g')
        plt.plot(pred_states[:,i], 'r')
    fig.suptitle('MPE: ' + str(mpe))
    figname = log_dir+'/itr_'+str(itr)+'_predictions.png' if itr is not None else log_dir + '/prediction{}.png'.format(horizon)
    fig.savefig(figname, dpi=200, bbox_inches='tight')





def from_numpy(arr):
    return torch.tensor(arr, requires_grad=False).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


def collect_trajectories(env, policies, size, local_steps_per_epoch=4000):
    '''
        Using the policy to collect transition dynamics
    '''
    print('-----------------Collecting Trajectories------------------------')
    counter = 0
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    obs  =  np.zeros((size, obs_dim), dtype=np.float32)
    acs  =  np.zeros((size, act_dim), dtype=np.float32)
    n_obs = np.zeros((size, obs_dim), dtype=np.float32)
    while (counter < size):
        o = env.reset()
        for t in range(local_steps_per_epoch):
            policy = random.choice(policies)
            a, _, _ = policy.step(torch.as_tensor(o, dtype=torch.float32))
            next_o, _, _, _ = env.step(a)
            obs[counter]   = o
            n_obs[counter] = next_o
            acs[counter]   = a
            o = next_o
            
            counter+=1
            if (counter>=size):
                break
                
    return obs, acs, n_obs
    

class ModelReplayBuffer:
    
    def __init__(self, obs_dim, act_dim, max_size=1000000):

        self.max_size = max_size
        self.obs = np.zeros((max_size, obs_dim))
        self.acs = np.zeros((max_size, act_dim))
        self.n_obs = np.zeros((max_size, obs_dim))
        self.counter = 0
    
    def add_to_buffer(self, obs, acs, n_obs, noise=True):
        size = obs.shape[0]
        if (noise):
            obs, acs, n_obs = add_noise(obs), add_noise(acs), add_noise(n_obs)
        self.obs[self.counter : self.counter+size]  = obs
        self.acs[self.counter : self.counter+size]  = acs
        self.n_obs[self.counter : self.counter+size] = n_obs
        self.counter += size
        
        
def add_noise(data_inp, noiseToSignal=0.01):

    data = copy.deepcopy(data_inp) #(num data points, dim)

    #mean of data
    mean_data = np.mean(data, axis=0)

    #if mean is 0,
    #make it 0.001 to avoid 0 issues later for dividing by std
    mean_data[mean_data == 0] = 0.000001

    #width of normal distribution to sample noise from
    #larger magnitude number = could have larger magnitude noise
    std_of_noise = mean_data * noiseToSignal
    for j in range(mean_data.shape[0]):
        data[:, j] = np.copy(data[:, j] + np.random.normal(
            0, np.absolute(std_of_noise[j]), (data.shape[0],)))

    return data

############### Implementation for Random Network Distillation (RND) ####################
### Helper_method for RND
def init_method_1(model):
    model.weight.data.uniform_()
    model.bias.data.uniform_()

def init_method_2(model):
    model.weight.data.normal_()
    model.bias.data.normal_()
    
### Implement Random Network Distillation for Exploration
### Checkout this paper for more details: https://arxiv.org/pdf/1810.12894.pdf
###
class RND(object):
    def __init__(self, params):
        self.target  = build_mlp(params.input_size, params.output_size, \
                                 params.n_layers,params.size, nn.ReLU(), 
                                 init_method=init_method_1).to(Param.device)
        self.predict = build_mlp(params.input_size, params.output_size, \
                                 params.n_layers,params.size, nn.ReLU(), 
                                 init_method=init_method_2).to(Param.device)
        self.optimizer = torch.optim.Adam(self.predict.parameters(),1.0)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                            self.optimizer,\
                            params.learning_rate_schedule)
    ### Given a state (torch tensor), output its intrinsic rewards
    def get_intrinsic_rewards(self, obs_batch):
        target = self.target(obs_batch)
        prediction = self.predict(obs_batch)
        loss = F.mse_loss(target, prediction)
        int_reward = loss.detach()
        ### Optimize the prediction network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        normalize = lambda x: (x-torch.mean(x))/torch.std(x)
        return normalize(int_reward)

################ RND Agent #######################
class DQN_RND_Agent(nn.Module):
    def __init__(self, q_func, params, learning_rate= 5e-4, test=True, doubleQ=False):
        super(DQN_RND_Agent, self).__init__()
        self.exploit_Q = q_func().to(Param.device).type(Param.dtype)
        self.exploit_target_Q = q_func().to(Param.device).type(Param.dtype)
        self.explore_Q = q_func().to(Param.device).type(Param.dtype)
        self.explore_target_Q = q_func().to(Param.device).type(Param.dtype)
        self.optimizer_exploit = torch.optim.Adam(self.exploit_Q.parameters(), lr=learning_rate)
        self.optimizer_explore = torch.optim.Adam(self.explore_Q.parameters(), lr=learning_rate)
        self.exploration_schedule = params.exploration_schedule
        self.test = test
        self.doubleQ = doubleQ
        self.RND = RND(params)
        self.steps = 0
        self.exploration = True
        
    def update(self, obs_batch, act_batch, rew_batch,\
               next_obs_batch, not_done_mask, gamma, tau):
        ### Optimize Exploitation Network
        exploit_current_Q_values = self.exploit_Q(obs_batch).gather(1, act_batch)
        if self.doubleQ:
            exploit_indices = self.exploit_Q(next_obs_batch).max(1)[-1].unsqueeze(1)
            exploit_next_max_q = self.exploit_target_Q(next_obs_batch)
            exploit_next_max_q = exploit_next_max_q.gather(1, exploit_indices)
        else:
            exploit_next_max_q = (self.exploit_target_Q(next_obs_batch).detach().max(1)[0]).unsqueeze(1)
        exploit_next_Q_values = not_done_mask * exploit_next_max_q
        exploit_target_Q_values = rew_batch + (gamma * exploit_next_Q_values)
        loss_exploit = F.smooth_l1_loss(exploit_current_Q_values, exploit_target_Q_values)
        self.optimizer_exploit.zero_grad()
        loss_exploit.backward()
        self.optimizer_exploit.step()
        
        
        ### compute intrinsic reward
        int_reward = self.RND.get_intrinsic_rewards(obs_batch)
        alpha = self.exploration_schedule(self.steps)
        
        if (self.steps % 2500 ==0 and self.exploration):
            print("Exploration Weight:{}".format(alpha))
        ### Exploration has ended, so no need to update Exploration Network
        if (alpha < 1e-3):
            self.soft_update(tau)
            self.exploration = False
            self.steps += 1
            return
        ### Combine the intrinsic and extrinsic reward
        combined_reward = alpha*int_reward + (1-alpha)*rew_batch
        
        ### Optimize Exploration Network
        explore_current_Q_values = self.explore_Q(obs_batch).gather(1, act_batch)
        if self.doubleQ:
            explore_indices = self.explore_Q(next_obs_batch).max(1)[-1].unsqueeze(1)
            explore_next_max_q = self.explore_target_Q(next_obs_batch)
            explore_next_max_q = explore_next_max_q.gather(1, explore_indices)
        else:
            explore_next_max_q = (self.explore_target_Q(next_obs_batch).detach().max(1)[0]).unsqueeze(1)
        explore_next_Q_values = not_done_mask * explore_next_max_q
        explore_target_Q_values = combined_reward + (gamma * explore_next_Q_values)
    
        ### Update Exploration Q network
        loss_explore = F.smooth_l1_loss(explore_current_Q_values, explore_target_Q_values)
        self.optimizer_explore.zero_grad()
        loss_explore.backward()
        self.optimizer_explore.step()
        
        ### Soft Update the Target Network  
        self.soft_update(tau)
        
        self.steps += 1
    
    
    def soft_update(self, tau=1e-3):
        """Soft update model parameters"""
        for target_param, local_param in zip(self.exploit_target_Q.parameters(), 
                                             self.exploit_Q.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
        if (self.exploration):
            for target_param, local_param in zip(self.explore_target_Q.parameters(),
                                                 self.explore_Q.parameters()):
                target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def select_epilson_greedy_action(self, obs, t, exploration, num_actions):
        sample = random.random()
        eps_threshold = exploration.value(t)
        if sample > eps_threshold:
            obs = torch.from_numpy(obs).to(Param.device).type(Param.dtype).unsqueeze(0)
            if (not self.test):
                 obs /= 255.0  ### Normalize in Atari Games
            with torch.no_grad():
                if self.exploration:
                    return int(self.explore_Q(obs).data.max(1)[1].cpu())
                else:
                    return int(self.exploit_Q(obs).data.max(1)[1].cpu())
        else:
            with torch.no_grad():
                return random.randrange(num_actions)
   
    ### Choose the best action, happened during test time
    def step(self,obs):
        with torch.no_grad():
            obs = torch.from_numpy(obs).to(Param.device).type(Param.dtype).unsqueeze(0)
            return self.exploit_Q(obs).data.max(1)[1].cpu()[0]
    
    def save(self,log_dir=r'./learned_models/', exp_name='dqn'):
        torch.save(self.state_dict(), os.path.join(log_dir, exp_name))