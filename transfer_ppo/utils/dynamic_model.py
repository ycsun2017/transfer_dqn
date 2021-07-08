import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from transfer_ppo.utils.param import  Param
import matplotlib.pyplot as plt

def from_numpy(n_array, dtype=None):
    if dtype is None:
        return torch.from_numpy(n_array).to(Param.device).type(Param.dtype)
    else:
        return torch.from_numpy(n_array).to(Param.device).type(dtype)
    
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)    
 
class Dynamic_Model(nn.Module):

    def __init__(self, feature_size, act_dim, hidden_sizes=(), lr=1e-5, 
                 cont=False, normalize=True, no_grad_encoder=True,
                 delta=True, obs_only=False, env=None):
        super(Dynamic_Model, self).__init__()
        
        self.env = env
        self.epoch = 0
        self.learning_rate = lr
        self.act_dim = act_dim
        self.feature_size = feature_size
        self.delta_network = mlp([feature_size + act_dim] + hidden_sizes +[feature_size], nn.Tanh)
        self.delta_network.to(Param.device)
        self.delta_optimizer = Adam(
            self.delta_network.parameters(),
            self.learning_rate,
        )
        if not obs_only:
            self.reward_network = mlp([feature_size + act_dim] + hidden_sizes +[1], nn.Tanh)
            self.reward_network.to(Param.device)
            self.reward_optimizer = Adam(
                self.reward_network.parameters(),
                self.learning_rate,
            )
        self.loss = nn.MSELoss()
        ### If delta is true, only predict s_{t+1}-s_{t} instead of s_{t+1}
        self.delta = delta
        self.cont = cont 
        self.normalize = normalize
        self.no_grad_encoder = no_grad_encoder
        self.obs_only = obs_only
        
            
    def forward(self, encoder, obs_unnormalized, acs_unnormalized,
                data_statistics):
        if self.normalize:
            obs_normalized =  from_numpy((obs_unnormalized - data_statistics['obs_mean'])/data_statistics['obs_std'])
        else:
            obs_normalized = from_numpy(obs_unnormalized)
        feature_vec = encoder(obs_normalized).detach()
        
        # normalize input action if action space is continuous
        if self.cont:
            if self.normalize:
                acs_normalized =  from_numpy((acs_unnormalized - data_statistics['acs_mean'])/data_statistics['acs_std'])
            else:
                acs_normalized = from_numpy(acs_unnormalized)
        else:
            convert = lambda acs: torch.tensor(acs).to(Param.device).type(torch.int64)
            acs_normalized =  torch.nn.functional.one_hot(convert(acs_unnormalized), self.act_dim).float() 
        concatenated_input = torch.cat([feature_vec, acs_normalized], dim=1)
        if self.no_grad_encoder:
            concatenated_input.detach()
            
        # Unnormalize prediction
        delta_pred  = self.delta_network(concatenated_input)
        
        ### Convert the torch tensor to numpy array
        if self.delta:
            next_obs_pred    = feature_vec.cpu().detach().numpy() + delta_pred.cpu().detach().numpy()
        else:
            next_obs_pred    = delta_pred.cpu().detach().numpy()
        
        if not self.obs_only:
            reward_pred = self.reward_network(concatenated_input)
            next_reward_pred = reward_pred.cpu().detach().numpy()
            return next_obs_pred, next_reward_pred, reward_pred, delta_pred
        else:
            return next_obs_pred, delta_pred

    ### Compute s_{t+1} and r_{t}
    def get_prediction(self, encoder, obs, acs, data_statistics):
        if not self.obs_only:
            next_obs_pred, reward_pred, _,_ ,_ = self.forward(obs, acs, data_statistics)
            return next_obs_pred, reward_pred
        else:
            next_obs_pred, _ ,_                = self.forward(obs, acs, data_statistics)
            return next_obs_pred
    
    def compute_loss(self, encoder, data, data_statistics):
        observations, actions, next_observations, rewards = data
        if self.normalize:
            observations      = (observations - data_statistics['obs_mean'])/data_statistics['obs_std']
            next_observations = (next_observations - data_statistics['obs_mean'])/data_statistics['obs_std']
        
        if self.delta:
            ### Target delta: s_{t+1} - s_{t}
            delta_target = (encoder(from_numpy(next_observations))-encoder(from_numpy(observations))).detach()
            ### Predicted delta
        else:
            delta_target = encoder(from_numpy(next_observations))
            
        if not self.obs_only:
            _,_, reward_pred, delta_pred  = self.forward(encoder, observations, actions, data_statistics)
            ### compute the loss of reward and transition models
            loss_obs    = self.loss(delta_pred, delta_target)
            loss_reward = self.loss(reward_pred, from_numpy(rewards).unsqueeze(1))
            return loss_reward, loss_obs
        else:
            _,  delta_pred  = self.forward(encoder, observations, actions, data_statistics)
            loss_obs          = self.loss(delta_pred, delta_target)
            return loss_obs
    
    def update(self, encoder, data, data_statistics, train_dynamic_iters):
        log_model_predictions(self.env, self.feature_size, encoder, self,
                              data_statistics,  itr='{}_{}'.format(self.epoch, 0), 
                              cont=self.cont, log_dir=Param.plot_dir)
        for i in range(train_dynamic_iters):
            if self.obs_only:
                loss_obs = self.compute_loss(encoder, data, data_statistics)
            else:
                loss_reward, loss_obs = self.compute_loss(encoder, data, data_statistics)

            self.delta_optimizer.zero_grad() 
            loss_obs.backward()
            self.delta_optimizer.step()
            if not self.obs_only:
                self.reward_optimizer.zero_grad() 
                loss_reward.backward()
                self.reward_optimizer.step()
        log_model_predictions(self.env, self.feature_size, encoder, self, 
                              data_statistics, itr='{}_{}'.format(self.epoch, train_dynamic_iters), 
                              cont=self.cont,  log_dir=Param.plot_dir)
        self.epoch += 1
        return loss_obs.item() if self.obs_only else (loss_obs.item(), loss_reward.item())
    
    
    
def calculate_mean_prediction_error(env, encoder, action_sequence, dyn_model, data_statistics):
    true_states, pred_states, obs = [], [], env.reset()
    for a in action_sequence:
        results = dyn_model(encoder, np.expand_dims(obs,axis=0), [a], data_statistics)
        pred_state = results[0]
        normalize  = lambda x: (x - data_statistics['obs_mean'])/data_statistics['obs_std']
        true_state = encoder(from_numpy(normalize(env.step(a)[0]))).cpu().numpy()
        true_states.append(true_state)
        pred_states.append(pred_state)
        obs,_,_,_ = env.step(a)
        
    true_states = np.squeeze(true_states)
    pred_states = np.squeeze(pred_states)
    
    # compute mean prediction error
    mean_squared_error = lambda a, b: np.mean((a-b)**2)
    mpe = mean_squared_error(pred_states, true_states)
    
    return mpe, true_states, pred_states

def log_model_predictions(env, feature_size, encoder, dyn_model, 
                          data_statistics,  itr, cont=False, 
                          horizon=20, k=8, log_dir='plot/'):
    # model predictions
    fig = plt.figure()
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0] if cont else env.action_space.n

    if cont:
        # calculate and log model prediction error
        low, high = env.action_space.low, env.action_space.high
        action_sequence = np.random.uniform(low, high,(horizon, act_dim))
    else:
        action_sequence = np.random.randint(act_dim, size=(horizon,))
    mpe, true_states, pred_states = calculate_mean_prediction_error(env, encoder, action_sequence, dyn_model, data_statistics)
    dimensions = np.random.choice(feature_size, size=(k, ), replace=False)

    # plot the predictions
    fig.clf()
    counter = 1
    for i in dimensions:
        plt.subplot(k/2, 2, counter)
        counter += 1
        plt.plot(true_states[:,i], 'g')
        plt.plot(pred_states[:,i], 'r')
    fig.suptitle('MPE: ' + str(mpe))
    figname = log_dir+'/itr_'+str(itr)+'_predictions.png' if itr is not None else log_dir + '/prediction{}.png'.format(horizon)
    fig.savefig(figname, dpi=200, bbox_inches='tight')