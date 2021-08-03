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
                 cont=False, normalize=True, no_grad_encoder=True, source_aux=False,
                 delta=True, obs_only=False, env=None, act_encoder=False,
                 classifier=False, source_buffer=None):
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
        self.classifier = None
        if classifier:
            self.classifier = mlp([feature_size]+hidden_sizes+[2], nn.Tanh)
            self.classifier.to(Param.device)
            self.classifier_optimizer = Adam(
                self.classifier.parameters(),
                self.learning_rate
            )
            self.source_buffer = source_buffer
            
        ### If delta is true, only predict s_{t+1}-s_{t} instead of s_{t+1}
        self.delta = delta
        self.cont = cont 
        self.normalize = normalize
        self.no_grad_encoder = no_grad_encoder
        self.obs_only = obs_only
        self.source_aux = source_aux
        self.act_encoder = act_encoder
        if act_encoder:
            ### Use our two linear model
            self.r_encoder = mlp([act_dim]+[hidden_sizes[0]]+[feature_size], nn.Tanh, output_activation=nn.Tanh)
            self.s_encoder = mlp([act_dim]+[hidden_sizes[0]]+[feature_size], nn.Tanh, output_activation=nn.Tanh)
            self.s_predict_net = mlp([feature_size, feature_size], nn.Identity)
            self.r_predict_net = mlp([feature_size, 1], nn.Identity)
            self.encoder_optimizer = Adam(
                list(self.s_encoder.parameters()) + 
                list(self.r_encoder.parameters()) +
                list(self.s_predcit_net.parameters()) + 
                list(self.r_predict_net.parameters()),
                self.learning_rate
            )
            
        
            
    def forward(self, encoder, obs_normalized, acs_unnormalized,
                data_statistics):
        
        obs_normalized = from_numpy(obs_normalized)
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
        
        ### compute next state prediction
        if self.act_encoder:
            r_act = self.r_encoder(acs_normalized)
            s_act = self.s_encoder(s_encoder)
            delta_pred = self.s_predict_net(feature_vec*s_act)
        else:
            concatenated_input = torch.cat([feature_vec, acs_normalized], dim=1)
            if self.no_grad_encoder and (not self.source_aux):
                concatenated_input.detach()
            
            # Unnormalize prediction
            delta_pred  = self.delta_network(concatenated_input)
        
        ### Convert the torch tensor to numpy array
        if self.delta:
            next_obs_pred    = feature_vec.cpu().detach().numpy() + delta_pred.cpu().detach().numpy()
        else:
            next_obs_pred    = delta_pred.cpu().detach().numpy()
        
        if not self.obs_only:
            if self.act_encoder:
                reward_pred = self.r_predict_net(feature_vec*r_act)
            else:
                reward_pred = self.reward_network(concatenated_input)
            next_reward_pred = reward_pred.cpu().detach().numpy()
            return next_obs_pred, next_reward_pred, reward_pred, delta_pred
        else:
            return next_obs_pred, delta_pred
    
    def compute_loss(self, encoder, data, data_statistics):
        observations, actions, next_observations, rewards = data
        
        if self.classifier
        
        ### Normalize the observation
        observations_normalize      = (observations - data_statistics['obs_mean'])/(data_statistics['obs_std']+1e-6)
        next_observations_normalize = (next_observations - data_statistics['obs_mean'])/(data_statistics['obs_std']+1e-6)
        
        ### Compute ground truth next state prediction
        if self.delta:
            ### Target delta: s_{t+1} - s_{t}
            delta_target = (encoder(from_numpy(next_observations_normalize))-encoder(from_numpy(observations_normalize))).detach()
            ### Predicted delta
        else:
            delta_target = encoder(from_numpy(next_observations_normalize))
        
        ### Predict next state (and reward if obs_only is False)
        ### Compute the MSE prediction loss
        if not self.obs_only:
            _,_, reward_pred, delta_pred  = self.forward(encoder, observations_normalize, actions, data_statistics)
            ### compute the loss of reward and transition models
            loss_obs    = self.loss(delta_pred, delta_target)
            loss_reward = self.loss(reward_pred, from_numpy(rewards).unsqueeze(1))
            return loss_reward, loss_obs    
        else:
            _,  delta_pred  = self.forward(encoder, observations_normalize, actions, data_statistics)
            loss_obs          = self.loss(delta_pred, delta_target)
            return loss_obs
        
        
        
    def update(self, encoder, op_memory, data_statistics, train_dynamic_iters):
        #log_model_predictions(self.env, self.feature_size, encoder, self,
                              #data_statistics,  itr='{}_{}'.format(self.epoch, 0), 
                              #cont=self.cont, log_dir=Param.plot_dir)
        for i in range(train_dynamic_iters):
            data = op_memory.sample()
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
                
        #log_model_predictions(self.env, self.feature_size, encoder, self, 
                              #data_statistics, itr='{}_{}'.format(self.epoch, train_dynamic_iters), 
                              #cont=self.cont,  log_dir=Param.plot_dir)
        #self.epoch += 1
        return loss_obs.item() if self.obs_only else (loss_obs.item(), loss_reward.item())
    
    def train_classifier(self, encoder, source_memory, target_memory, 
                         batch_size=64, train_iter=100):
        loss_fn, loss = torch.nn.CrossEntropyLoss(), None
        for i in range(train_iter):
            source_batch = source_memory.sample(batch_size/2)
            target_batch = to_numpy(target_memory.sample(batch_size/2)[0])
            train_batch  = torch.vstack((source_batch, target_batch)
            shuffle      = torch.randperm(batch_size)
            train_batch  = train_batch[shuffle]
            train_label  = torch.hstack((torch.zeros(batch_size/2),
                                       torch.ones(batch_size/2))[shuffle]
                                        
            loss         = loss_fn(self.classifier_network(train_batch),
                                   train_label)
            
            self.classifier_optimizer.zero_grad()
            loss.backward()
            self.classifier_optimizer.step()
        return loss
    
    def classifier_loss(self, source_encoding, target_encoding):
        loss_fn = torch.nn.CrossEntropyLoss()
        label   = torch.hstack((torch.zeros(source_encoding.shape[0]),
                                torch.ones(target_encoding..shape[1])
                        ))
        batch   = torch.vstack(source_encoding, target_encoding)
        logits  = self.classifier_network(batch)
        return loss_fn(logits, batch)
    
            
            
        
    
def calculate_mean_prediction_error(env, encoder, action_sequence, dyn_model, data_statistics):
    true_states, pred_states, obs = [], [], env.reset()
    normalize  = lambda x: (x - data_statistics['obs_mean'])/(data_statistics['obs_std']+1e-6)
    unnormalize  = lambda x: x*data_statistics['obs_std']+data_statistics['obs_mean']
    obs_p = np.copy(obs)
    for i in range(200):
        low, high = env.action_space.low, env.action_space.high
        obs, _, _, _ = env.step(np.random.uniform(low, high,(env.action_space.shape[0],)))
    for a in action_sequence:
        true_states.append(obs_p)
        obs_p,_,_,_ = env.step(a)
    count = 0
    for a in action_sequence:
        pred_states.append(obs)
        obs = dyn_model(encoder, np.expand_dims(normalize(obs),axis=0),  np.expand_dims(a,axis=0), data_statistics)[0]
        obs = unnormalize(obs)
        obs = np.squeeze(obs,axis=0)
        count += 1
        
    true_states = np.squeeze(true_states)
    pred_states = np.squeeze(pred_states)
    
    # compute mean prediction error
    loss = lambda a, b: np.mean((a-b)**2)
    mpe = loss(pred_states, true_states)
    
    return mpe, true_states, pred_states

def log_model_predictions(env, feature_size, encoder, dyn_model, 
                          data_statistics,  itr, cont=False, 
                          horizon=20, k=14, log_dir='plot/'):
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

    # plot the predictions
    fig.clf()
    counter = 1
    obs_dim = 2*(obs_dim//2)
    dimensions = np.random.choice(obs_dim, size=(k, ), replace=False)
    #for i in range(obs_dim):
    for i in dimensions:
        #plt.subplot(obs_dim/2, 2, counter)
        plt.subplot(k/2, 2, counter)
        counter += 1
        plt.plot(true_states[:,i], 'g')
        plt.plot(pred_states[:,i], 'r')
    fig.suptitle('MPE: ' + str(mpe))
    figname = log_dir+'/itr_'+str(itr)+'_predictions.png' if itr is not None else log_dir + '/prediction{}.png'.format(horizon)
    fig.savefig(figname, dpi=200, bbox_inches='tight')