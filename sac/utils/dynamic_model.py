import random
import torch
import torch.nn as nn
from torch.optim import Adam

class ProbabilisticTransitionModel(nn.Module):

    def __init__(self, encoder_feature_dim, action_shape, announce=True, 
                       max_sigma=1e1, min_sigma=1e-4, lr=1e-3, env_linear=False):
        super().__init__()
        layer_width = encoder_feature_dim
        self.env_linear = env_linear
        if not env_linear:
            self.fc = nn.Linear(encoder_feature_dim, encoder_feature_dim)
            self.ln = nn.LayerNorm(encoder_feature_dim)
        self.fc_mu = nn.Linear(encoder_feature_dim, encoder_feature_dim)
        self.fc_sigma = nn.Linear(encoder_feature_dim, encoder_feature_dim)
        self.reward_net = nn.Sequential(
            nn.Linear(encoder_feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1))
        self.optimizer = Adam(self.parameters(), lr)

        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        assert(self.max_sigma >= self.min_sigma)
        if announce:
            print("Probabilistic transition model chosen.")

    def forward(self, x):
        if not self.env_linear:
            x = self.fc(x)
            x = self.ln(x)
            x = torch.relu(x)
        mu = self.fc_mu(x)
        sigma = torch.sigmoid(self.fc_sigma(x))  # range (0, 1.)
        sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma  # scaled range (min_sigma, max_sigma)
        return mu, sigma

    def sample_prediction(self, x):
        mu, sigma = self(x)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps
    
    def reward_prediction(self, x):
        return self.reward_net(x)
        
    
    def save(self, save_dir):
        torch.save(self.state_dict(), save_dir)