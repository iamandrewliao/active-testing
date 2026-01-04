'''
MLP that takes in factor values and returns Mixture of Gaussians distribution parameters
(exactly like https://arxiv.org/pdf/2502.09829)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, MixtureSameFamily
from botorch.posteriors import Posterior
from botorch.posteriors.torch import TorchPosterior
from botorch.models.model import Model
from botorch.acquisition.acquisition import AcquisitionFunction
import math

OUTCOME_RANGE = (0, 4)

# Mixture of Gaussians MLP (Mixture Density Network)
class MDN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_components, n_hidden_layers=3, dropout_prob=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.n_components = n_components
        
        # 1. Start with the input layer
        layers_list = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        ]
        
        # 2. Add n_hidden_layers - 1 additional hidden layers
        for _ in range(n_hidden_layers - 1):
            layers_list.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_prob)
            ])
            
        self.feature_extractor = nn.Sequential(*layers_list)
        
        self.pi_head = nn.Linear(hidden_dim, n_components) # mixing coefficients
        self.sigma_head = nn.Linear(hidden_dim, n_components) # std devs
        self.mu_head = nn.Linear(hidden_dim, n_components) # means

    def forward(self, x):
        features = self.feature_extractor(x)
        pi_logits = self.pi_head(features)
        mu = self.mu_head(features)
        sigma_raw = self.sigma_head(features)
        
        pi = F.softmax(pi_logits, dim=-1)
        sigma = F.exp(sigma_raw)
        return pi, mu, sigma

# Wrap MDN model as a BoTorch model
class MDNWrapper(Model): # Inherit from BoTorch Model
    def __init__(self, model, bounds, outcome_stats=None):
        super().__init__()
        model.eval()
        self.model = model
        self.register_buffer("bounds", bounds)
        if outcome_stats is not None:
            self.register_buffer("y_mean", outcome_stats[0])
            self.register_buffer("y_std", outcome_stats[1])
        else:
            self.register_buffer("y_mean", torch.tensor(0.0))
            self.register_buffer("y_std", torch.tensor(1.0))

    def forward(self, x):
        # Normalize inputs automatically (like input_transform in BoTorch)
        x_norm = (x - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
        return self.model(x_norm)

    def posterior(self, X, **kwargs):
        # 1. Forward pass
        pi, mu, sigma = self.forward(X)
        
        # 2. Un-standardize
        mu = mu * self.y_std + self.y_mean
        sigma = sigma * self.y_std
        
        # 3. Create the Mixture of Gaussians Distribution
        # Current shapes:
        # pi: [batch_shape, q, n_components]
        # mu: [batch_shape, q, n_components]
        # sigma: [batch_shape, q, n_components]
        
        # We need the resulting sample to be [batch_shape, q, 1]
        # So we treat the '1' as the event dimension.
        # We unsqueeze mu and sigma to [batch_shape, q, n_components, 1]
        mu = mu.unsqueeze(-1)
        sigma = sigma.unsqueeze(-1)
        
        # Mixture Distribution (Categorical)
        # probs shape: [batch_shape, q, n_components]
        mix = Categorical(probs=pi)
        
        # Component Distribution (Normal)
        # loc/scale shape: [batch_shape, q, n_components, 1]
        # The 'n_components' dim is the mixture dimension (rightmost batch dim)
        comp = Normal(loc=mu, scale=sigma)
        
        # Mixture of Gaussians
        distribution = MixtureSameFamily(mix, comp)
        
        return TorchPosterior(distribution)

# MDN Helper Functions
def mdn_loss(pi, mu, sigma, target):
    """
    Computes the Negative Log Likelihood (NLL) for a Mixture Density Network.
    
    Args:
        pi: [batch_size, n_components] mixing coefficients
        mu: [batch_size, n_components] means
        sigma: [batch_size, n_components] standard deviations
        target: [batch_size, 1] or [batch_size] target values
        
    Returns:
        loss: Scalar (mean NLL)
    """
    # Ensure target has correct shape for broadcasting: [batch, 1]
    if target.dim() == 1:
        target = target.unsqueeze(-1)
        
    # 1. Create component distributions
    # Normal expects mu and sigma. 
    # We use Normal.log_prob to get log(N(y | mu, sigma))
    # Shape: [batch, n_components]
    dist = Normal(loc=mu, scale=sigma)
    log_probs_components = dist.log_prob(target)
    
    # 2. Calculate log-likelihood of the mixture
    # log( sum_k( pi_k * N_k ) )
    # = log( sum_k( exp( log(pi_k) + log(N_k) ) ) )
    # = LogSumExp( log(pi_k) + log_prob_component )
    # Add epsilon to pi for numerical stability inside log
    log_pi = torch.log(pi + 1e-6)
    # Calculate argument for LogSumExp
    # Shape: [batch, n_components]
    weighted_log_probs = log_pi + log_probs_components
    
    # 3. Apply LogSumExp across the component dimension (dim=1)
    # Shape: [batch, 1]
    log_likelihood = torch.logsumexp(weighted_log_probs, dim=1)
    
    # 4. Return Negative Mean Log Likelihood
    return -torch.mean(log_likelihood)

def train_mdn(model, train_X, train_Y, bounds=None, epochs=100, lr=0.01):
    """
    Trains the MDN model. 
    This handles normalization internally so the raw model learns on scaled data [0, 1].
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 1. Calculate stats for standardization
    y_mean = train_Y.mean()
    y_std = train_Y.std()
    
    # 2. Standardize Targets
    train_Y_standardized = (train_Y - y_mean) / (y_std + 1e-6)

    # 3. Normalize Inputs
    if bounds is not None:
        bounds = bounds.to(train_X)
        train_X = (train_X - bounds[0]) / (bounds[1] - bounds[0])

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Forward pass on raw model with normalized data
        pi, mu, sigma = model(train_X)
        loss = mdn_loss(pi, mu, sigma, train_Y_standardized)
        loss.backward()
        optimizer.step()
    model.eval()