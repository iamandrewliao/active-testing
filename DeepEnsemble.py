'''
Deep Ensemble: Ensemble of DNNs, each of which outputs mean and variance.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical, MixtureSameFamily
from botorch.posteriors.torch import TorchPosterior
from botorch.models.model import Model

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_hidden_layers=3, dropout_prob=0.1):
        super().__init__()
        
        # 1. Start with the input layer
        layers_list = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        ]
        
        # 2. Add n_hidden_layers - 1 additional hidden layers
        # The original code had 2 hidden layers, so we repeat the pattern n_hidden_layers - 1 times
        for _ in range(n_hidden_layers - 1):
            layers_list.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_prob)
            ])
            
        self.layers = nn.Sequential(*layers_list)
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.var_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        h = self.layers(x)
        mean = self.mean_head(h)
        var = F.exp(self.var_head(h))
        return mean, var

def train_ensemble(models, train_X, train_Y, bounds=None, epochs=100, lr=0.01):
    """
    Trains each model in the list independently.
    """
    optimizers = [optim.Adam(m.parameters(), lr=lr) for m in models]
    
    # Gaussian NLL Loss: L = 0.5 * (log(var) + (y - mean)^2 / var)
    loss_fn = nn.GaussianNLLLoss()

    # Standardize targets
    train_Y_standardized = (train_Y - train_Y.mean()) / (train_Y.std() + 1e-6)

    # Normalize inputs for training if bounds provided
    if bounds is not None:
        # Ensure bounds match X's device and dtype
        bounds = bounds.to(train_X)
        train_X = (train_X - bounds[0]) / (bounds[1] - bounds[0])
    
    for i, (model, optimizer) in enumerate(zip(models, optimizers)):
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            mean, var = model(train_X)
            loss = loss_fn(mean, train_Y_standardized, var)
            loss.backward()
            optimizer.step()

# Additional code to integrate with BoTorch

# https://botorch.org/docs/tutorials/custom_model/
class DeepEnsembleWrapper(Model):
    def __init__(self, models, bounds, outcome_stats=None):
        super().__init__()
        models = [m.eval() for m in models]
        self.models = torch.nn.ModuleList(models)
        self.register_buffer("bounds", bounds)
        # Store mean/std buffers for un-standardizing later
        if outcome_stats is not None:
            self.register_buffer("y_mean", outcome_stats[0])
            self.register_buffer("y_std", outcome_stats[1])
        else:
            # Default to no scaling if not provided
            self.register_buffer("y_mean", torch.tensor(0.0))
            self.register_buffer("y_std", torch.tensor(1.0))

    def forward(self, x):
        # Normalize inputs (like input_transform in BoTorch)
        x_norm = (x - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
        return self._predict_ensemble(x_norm)

    def _predict_ensemble(self, x):
        """Returns predictions from all M models: [M, batch, 1]"""
        means_list = []
        vars_list = []
        # No grad needed for posterior/prediction usually
        with torch.no_grad():
            for model in self.models:
                model.eval()
                m, v = model(x)
                means_list.append(m)
                vars_list.append(v)
        return torch.stack(means_list), torch.stack(vars_list)

    def posterior(self, X, **kwargs):
        # 1. Forward pass (includes input normalization)
        # X shape: (batch_shape, q, d)
        means, vars = self.forward(X)

        # 2. Un-standardize outputs back to original scale
        # Mean: mu_real = mu_norm * std + mean
        means = means * self.y_std + self.y_mean
        # Variance: var_real = var_norm * (std^2)
        vars = vars * (self.y_std ** 2)

        # 3. Create the Mixture of Gaussians distribution
        # Current shape: [M, batch_shape, q, 1]
        # We need to move M to the dim expected by MixtureSameFamily (the rightmost batch dim)
        # Target shape for params: [batch_shape, q, M, 1]
        # Move dim 0 (M) to dim -2 (before the event dim '1')
        means = means.movedim(0, -2) 
        vars = vars.movedim(0, -2)
        stds = vars.sqrt()

        # Mixture distribution
        # The mixture is over the M dimension (dim -2)
        # We create a Categorical distribution with uniform weights over M
        batch_shape = means.shape[:-2]  # (batch_shape, q)
        M = means.shape[-2]
        probs = torch.ones(*batch_shape, M, device=X.device, dtype=X.dtype)
        mix = Categorical(probs=probs)

        # Component distribution
        comp = Normal(loc=means, scale=stds)

        # MixtureSameFamily expects the component distribution to have batch_shape matching the mixture distribution. 
        # Mixture Dist Batch Shape: (..., M)
        # Component Dist Batch Shape: (..., M)
        # Note: MixtureSameFamily reduces the M dimension.
        distribution = MixtureSameFamily(mix, comp)

        return TorchPosterior(distribution)