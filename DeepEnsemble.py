'''
Deep Ensemble: Ensemble of DNNs, each of which outputs mean and variance.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical, MixtureSameFamily

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_prob, n_hidden_layers=2):
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
        var = F.softplus(self.var_head(h)) + 1e-6
        return mean, var


def train_ensemble(models, train_X, train_Y, bounds=None, epochs=100, lr=0.01):
    """
    Trains each model in the list independently.
    """
    optimizers = [optim.Adam(m.parameters(), lr=lr) for m in models]
    
    # Gaussian NLL Loss: L = 0.5 * (log(var) + (y - mean)^2 / var)
    loss_fn = nn.GaussianNLLLoss()

    # Normalize inputs for training if bounds provided
    if bounds is not None:
        # Ensure bounds match X's device and dtype
        bounds = bounds.to(train_X.device, dtype=train_X.dtype)
        train_X = (train_X - bounds[0]) / (bounds[1] - bounds[0])
    
    for i, (model, optimizer) in enumerate(zip(models, optimizers)):
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            mean, var = model(train_X)
            # GaussianNLLLoss expects variance, not std dev
            loss = loss_fn(mean, train_Y, var)
            loss.backward()
            optimizer.step()

# Additional code to integrate with BoTorch
from botorch.models.model import Model
from botorch.posteriors import Posterior

class EnsemblePosterior(Posterior):
    """Custom Posterior for Deep Ensemble."""
    def __init__(self, means, variances):
        """
        Args:
            means: [M, batch_shape, 1]
            variances: [M, batch_shape, 1]
        """
        # 1. Moment Matching for BoTorch Acquisition Functions (The part you were missing)
        # E[y] = 1/M * sum(mu)
        self._mean = means.mean(dim=0) 
        
        # Var(y) = 1/M sum(sigma^2 + mu^2) - E[y]^2
        avg_var = variances.mean(dim=0)
        var_of_means = means.var(dim=0, unbiased=False)
        self._variance = avg_var + var_of_means

        # 2. Create Mixture Distribution for Log Likelihood Evaluation
        # Transpose to [batch_shape, M] for distribution components
        # Remove the last output dimension (which is 1)
        means_squeezed = means.squeeze(-1)       # [M, batch_shape...]
        vars_squeezed = variances.squeeze(-1)    # [M, batch_shape...]
        
        # Move the ensemble dimension (0) to the end (-1)
        batch_means = means_squeezed.movedim(0, -1)      # [batch_shape..., M]
        batch_sigmas = vars_squeezed.sqrt().movedim(0, -1) # [batch_shape..., M]
        
        # Create probabilities matching the batch shape
        # shape: [batch_shape..., M]
        batch_shape = batch_means.shape[:-1]
        M = batch_means.shape[-1]
        probs = torch.ones(*batch_shape, M, device=means.device) / M
        
        mix = Categorical(probs=probs)
        comp = Normal(batch_means, batch_sigmas)
        
        # This allows .log_prob() to work correctly on the mixture
        self.distribution = MixtureSameFamily(mix, comp)

    @property
    def mean(self):
        return self._mean
    @property
    def variance(self):
        return self._variance
    @property
    def device(self):
        return self._mean.device
    @property
    def dtype(self):
        return self._mean.dtype
    
    def rsample(self, sample_shape=torch.Size(), base_samples=None):
        # Simple Gaussian sampling approximation for qNIPV compatibility
        # Shape: sample_shape x batch_shape x q x m
        shape = sample_shape + self._mean.shape
        eps = torch.randn(shape, device=self.device, dtype=self.dtype)
        return self._mean + torch.sqrt(self._variance) * eps

class DeepEnsembleWrapper(Model):
    def __init__(self, models, bounds, outcome_stats=None):
        super().__init__()
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
        self._num_outputs = 1

    def forward(self, x):
        # Normalize inputs
        x_norm = (x - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
        return self._predict_individual(x_norm)

    def _predict_individual(self, x):
        """Returns predictions from all models: [M, batch, 1]"""
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

    def posterior(self, X, observation_noise=False, **kwargs):
        # 1. Forward pass
        # X shape: (batch_shape, q, d)
        means, vars = self.forward(X)
        # 2. Un-standardize them back to original scale
        # Mean: mu_real = mu_norm * std + mean
        means = means * self.y_std + self.y_mean
        # Variance: var_real = var_norm * (std^2)
        vars = vars * (self.y_std ** 2)
        # 3. Return EnsemblePosterior
        return EnsemblePosterior(means, vars)
    
    @property
    def num_outputs(self):
        return self._num_outputs