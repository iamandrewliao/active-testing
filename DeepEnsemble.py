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


class _ShapeCompatiblePosterior(TorchPosterior):
    """
    Wrapper for TorchPosterior that ensures mean and variance have shape (..., q, 1)
    for BoTorch compatibility.
    """
    def __init__(self, distribution, q_dim=1):
        super().__init__(distribution)
        self.q_dim = q_dim
    
    @property
    def mean(self):
        """Ensure mean has shape (..., q, 1) for BoTorch compatibility."""
        mean = self.distribution.mean
        # If q_dim > 1, mean should already have q in batch_shape
        # If q_dim == 1, we need to add it back
        if self.q_dim == 1:
            # Add q and output dimensions: (batch,) -> (batch, 1, 1)
            while mean.ndim < 3:
                mean = mean.unsqueeze(-1)
        else:
            # Ensure output dimension: (batch, q) -> (batch, q, 1)
            if mean.ndim == 2:
                mean = mean.unsqueeze(-1)
        return mean
    
    @property
    def variance(self):
        """Ensure variance has shape (..., q, 1) for BoTorch compatibility."""
        variance = self.distribution.variance
        # If q_dim > 1, variance should already have q in batch_shape
        # If q_dim == 1, we need to add it back
        if self.q_dim == 1:
            # Add q and output dimensions: (batch,) -> (batch, 1, 1)
            while variance.ndim < 3:
                variance = variance.unsqueeze(-1)
        else:
            # Ensure output dimension: (batch, q) -> (batch, q, 1)
            if variance.ndim == 2:
                variance = variance.unsqueeze(-1)
        return variance

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
        var = torch.exp(self.var_head(h))
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

    @property
    def num_outputs(self):
        """Number of outputs (always 1 for regression)."""
        return 1

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
        # X shape: (batch_shape, q, d) or (batch_shape, d)
        # Handle both cases: ensure we have (..., q, d) format for processing
        if X.ndim == 2:
            # X is (batch, d), add q dimension: (batch, 1, d)
            X = X.unsqueeze(-2)
            q_dim = 1
        else:
            q_dim = X.shape[-2]
        
        means, vars = self.forward(X)

        # 2. Un-standardize outputs back to original scale
        # Mean: mu_real = mu_norm * std + mean
        means = means * self.y_std + self.y_mean
        # Variance: var_real = var_norm * (std^2)
        vars = vars * (self.y_std ** 2)

        # 3. Create the Mixture of Gaussians distribution
        # Current shape: [M, batch_shape, q, 1]
        # We need to reshape to [batch_shape, q, M, 1] for MixtureSameFamily
        # Move M from dim 0 to dim -2 (before the event dim)
        means = means.movedim(0, -2)  # [M, ..., q, 1] -> [..., q, M, 1]
        vars = vars.movedim(0, -2)
        stds = vars.sqrt()

        # MixtureSameFamily expects:
        # - mixture_distribution.batch_shape = (batch_shape,)
        # - component_distribution.batch_shape = (batch_shape, M)
        # The M dimension should be the rightmost batch dimension in component
        
        # Current shape: [..., q, M, 1]
        # We need to handle q dimension:
        # - If q=1, squeeze it: [..., M, 1]
        # - If q>1, we need to treat each (batch, q) combination separately
        # For simplicity and correctness, let's handle q=1 case (most common)
        # and reshape for q>1 case
        
        q_dim = means.shape[-3]
        if q_dim == 1:
            # Squeeze q dimension: [..., 1, M, 1] -> [..., M, 1]
            means = means.squeeze(-3)
            stds = stds.squeeze(-3)
        else:
            # q > 1: reshape to treat each q as separate batch element
            # [..., q, M, 1] -> [..., q*M, 1] but we want [..., q, M, 1] -> [..., q*M, 1]?
            # Actually, for MixtureSameFamily, we want component batch_shape = (..., q*M)
            # But that doesn't make sense. Let's think differently:
            # We want to create a separate mixture for each q
            # So batch_shape should include q: (..., q) and component should be (..., q, M)
            # But MixtureSameFamily doesn't support this directly.
            # For now, let's flatten: treat each (batch, q) as a separate batch element
            batch_dims = means.shape[:-3]  # everything before q
            q = means.shape[-3]
            M = means.shape[-2]
            # Reshape to [..., q*M, 1] where we treat q*M as the mixture dimension
            means = means.reshape(*batch_dims, q * M, 1)
            stds = stds.reshape(*batch_dims, q * M, 1)
            M = q * M  # Update M to be the combined dimension
        
        # Now means and stds have shape [..., M, 1] where M is the mixture dimension
        # batch_shape is everything except the last 2 dims (M and event)
        batch_shape = means.shape[:-2]
        M = means.shape[-2]
        
        # Squeeze the last dimension (event dim) for Normal distribution
        # Normal expects (batch_shape, M) for loc/scale, not (batch_shape, M, 1)
        means = means.squeeze(-1)  # [..., M, 1] -> [..., M]
        stds = stds.squeeze(-1)    # [..., M, 1] -> [..., M]
        
        # Create mixture distribution with batch_shape and M components
        probs = torch.ones(*batch_shape, M, device=X.device, dtype=X.dtype) / M
        mix = Categorical(probs=probs)

        # Component distribution: Normal with shape [..., M]
        # This will have batch_shape (..., M) and event_shape ()
        comp = Normal(loc=means, scale=stds)

        # MixtureSameFamily: mixture batch_shape should match component batch_shape[:-1]
        # mixture: (batch_shape,)
        # component: (batch_shape, M)
        distribution = MixtureSameFamily(mix, comp)

        # Wrap in a custom posterior that ensures correct shape for BoTorch
        return _ShapeCompatiblePosterior(distribution, q_dim=q_dim)