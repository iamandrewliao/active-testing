'''
MLP that takes in factor values and returns Mixture of Gaussians distribution parameters
(exactly like https://arxiv.org/pdf/2502.09829)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, MixtureSameFamily
from botorch.posteriors.torch import TorchPosterior
from botorch.models.model import Model
from botorch.posteriors.posterior import Posterior


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
        sigma = torch.exp(sigma_raw)
        return pi, mu, sigma

# Wrap MDN model as a BoTorch model
class MDNWrapper(Model): # Inherit from BoTorch Model
    def __init__(self, model, bounds, outcome_stats=None, use_mc_dropout=False, num_mc_samples=10):
        """
        Args:
            model: The MDN model
            bounds: Input bounds for normalization
            outcome_stats: Tuple of (y_mean, y_std) for un-standardization
            use_mc_dropout: If True, use MC dropout at test time (as in paper)
            num_mc_samples: Number of MC dropout samples (default 10 as in paper)
        """
        super().__init__()
        self.model = model
        self.use_mc_dropout = use_mc_dropout
        self.num_mc_samples = num_mc_samples
        self.register_buffer("bounds", bounds)
        if outcome_stats is not None:
            self.register_buffer("y_mean", outcome_stats[0])
            self.register_buffer("y_std", outcome_stats[1])
        else:
            self.register_buffer("y_mean", torch.tensor(0.0))
            self.register_buffer("y_std", torch.tensor(1.0))
        
        # Set model to eval mode (dropout will be enabled manually for MC sampling)
        if not use_mc_dropout:
            model.eval()

    @property
    def num_outputs(self):
        """Number of outputs (always 1 for regression)."""
        return 1

    def forward(self, x, enable_dropout=False):
        """
        Forward pass with optional dropout enabled for MC sampling.
        
        Args:
            x: Input tensor
            enable_dropout: If True, enable dropout layers (for MC dropout)
        """
        # Normalize inputs automatically (like input_transform in BoTorch)
        x_norm = (x - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
        
        if enable_dropout:
            # Enable dropout for MC sampling
            self.model.train()
            # Set all dropout layers to training mode
            for module in self.model.modules():
                if isinstance(module, nn.Dropout):
                    module.train()
        else:
            self.model.eval()
            
        return self.model(x_norm)

    def _create_mixture_distribution(self, pi, mu, sigma, keep_q_dim=False):
        """
        Helper to create MixtureSameFamily distribution from parameters.
        
        Args:
            pi: Mixing coefficients [batch_shape, q, n_components]
            mu: Means [batch_shape, q, n_components]
            sigma: Standard deviations [batch_shape, q, n_components]
            keep_q_dim: If True, keep q dimension in batch_shape (for BoTorch compatibility)
        """
        # Handle q dimension
        q_dim = pi.shape[-2] if pi.ndim >= 2 else 1
        
        if keep_q_dim and q_dim > 1:
            # Keep q in batch_shape: [batch, q, n_components]
            # This creates batch_shape (batch, q) and component batch_shape (batch, q, n_components)
            pi = pi / (pi.sum(dim=-1, keepdim=True) + 1e-8)
            mix = Categorical(probs=pi)
            comp = Normal(loc=mu, scale=sigma)
            distribution = MixtureSameFamily(mix, comp)
        else:
            # Squeeze q dimension if q=1 or if not keeping it
            if q_dim == 1:
                pi = pi.squeeze(-2)
                mu = mu.squeeze(-2)
                sigma = sigma.squeeze(-2)
            
            # Mixture Distribution (Categorical)
            # probs shape: [batch_shape, n_components]
            # Normalize pi to ensure it's a valid probability distribution
            pi = pi / (pi.sum(dim=-1, keepdim=True) + 1e-8)
            mix = Categorical(probs=pi)
            
            # Component Distribution (Normal)
            # Normal expects loc/scale with shape [batch_shape, n_components]
            # This will create batch_shape (batch_shape, n_components) and event_shape ()
            comp = Normal(loc=mu, scale=sigma)
            
            # Mixture of Gaussians
            # mixture: (batch_shape,)
            # component: (batch_shape, n_components)
            distribution = MixtureSameFamily(mix, comp)
        
        return distribution

    def _sample_mc_dropout_distributions(self, X):
        """
        Sample multiple distributions using MC dropout.
        Returns individual samples for computing expected conditional entropy.
        
        Returns:
            individual_dists: List of MixtureSameFamily distributions (one per MC sample)
            combined_dist: Combined mixture distribution (for marginal entropy)
        """
        pi_samples = []
        mu_samples = []
        sigma_samples = []
        
        with torch.no_grad():  # No gradient needed for inference
            for _ in range(self.num_mc_samples):
                pi, mu, sigma = self.forward(X, enable_dropout=True)
                pi_samples.append(pi)
                mu_samples.append(mu)
                sigma_samples.append(sigma)
        
        # Stack samples: [num_mc_samples, batch_shape, q, n_components]
        pi_stack = torch.stack(pi_samples, dim=0)
        mu_stack = torch.stack(mu_samples, dim=0)
        sigma_stack = torch.stack(sigma_samples, dim=0)
        
        # Un-standardize
        mu_stack = mu_stack * self.y_std + self.y_mean
        sigma_stack = sigma_stack * self.y_std
        
        # Create individual distributions for conditional entropy computation
        individual_dists = []
        for i in range(self.num_mc_samples):
            dist = self._create_mixture_distribution(
                pi_stack[i], mu_stack[i], sigma_stack[i]
            )
            individual_dists.append(dist)
        
        # Create combined distribution for marginal entropy
        # Combine all MC samples into one large mixture
        batch_shape = pi_stack.shape[1:-2]  # everything except (q, n_components)
        q_dim = pi_stack.shape[-2]
        n_components = pi_stack.shape[-1]
        
        # Flatten MC samples and components: [batch, q, num_mc_samples * n_components]
        pi_combined = pi_stack.permute(1, 2, 0, 3).reshape(*batch_shape, q_dim, self.num_mc_samples * n_components)
        mu_combined = mu_stack.permute(1, 2, 0, 3).reshape(*batch_shape, q_dim, self.num_mc_samples * n_components)
        sigma_combined = sigma_stack.permute(1, 2, 0, 3).reshape(*batch_shape, q_dim, self.num_mc_samples * n_components)
        
        # Normalize mixing coefficients across all components
        pi_combined = pi_combined / (pi_combined.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Create combined distribution
        combined_dist = self._create_mixture_distribution(pi_combined, mu_combined, sigma_combined)
        
        return individual_dists, combined_dist

    def posterior(self, X, **kwargs):
        """
        Compute posterior distribution.
        If use_mc_dropout is True, samples multiple distributions using MC dropout
        and returns a mixture of them (as in the paper).
        """
        # Handle both (batch, d) and (batch, q, d) input shapes
        original_ndim = X.ndim
        if X.ndim == 2:
            # X is (batch, d), add q dimension: (batch, 1, d)
            X = X.unsqueeze(-2)
            q_dim = 1
        else:
            q_dim = X.shape[-2]
        
        if self.use_mc_dropout:
            # MC Dropout: Sample multiple distributions (as in paper)
            # Paper uses 10 samples
            _, combined_dist = self._sample_mc_dropout_distributions(X)
            distribution = combined_dist
        else:
            # Standard forward pass (no MC dropout)
            pi, mu, sigma = self.forward(X, enable_dropout=False)
            
            # Un-standardize
            mu = mu * self.y_std + self.y_mean
            sigma = sigma * self.y_std
            
            # Create the Mixture of Gaussians Distribution
            # For BoTorch compatibility, we'll handle q_dim in the wrapper
            distribution = self._create_mixture_distribution(pi, mu, sigma, keep_q_dim=False)
        
        # Wrap in a custom posterior that ensures correct shape for BoTorch
        return _ShapeCompatiblePosterior(distribution, q_dim=q_dim)

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