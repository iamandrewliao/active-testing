'''
MLP that takes in factor values and returns Mixture of Gaussians distribution parameters
(exactly like https://arxiv.org/pdf/2502.09829)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, MixtureSameFamily
from botorch.posteriors import Posterior
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

class MDNPosterior(Posterior):  # for BoTorch compatibility
    def __init__(self, pi, mu, sigma):
        """
        Args:
            pi: [batch, K] (mixing weights)
            mu: [batch, K] (means)
            sigma: [batch, K] (std devs)
        """
        self.pi = pi
        self.mu = mu
        self.sigma = sigma
        
        # 1. Create distribution for log likelihood
        mix = Categorical(probs=pi)
        comp = Normal(mu, sigma)
        self.distribution = MixtureSameFamily(mix, comp)

        # 2. Get mean and variance of the mixture
        # Mixture mean = sum(pi * mu)
        mixture_mean = torch.sum(pi * mu, dim=-1, keepdim=True)
        
        # Mixture variance = mean of variances + variance of mean = E[X^2]-E[X]^2
        # variance of a single component = sigma**2 + mu**2
        # mean of variances = sum(pi * (single component variance))
        mean_of_vars = torch.sum(pi * (sigma**2 + mu**2), dim=-1, keepdim=True)
        # variance of mean = mixture_mean**2
        var_of_mean = mixture_mean**2
        mixture_var = mean_of_vars - var_of_mean
        
        self._mean = mixture_mean
        self._variance = mixture_var

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
        # Approximate sampling using the single Gaussian moment-matched statistics
        shape = sample_shape + self._mean.shape
        eps = torch.randn(shape, device=self.device, dtype=self.dtype)
        return self._mean + torch.sqrt(self._variance) * eps

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
        # self._num_outputs = 1 # Required by BoTorch

    def forward(self, x):
        # Normalize inputs automatically (like input_transform in BoTorch)
        x_norm = (x - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
        return self.model(x_norm)

    # @property
    # def num_outputs(self):
    #     return self._num_outputs

    def posterior(self, X, **kwargs):
        # 1. Forward pass
        pi, mu, sigma = self.forward(X)
        # 2. Un-standardize
        # pi (mixing weights) do not change.
        # mu_real = mu_norm * std + mean
        mu = mu * self.y_std + self.y_mean
        # sigma_real = sigma_norm * std (Note: sigma is std dev, not variance)
        sigma = sigma * self.y_std
        # 3. Return MDN Posterior (which handles distribution creation)
        return MDNPosterior(pi, mu, sigma)

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


# class MDN_BALD(AcquisitionFunction):
#     def __init__(self, model, num_dropout_samples=10, num_bins=25, outcome_range=OUTCOME_RANGE):
#         """
#         Implements BALD for MDN using MC Dropout and discretization.
        
#         Args:
#             model: The MDNWrapper instance.
#             num_dropout_samples: T=10 samples as per paper.
#             num_bins: bins to discretize outcome range for entropy estimation as per paper.
#             outcome_range: min/max of Y to create the discretized empirical distribution.
#         """
#         super().__init__(model)
#         self.num_dropout_samples = num_dropout_samples
        
#         # Discretize empirical distribution (outcome range) for entropy calculation 
#         # We create a fixed grid over the outcome range, shape: [num_bins]
#         self.outcome_range = outcome_range
#         self.num_bins = num_bins
#         tkwargs = {"device": model.bounds.device, "dtype": model.bounds.dtype}
#         y_discrete = torch.linspace(outcome_range[0], outcome_range[1], num_bins, **tkwargs)
#         self.register_buffer("y_discrete", y_discrete)
#         self.delta_y = self.y_discrete[1] - self.y_discrete[0]

#     def forward(self, X):
#         """
#         Compute BALD score.
#         X shape: [batch_size, 1, d] (BoTorch convention) or [batch_size, d]
#         Returns: [batch_size]
#         """
#         # BoTorch optimization usually passes a 'q' dimension (batch, q, d).
#         # We usually optimize q=1. Squeeze it out.
#         if X.ndim == 3:
#             X = X.squeeze(1) # [batch, d]
        
#         # 1. MC Dropout sampling
#         # We need to force the underlying MDN to train mode to activate Dropout
#         self.model.model.train() 
        
#         pi_samples, mu_samples, sigma_samples = [], [], []
        
#         with torch.no_grad():
#             for _ in range(self.num_dropout_samples):
#                 # self.model is MDNWrapper, it handles normalization
#                 pi, mu, sigma = self.model(X) 
#                 pi_samples.append(pi)       # [batch, K]
#                 mu_samples.append(mu)       # [batch, K]
#                 sigma_samples.append(sigma) # [batch, K]
        
#         self.model.model.eval() # Reset to eval mode
        
#         # Stack samples: Shape [S, batch, K]
#         # S = num_dropout_samples
#         pi_s = torch.stack(pi_samples)
#         mu_s = torch.stack(mu_samples)
#         sigma_s = torch.stack(sigma_samples)
        
#         # 2. Discretize and compute probabilities on the grid
#         # We need to evaluate the PDF P(y|x) at every grid point y for every sample s
#         # y_eval: [1, 1, 1, num_bins] to broadcast against [S, batch, K, 1]
#         y_eval = self.y_discrete.view(1, 1, 1, -1)
        
#         # Unsqueeze params to [S, batch, K, 1]
#         mu_s = mu_s.unsqueeze(-1)
#         sigma_s = sigma_s.unsqueeze(-1)
#         pi_s = pi_s.unsqueeze(-1)
        
#         # Compute log Gaussian PDF for each component: log N(y|mu, sigma)
#         # Result: [S, batch, K, num_bins]
#         term1 = -0.5 * math.log(2 * math.pi) - torch.log(sigma_s)
#         term2 = -0.5 * ((y_eval - mu_s) / sigma_s)**2
#         log_pdf_k = term1 + term2
#         # Compute mixture PDF for each dropout sample s: 
#         # p(y|x, theta_s) = sum_k pi_k * N(y | ...)
#         # log_prob_s: [S, batch, num_bins]
#         log_prob_s = torch.logsumexp(torch.log(pi_s + 1e-10) + log_pdf_k, dim=2)
#         # Convert to probability mass (approximate integration via Riemann sum)
#         prob_s = torch.exp(log_prob_s) * self.delta_y 
        
#         # 3. Compute entropy terms
#         # A. Marginal probability: P(y|x) = 1/S * sum_s P(y|x, theta_s)
#         # Average across dropout samples
#         prob_marginal = prob_s.mean(dim=0) # [batch, num_bins]
#         # Marginal entropy: H[P(y|x)]
#         entropy_marginal = -torch.sum(prob_marginal * torch.log(prob_marginal + 1e-10), dim=-1)
        
#         # B. Conditional entropy: E_theta[H[P(y|x, theta)]]
#         # Compute entropy for each sample s individually
#         # Entropy_s: [S, batch]
#         entropy_s = -torch.sum(prob_s * torch.log(prob_s + 1e-10), dim=-1)
#         # Expected conditional entropy
#         expected_conditional_entropy = entropy_s.mean(dim=0) # [batch]
        
#         # BALD = marginal entropy - expected conditional entropy
#         bald_score = entropy_marginal - expected_conditional_entropy
        
#         return bald_score
