'''
MLP that takes in factor values and returns Mixture of Gaussians distribution parameters
(exactly like https://arxiv.org/pdf/2502.09829)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from botorch.posteriors import Posterior
from botorch.models.model import Model
from botorch.acquisition.acquisition import AcquisitionFunction
import math

OUTCOME_RANGE = (0, 4)

# Mixture of Gaussians MLP (Mixture Density Network)
class MDN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_components, dropout_prob=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.n_components = n_components
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
        
        self.pi_head = nn.Linear(hidden_dim, n_components)
        self.sigma_head = nn.Linear(hidden_dim, n_components)
        self.mu_head = nn.Linear(hidden_dim, n_components)

    def forward(self, x):
        features = self.feature_extractor(x)
        pi_logits = self.pi_head(features)
        mu = self.mu_head(features)
        sigma_raw = self.sigma_head(features)
        
        pi = F.softmax(pi_logits, dim=-1) 
        sigma = F.softplus(sigma_raw) + 1e-6
        return pi, mu, sigma

class MDNPosterior(Posterior):
    def __init__(self, mean, variance):
        self._mean = mean
        self._variance = variance

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

class MDNWrapper(Model): # Inherit from BoTorch Model
    def __init__(self, model, bounds):
        super().__init__()
        self.model = model
        self.register_buffer("bounds", bounds)
        self._num_outputs = 1 # Required by BoTorch

    def forward(self, x):
        # Normalize inputs automatically
        x_norm = (x - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
        return self.model(x_norm)

    @property
    def num_outputs(self):
        return self._num_outputs

    def posterior(self, X, observation_noise=False, **kwargs):
        # 1. Forward pass (returns mixture params)
        pi, mu, sigma = self.forward(X)
        
        # 2. Moment Matching: Convert Mixture -> Single Gaussian
        # Mixture Mean = sum(pi * mu)
        mixture_mean = torch.sum(pi * mu, dim=-1, keepdim=True)
        
        # Mixture Variance = sum(pi * (sigma^2 + mu^2)) - mixture_mean^2
        # (Law of total variance)
        second_moment = torch.sum(pi * (sigma**2 + mu**2), dim=-1, keepdim=True)
        mixture_var = second_moment - mixture_mean**2
        
        # 3. Return BoTorch Posterior
        return MDNPosterior(mixture_mean, mixture_var)

# MDN Helper Functions
def mdn_loss(pi, mu, sigma, target):
    """
    Computes Negative Log Likelihood (NLL) for Gaussian Mixture.
    target shape: [batch, 1]
    pi, mu, sigma shape: [batch, K]
    """
    # Ensure target matches component count for broadcasting: [batch, K]
    target = target.expand_as(mu)
    
    # Log N(y | mu, sigma) = -0.5*log(2pi) - log(sigma) - 0.5*((y-mu)/sigma)^2
    # We use log_sigma directly if possible, but here we have sigma.
    log_component_prob = -0.5 * math.log(2 * math.pi) \
                         - torch.log(sigma) \
                         - 0.5 * ((target - mu) / sigma)**2
    
    # Log of mixture: log( sum(pi * prob) ) = logsumexp( log(pi) + log_prob )
    # This is numerically stable.
    log_mix_prob = torch.logsumexp(torch.log(pi + 1e-10) + log_component_prob, dim=1)
    
    return -torch.mean(log_mix_prob)

def train_mdn(model, train_X, train_Y, bounds=None, num_epochs=200, lr=0.01):
    """
    Trains the MDN model. 
    Similar to train_ensemble, this handles normalization internally 
    so the raw model learns on scaled data [0, 1].
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Normalize inputs for training if bounds provided
    if bounds is not None:
        # Ensure bounds match X's device and dtype
        bounds = bounds.to(train_X)
        train_X = (train_X - bounds[0]) / (bounds[1] - bounds[0])

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # Forward pass on raw model with normalized data
        pi, mu, sigma = model(train_X)
        loss = mdn_loss(pi, mu, sigma, train_Y)
        loss.backward()
        optimizer.step()
    model.eval()

class MDN_BALD(AcquisitionFunction):
    def __init__(self, model, num_dropout_samples=10, num_bins=25, outcome_range=OUTCOME_RANGE):
        """
        Implements BALD for MDN using MC Dropout and Discretization.
        
        Args:
            model: The MDNWrapper instance.
            num_dropout_samples: T=10 samples as per paper.
            num_bins: n=25 bins for entropy estimation as per paper.
            outcome_range: Min/Max of Y to create the grid.
        """
        super().__init__(model)
        self.num_dropout_samples = num_dropout_samples
        
        # Discretization grid for entropy calculation 
        # We create a fixed grid over the Y-space (outcomes)
        self.outcome_range = outcome_range
        self.num_bins = num_bins
        
        # Grid: Shape [Bins]
        tkwargs = {"device": model.bounds.device, "dtype": model.bounds.dtype}
        y_grid = torch.linspace(outcome_range[0], outcome_range[1], num_bins, **tkwargs)
        self.register_buffer("y_grid", y_grid)
        self.delta_y = self.y_grid[1] - self.y_grid[0]

    def forward(self, X):
        """
        Compute BALD score.
        X shape: [batch_size, 1, d] (BoTorch convention) or [batch_size, d]
        Returns: [batch_size]
        """
        # BoTorch optimization usually passes a 'q' dimension (batch, q, d).
        # We usually optimize q=1. Squeeze it out.
        if X.ndim == 3:
            X = X.squeeze(1) # [batch, d]
        
        # 1. MC Dropout Sampling
        # We need to force the underlying MDN to train mode to activate Dropout
        self.model.model.train() 
        
        pi_samples, mu_samples, sigma_samples = [], [], []
        
        with torch.no_grad():
            for _ in range(self.num_dropout_samples):
                # self.model is MDNWrapper, it handles normalization
                pi, mu, sigma = self.model(X) 
                pi_samples.append(pi)       # [batch, K]
                mu_samples.append(mu)       # [batch, K]
                sigma_samples.append(sigma) # [batch, K]
        
        self.model.model.eval() # Reset to eval mode
        
        # Stack samples: Shape [S, batch, K]
        # S = num_dropout_samples
        pi_s = torch.stack(pi_samples)
        mu_s = torch.stack(mu_samples)
        sigma_s = torch.stack(sigma_samples)
        
        # 2. Discretize and compute probabilities on the grid
        # We need to evaluate the PDF P(y|x) at every grid point y for every sample s
        # y_eval: [1, 1, 1, Bins] to broadcast against [S, batch, K, 1]
        y_eval = self.y_grid.view(1, 1, 1, -1)
        
        # Unsqueeze params to [S, batch, K, 1]
        mu_s = mu_s.unsqueeze(-1)
        sigma_s = sigma_s.unsqueeze(-1)
        pi_s = pi_s.unsqueeze(-1)
        
        # Compute Log Gaussian PDF for each component: log N(y|mu, sigma)
        # Result: [S, batch, K, Bins]
        term1 = -0.5 * math.log(2 * math.pi) - torch.log(sigma_s)
        term2 = -0.5 * ((y_eval - mu_s) / sigma_s)**2
        log_pdf_k = term1 + term2
        # Compute Mixture PDF for each dropout sample s: 
        # p(y|x, theta_s) = sum_k pi_k * N(y | ...)
        # log_prob_s: [S, batch, Bins]
        log_prob_s = torch.logsumexp(torch.log(pi_s + 1e-10) + log_pdf_k, dim=2)
        # Convert to probability mass (approximate integration via Riemann sum)
        prob_s = torch.exp(log_prob_s) * self.delta_y 
        
        # 3. Compute Entropy Terms
        # A. Marginal Probability: P(y|x) = 1/S * sum_s P(y|x, theta_s)
        # Average across dropout samples
        prob_marginal = prob_s.mean(dim=0) # [batch, Bins]
        # Marginal Entropy: H[P(y|x)]
        entropy_marginal = -torch.sum(prob_marginal * torch.log(prob_marginal + 1e-10), dim=-1)
        
        # B. Conditional Entropy: E_theta[H[P(y|x, theta)]]
        # Compute entropy for each sample s individually
        # Entropy_s: [S, batch]
        entropy_s = -torch.sum(prob_s * torch.log(prob_s + 1e-10), dim=-1)
        # Expected Conditional Entropy
        expected_conditional_entropy = entropy_s.mean(dim=0) # [batch]
        
        # BALD = Marginal Entropy - Expected Conditional Entropy
        bald_score = entropy_marginal - expected_conditional_entropy
        
        return bald_score