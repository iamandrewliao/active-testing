'''
MLP that takes in factor values and returns Mixture of Gaussians distribution parameters
(exactly like https://arxiv.org/pdf/2502.09829)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

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

# Model Wrapper
class MDNWrapper(nn.Module):
    """
    A simple wrapper that normalizes the model inputs.
    """
    def __init__(self, model, bounds):
        super().__init__()
        self.model = model
        # Registering bounds as a buffer ensures they move to GPU 
        # automatically when you call .to('cuda') on this object.
        self.register_buffer("bounds", bounds)

    def forward(self, x):
        # Normalize inputs automatically
        x_norm = (x - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
        return self.model(x_norm)


# MDN Helper Functions
import math

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


def fit_mdn_model(train_X, train_Y, bounds, num_epochs=200, lr=0.01, hidden_dim=64, components=2):
    """
    Trains the MDN surrogate model.
    Uses K=2 components as suggested in the paper.
    """
    device = train_X.device
    bounds = bounds.to(device)
    
    # Initialize model
    input_dim = train_X.shape[-1]
    
    # The paper uses K=2 components
    model = MDN(input_dim=input_dim, hidden_dim=hidden_dim, n_components=components)
    model.to(device)
    
    # Wrap it to handle normalization automatically
    wrapped_model = MDNWrapper(model, bounds)
    
    optimizer = torch.optim.Adam(wrapped_model.parameters(), lr=lr)
    
    # Simple full-batch training loop
    wrapped_model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # forward pass via wrapper (normalizes X)
        pi, mu, sigma = wrapped_model(train_X)
        loss = mdn_loss(pi, mu, sigma, train_Y)
        loss.backward()
        optimizer.step()
        
    return wrapped_model

from botorch.acquisition.acquisition import AcquisitionFunction

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
        y_grid = torch.linspace(outcome_range[0], outcome_range[1], num_bins)
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