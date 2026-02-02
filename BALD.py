'''BALD (Bayesian Active Learning by Disagreement) acquisition function for active testing.
Implements the method from: https://arxiv.org/pdf/2502.09829 (EfficientEval paper)

BALD: https://arxiv.org/pdf/1112.5745
'''

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from torch.distributions import MixtureSameFamily


class BALD(AcquisitionFunction):
    def __init__(self, model: Model, num_samples: int = 128, use_discretization: bool = True, n_bins: int = 25):
        """
        Args:
            model: The surrogate model (should support MC dropout if using MDN)
            num_samples: Number of samples for entropy estimation via sampling
            use_discretization: If True, use discretization method for entropy (as in paper)
            n_bins: Number of bins for discretization (paper uses n=25)
        """
        super().__init__(model)
        self.num_samples = num_samples
        self.use_discretization = use_discretization
        self.n_bins = n_bins

    def _compute_entropy_discretization(self, dist, n_bins=25):
        """
        Compute entropy by discretizing the empirical distribution into bins.
        This matches the method described in the paper (n=25 bins).
        
        Args:
            dist: Distribution (MixtureSameFamily or other)
            n_bins: Number of bins for discretization (paper uses 25)
        
        Returns:
            entropy: (batch_shape,) tensor of entropy values
        """
        # Sample from the distribution to get empirical range and histogram
        # Paper uses discretization, which we approximate via histogram of samples
        with torch.no_grad():
            # Sample many points to get good histogram estimate
            samples = dist.sample((10000,))  # (num_samples, batch_shape)
            
            # Get batch shape
            if samples.ndim > 1:
                batch_shape = samples.shape[1:]
                # Flatten batch dimensions for processing
                samples_flat = samples.flatten(1)  # (num_samples, batch_size)
                batch_size = samples_flat.shape[1]
            else:
                samples_flat = samples.unsqueeze(-1)  # (num_samples, 1)
                batch_size = 1
                batch_shape = ()
            
            # Compute histogram for each batch element
            min_val = samples_flat.min()
            max_val = samples_flat.max()
            # Add small padding
            padding = (max_val - min_val) * 0.1
            min_val = min_val - padding
            max_val = max_val + padding
            
            # Create bins (on same device as samples)
            bin_edges = torch.linspace(min_val.item(), max_val.item(), n_bins + 1, 
                                      device=samples.device, dtype=samples.dtype)
            
            # Compute histogram for each batch element
            # torch.histogram only works on CPU, so we need to move to CPU for histogram computation
            bin_probs = torch.zeros(batch_size, n_bins, 
                                   device=samples.device, dtype=samples.dtype)
            
            for i in range(batch_size):
                # Move to CPU for histogram computation (torch.histogram doesn't support CUDA)
                samples_cpu = samples_flat[:, i].cpu()
                bin_edges_cpu = bin_edges.cpu()
                
                # Count samples in each bin
                hist, _ = torch.histogram(samples_cpu, bins=bin_edges_cpu)
                
                # Normalize to get probabilities
                hist_sum = hist.sum().float()
                if hist_sum > 0:
                    bin_probs[i] = (hist.float() / hist_sum).to(samples.device)
                else:
                    # Fallback: uniform distribution if no samples
                    bin_probs[i] = torch.ones(n_bins, device=samples.device, dtype=samples.dtype) / n_bins
            
            # Reshape back to original batch shape
            if len(batch_shape) > 0:
                bin_probs = bin_probs.reshape(*batch_shape, n_bins)
            else:
                bin_probs = bin_probs.squeeze(0)  # Remove batch dim if it was added
        
        # Add small epsilon to avoid log(0)
        bin_probs = bin_probs + 1e-10
        bin_probs = bin_probs / bin_probs.sum(dim=-1, keepdim=True)  # Normalize
        
        # Compute entropy: H = -sum(p * log(p))
        entropy = -(bin_probs * torch.log(bin_probs)).sum(dim=-1)
        
        return entropy

    def _compute_entropy_sampling(self, dist, num_samples=128):
        """
        Compute entropy using sampling method (alternative to discretization).
        
        Args:
            dist: Distribution
            num_samples: Number of samples for estimation
        
        Returns:
            entropy: (batch_shape,) tensor of entropy values
        """
        # Sample from distribution
        samples = dist.sample(sample_shape=(num_samples,))
        # Compute log probabilities
        log_probs = dist.log_prob(samples)  # (num_samples, batch_shape)
        # Entropy estimate: H ≈ -E[log p(x)]
        entropy = -log_probs.mean(dim=0)
        return entropy

    def forward(self, X):
        """
        Compute BALD scores following the paper's method:
        I(π_i, T_j) = H[Q_ij] - E_θ[H[Q_ij|θ_ij]]
        
        where:
        - H[Q_ij] is the marginal entropy (entropy of the mixture)
        - E_θ[H[Q_ij|θ_ij]] is the expected conditional entropy over parameter samples
        """
        posterior = self.model.posterior(X)
        dist = posterior.distribution
        
        # 1. Marginal entropy: H[Q_ij]
        # This is the entropy of the mixture distribution
        if self.use_discretization:
            marginal_entropy = self._compute_entropy_discretization(dist, n_bins=self.n_bins)
        else:
            marginal_entropy = self._compute_entropy_sampling(dist, num_samples=self.num_samples)
        
        # Ensure correct shape: (batch_shape,)
        if marginal_entropy.ndim > 1:
            marginal_entropy = marginal_entropy.squeeze(-1)
        
        # 2. Expected conditional entropy: E_θ[H[Q_ij|θ_ij]]
        # This is the average entropy of individual parameter samples (MC dropout samples)
        # For MC dropout: E_θ[H[Q_ij|θ_ij]] = (1/num_samples) * sum(H[Q_ij|θ_ij])
        # where each Q_ij|θ_ij is one MC dropout sample
        
        # Check if model uses MC dropout and has method to get individual samples
        # This path is for MDN with MC dropout
        if hasattr(self.model, 'use_mc_dropout') and self.model.use_mc_dropout and \
           hasattr(self.model, '_sample_mc_dropout_distributions'):
            try:
                # Get individual MC dropout samples
                individual_dists, _ = self.model._sample_mc_dropout_distributions(X)
                
                # Compute entropy for each individual sample
                individual_entropies = []
                for ind_dist in individual_dists:
                    if self.use_discretization:
                        ent = self._compute_entropy_discretization(ind_dist, n_bins=self.n_bins)
                    else:
                        ent = self._compute_entropy_sampling(ind_dist, num_samples=self.num_samples)
                    individual_entropies.append(ent)
                
                # Average over MC samples: E_θ[H[Q_ij|θ_ij]] = (1/num_samples) * sum(H[Q_ij|θ_ij])
                exp_conditional_entropy = torch.stack(individual_entropies, dim=0).mean(dim=0)
            except Exception:
                # Fall back to standard mixture computation if MC dropout fails
                if isinstance(dist, MixtureSameFamily):
                    mix_dist = dist.mixture_distribution
                    comp_dist = dist.component_distribution
                    comp_entropy = comp_dist.entropy()  # (..., n_components)
                    mix_weights = mix_dist.probs  # (..., n_components)
                    exp_conditional_entropy = (mix_weights * comp_entropy).sum(dim=-1)  # (batch_shape,)
                else:
                    # Fall back to standard entropy
                    if self.use_discretization:
                        exp_conditional_entropy = self._compute_entropy_discretization(dist, n_bins=self.n_bins)
                    else:
                        exp_conditional_entropy = self._compute_entropy_sampling(dist, num_samples=self.num_samples)
                    if exp_conditional_entropy.ndim > 1:
                        exp_conditional_entropy = exp_conditional_entropy.squeeze(-1)
        elif isinstance(dist, MixtureSameFamily):
            # Standard mixture: compute expected conditional entropy from components
            # This path is for DeepEnsemble and MDN without MC dropout
            mix_dist = dist.mixture_distribution
            comp_dist = dist.component_distribution
            
            # Compute entropy for each component
            comp_entropy = comp_dist.entropy()  # (..., n_components)
            
            # Get mixture weights
            mix_weights = mix_dist.probs  # (..., n_components)
            
            # Expected conditional entropy: sum(π_k * H[N_k])
            # where π_k are mixing weights and H[N_k] are component entropies
            exp_conditional_entropy = (mix_weights * comp_entropy).sum(dim=-1)  # (batch_shape,)
            
            # Ensure correct shape
            if exp_conditional_entropy.ndim > 1:
                exp_conditional_entropy = exp_conditional_entropy.squeeze(-1)
        else:
            # For non-mixture distributions, conditional entropy is just the entropy
            if self.use_discretization:
                exp_conditional_entropy = self._compute_entropy_discretization(dist, n_bins=self.n_bins)
            else:
                exp_conditional_entropy = self._compute_entropy_sampling(dist, num_samples=self.num_samples)
            
            if exp_conditional_entropy.ndim > 1:
                exp_conditional_entropy = exp_conditional_entropy.squeeze(-1)
        
        # 3. BALD = marginal entropy - expected conditional entropy
        bald_scores = marginal_entropy - exp_conditional_entropy
        
        # Ensure output shape is (batch_shape, 1) for BoTorch compatibility
        if bald_scores.ndim == 1:
            bald_scores = bald_scores.unsqueeze(-1)
        
        return bald_scores