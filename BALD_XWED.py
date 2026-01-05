'''BALD and XWED acquisition functions for active testing.
BALD: https://arxiv.org/pdf/1112.5745
XWED: https://arxiv.org/pdf/2202.06881
'''

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model


class BALD(AcquisitionFunction):
    def __init__(self, model: Model, num_samples: int = 128):
        super().__init__(model)
        self.num_samples = num_samples

    def forward(self, X):
        posterior = self.model.posterior(X)
        dist = posterior.distribution
        # 1. Marginal entropy
        # sample y from mixture posterior
        samples = dist.sample(sample_shape=(self.num_samples,))
        # for each sample, compute log probability
        log_probs = dist.log_prob(samples)  # (num_samples, batch_shape, 1)
        # take average log prob over samples
        marginal_entropy = -log_probs.mean(dim=0)  # (batch_shape, 1)

        # 2. Expected conditional entropy
        mix_dist = dist.mixture_distribution
        comp_dist = dist.component_distribution
        # for each component, compute entropy
        comp_entropy = comp_dist.entropy()  # (..., M, 1) -> squeeze to (..., M)
        # combine with mixture weights to get conditional entropy
        mix_weights = mix_dist.probs  # (..., M)
        exp_conditional_entropy = -(mix_weights*comp_entropy).sum(dim=-1)  # (batch_shape, 1)

        # 3. BALD = marginal entropy - expected conditional entropy
        bald_scores = marginal_entropy - exp_conditional_entropy

        return bald_scores


class XWED(AcquisitionFunction):
    def __init__(self, model: Model, num_samples: int = 128, cost_fn=None, y_max=None):
        super().__init__(model)
        self.num_samples = num_samples
        self.cost_fn = cost_fn  # function that maps predicted mean to weights for each entropy term
        if cost_fn is None:
            self.cost_fn = lambda y: (y_max - y).clamp(min=0.0)
        else:
            self.cost_fn = cost_fn

    def forward(self, X):
        posterior = self.model.posterior(X)
        dist = posterior.distribution
        # 1. Marginal entropy
        # sample y from mixture posterior
        samples = dist.sample(sample_shape=(self.num_samples,))
        # for each sample, compute log probability
        log_probs = dist.log_prob(samples)
        # modification of XWED: weight marginal probability by a function of the surrogate model's prediction
        weights_marg = self.cost_fn(samples)
        weighted_marginal_entropy = (weights_marg * -log_probs).mean(dim=0)

        # 2. Expected conditional entropy
        mix_dist = dist.mixture_distribution
        comp_dist = dist.component_distribution
        # for each component distribution, sample and compute log prob for the samples
        y_comp_samples = comp_dist.sample((self.num_samples,))
        log_probs_comp = comp_dist.log_prob(y_comp_samples)
        # modification of XWED: weight conditional entropy by a function of the surrogate model's prediction
        weights_comp = self.cost_fn(y_comp_samples)
        weighted_exp_conditional_entropy = (weights_comp * -log_probs_comp).mean(dim=0)

        # 3. XWED = weighted marginal entropy - weighted expected conditional entropy
        xwed_scores = weighted_marginal_entropy - weighted_exp_conditional_entropy

        return xwed_scores