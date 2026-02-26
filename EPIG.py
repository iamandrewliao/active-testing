# Copied directly from this implementation: https://github.com/hipeneurips/HIPE/tree/main; all credit to the authors
# EPIG was introduced in this paper: https://arxiv.org/pdf/2304.08151
from __future__ import annotations

from typing import Any

import math
import torch
import gpytorch
from torch.distributions import Normal
from botorch import settings
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.input_constructors import acqf_input_constructor
from botorch.acquisition.monte_carlo import MCAcquisitionObjective
from botorch.acquisition.objective import (
    PosteriorTransform,
    ScalarizedPosteriorTransform,
)
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.acquisition.utils import get_optimal_samples
from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP
from botorch.models.utils.gpytorch_modules import MIN_INFERRED_NOISE_LEVEL
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import (
    concatenate_pending_points,
    is_fully_bayesian,
    t_batch_mode_transform,
)
from torch import Tensor

class qExpectedPredictiveInformationGain(AcquisitionFunction):
    def __init__(
        self,
        model: Model,
        mc_points: Tensor,
        posterior_transform: PosteriorTransform | None = None,
        X_pending: Tensor | None = None,
        map_model: Model | None = None,
    ) -> None:
        """Batch implementation of Expected Predictive Information Gain (EPIG),
        which maximizes the mutual information between the subsequent queries and
        a test set of interest under the specified model. The set set may be
        randomly drawn, constitute of data from previous tasks, or a user-defined
        distribution to infuse prior knowledge.

        Args:
            model: A SingleTask or fully bayesian model.
            mc_points: A `batch_shape x N x d` tensor of points to use for
                MC-integrating the posterior entropy. Usually, these are qMC
                samples on the whole design space, but biased sampling directly
                allows weighted integration over a biased sample of test_points.
            posterior_transform: A PosteriorTransform.
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points.
        """
        super().__init__(model)

        self.register_buffer("mc_points", mc_points)
        self.register_buffer("X_pending", X_pending)
        self.posterior_transform = posterior_transform
        self.map_model = map_model

    def _model_specific_condition_on_mean(self, X: Tensor) -> Model:
        """
        This method is used to condition the model on the observations.
        It is rather awkwardly implemented since the interface of 
        coniditon_on_observations is not uniform across models. 
        """
        if self.map_model is not None:
            acq_model = self.map_model
        else:
            acq_model = self.model
        posterior = acq_model.posterior(
            X, observation_noise=False, posterior_transform=self.posterior_transform
        )
 
        cond_Y = posterior.mean
        noise = (
            acq_model.posterior(
                X,
                observation_noise=True,
                posterior_transform=self.posterior_transform,
            ).variance - posterior.variance
        )
        # reshapes X: batch_size * q * D to  batch_size * num_models * q * D if
        # model is fully bayesian to accommodate the condition_on_observations call
        # NOTE @hvarfner: this too can be changed once condition_on_observations has
        # a unified interface
        if is_fully_bayesian(acq_model):
            cond_X = X.unsqueeze(-3).expand(*[cond_Y.shape[:-1] + X.shape[-1:]])
        else:
            cond_X = X
            
        # ModelListGP.condition_on_observations does not condition each sub-model on
        # each output which is what is intended, so we have to go into each submodel
        # and condition in these instead
        #with settings.propagate_grads(True):  # Huge Memory Leak
        if isinstance(acq_model, ModelListGP):
            # If we use a ScalarizedPosteriorTransform with ModelListGPs, we still
            # need to make sure that there are m output dimensions to condition each
            # model on the outputs  - i.e. a ModelListGP-specific workaround
            # for condition_on_observations
            # NOTE @hvarfner: this can be changed once condition_on_observations has
            # a unified interface
            cond_Y = cond_Y.expand(cond_X.shape[:-1] + (acq_model.num_outputs,))
            noise = noise.expand(cond_X.shape[:-1] + (acq_model.num_outputs,))
            # NOTE @hvarfner this is a hacky workaround for the same issue as above
            conditional_model = ModelListGP(
                *[
                    submodel.condition_on_observations(
                        cond_X, cond_Y[..., i : i + 1], noise=noise[..., i : i + 1]
                    )
                    for i, submodel in enumerate(acq_model.models)
                ]
            )
        elif isinstance(acq_model, SaasFullyBayesianMultiTaskGP):
            conditional_model = acq_model.condition_on_observations(
                X=X,
                Y=cond_Y[0:1, :, :],
                noise=noise[0:1, :, :],
            )
        else:
            conditional_model = acq_model.condition_on_observations(
                X=cond_X,
                Y=cond_Y,
                noise=noise,
            )
        return conditional_model, acq_model

    @concatenate_pending_points
    @t_batch_mode_transform(assert_output_shape=False)
    def forward(self, X: Tensor, average: bool = True) -> Tensor:
        conditional_model, acq_model = self._model_specific_condition_on_mean(X)
        cond_posterior = BatchedMultiOutputGPyTorchModel.posterior(
            conditional_model, 
            X=self.mc_points.unsqueeze(-2).unsqueeze(1), 
            observation_noise=True
        )
        uncond_var = BatchedMultiOutputGPyTorchModel.posterior(
            acq_model,
            X=self.mc_points.unsqueeze(-2).unsqueeze(1), 
            observation_noise=True
        ).variance
        cond_var = cond_posterior.variance.clamp_min(MIN_INFERRED_NOISE_LEVEL)
        
        # the argmax is independent of prev_entropy, but enforces non-negativity
        # summing over the number of objectives and mean over the number of samples
        prev_entropy = torch.log(
            uncond_var  * 2 * math.pi * math.exp(1)
        ).squeeze(-1).sum(-1) / 2
        post_entropy = torch.log(
            cond_var * 2 * math.pi * math.exp(1)
        ).squeeze(-1).sum(-1) / 2
        if not average:
            return (prev_entropy-post_entropy)
        
        if acq_model.train_targets.numel() == 0:
            return -post_entropy.mean(0)
        
        return (prev_entropy-post_entropy).mean(0)