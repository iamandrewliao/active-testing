'''
Using active testing to select the next best combination of generalization factors (e.g. lighting, # of distractors) to evaluate.
Typical flow: 
Evaluation rollouts (w/ data on factor values) -> Fit surrogate model -> Optimize acquisition function -> Select next factor values to evaluate with
'''
import torch

# Example rollout data
factor_values = {
    "lighting": [0.5, 0.7, 0.4, 0.8],  # brightness
    "obj_location": [0.2, 0.3, 0.5, 0.1],  # distance from nearest training location
    }
outcomes = [1, 0, 0, 1]  # success = 1
# Concatenate factor values and outcomes into training data

train_X = torch.tensor([factor_values["lighting"], factor_values["obj_location"]]).T
train_Y = torch.tensor(outcomes).unsqueeze(-1).to(torch.float32)

# BoTorch sources:
# https://botorch.org/docs/overview
# https://botorch.readthedocs.io/en/latest/index.html


# Fit a Gaussian Process model to data (https://botorch.org/docs/models)
# SingleTaskGP: a single-task exact GP that supports both inferred and observed noise. When noise observations are not provided, it infers a homoskedastic noise level.
# MixedSingleTaskGP: a single-task exact GP that supports mixed search spaces, which combine discrete and continuous features.
# SaasFullyBayesianSingleTaskGP: a fully Bayesian single-task GP with the SAAS prior. This model is suitable for sample-efficient high-dimensional Bayesian optimization.
# SingleTaskVariationalGP: an approximate model for faster computation when you have a lot of data or your responses are non-Gaussian.
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize
from botorch.fit import fit_gpytorch_mll  # fit surrogate model (suitable for exact GPs)
from gpytorch.mlls import ExactMarginalLogLikelihood

# train_X = torch.rand(10, 2, dtype=torch.double) * 2
# # explicit output dimension -- Y is 10 x 1
# train_Y = 1 - (train_X - 0.5).norm(dim=-1, keepdim=True)
# train_Y += 0.1 * torch.rand_like(train_Y)

gp = SingleTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    input_transform=Normalize(d=2),
)
# gp = SingleTaskGP(train_X, train_Y)

mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)

# https://botorch.org/docs/posteriors

# Construct an acquisition function (https://botorch.org/docs/acquisition)
# Joint Entropy Search (JES)
# Bayesian Active Learning By Disagreement (BALD)
# GIBBON (General-purpose Information-Based Bayesian OptimisatioN)

from botorch.acquisition.utils import get_optimal_samples

tkwargs = {"dtype": torch.double, "device": "cpu"}
num_samples = 12

# Bounds of the search space. If the model inputs are normalized, the bounds should be normalized as well.
bounds = torch.tensor([[0.0], [1.0]], **tkwargs)

optimal_inputs, optimal_outputs = get_optimal_samples(
    gp, bounds=bounds, num_optima=num_samples
)

from botorch.acquisition.joint_entropy_search import qJointEntropySearch

jes_lb = qJointEntropySearch(
    model=gp,
    optimal_inputs=optimal_inputs,
    optimal_outputs=optimal_outputs,
    estimation_type="LB",
)

# Optimize the acquisition function
from botorch.optim import optimize_acqf

candidate, acq_value = optimize_acqf(
    acq_function=jes_lb,
    bounds=bounds,
    q=1,
    num_restarts=4,
    raw_samples=256,
)
print("JES-LB: candidate={}, acq_value={}".format(candidate, acq_value))

# def main():
#     print("Hello from active-testing!")


# if __name__ == "__main__":
#     main()
