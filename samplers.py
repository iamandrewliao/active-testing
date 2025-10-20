'''
Different testing strategies
-Active testing (surrogate model + acquisition function)
-IID testing (uniform random sampling)

BoTorch sources:
https://botorch.org/docs/overview
https://botorch.readthedocs.io/en/latest/index.html
'''
import pandas as pd
import torch
import time

from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.joint_entropy_search import qJointEntropySearch
from botorch.acquisition.utils import get_optimal_samples
from botorch.optim import optimize_acqf

# Set up torch device and data type
tkwargs = {"dtype": torch.double, "device": "cpu"}

class ActiveTester:
    def __init__(self, initial_X, initial_Y, bounds):
        self.train_X = initial_X
        self.train_Y = initial_Y
        self.bounds = bounds
        self.model = None

    def _fit_model(self):
        """Fits a surrogate model to the current training data."""
        # Note: input_transform normalizes X to [0, 1]^d
        # outcome_transform standardizes Y to have zero mean and unit variance
        self.model = SingleTaskGP(
            train_X=self.train_X,
            train_Y=self.train_Y,
            input_transform=Normalize(d=self.train_X.shape[-1], bounds=self.bounds),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

    def get_next_point(self):
        """
        Fits the model and optimizes the acquisition function to find the
        next best point to sample.
        """
        print("Fitting surrogate model and optimizing acquisition function...")
        start_time = time.time()
        self._fit_model()

        # 1. Get some Monte Carlo samples of the optimal inputs and outputs to compute the acquisition function (needed by JES)
        # We use normalized bounds [0, 1] because the model normalizes inputs
        normalized_bounds = torch.tensor([[0.0] * self.bounds.shape[1], [1.0] * self.bounds.shape[1]], **tkwargs)
        optimal_inputs, optimal_outputs = get_optimal_samples(
            self.model, bounds=normalized_bounds, num_optima=16
        )

        # 2. Construct the acquisition function
        jes = qJointEntropySearch(
            model=self.model,
            optimal_inputs=optimal_inputs,
            optimal_outputs=optimal_outputs,
            estimation_type="LB",
        )

        # 3. Optimize the acquisition function
        candidate, _ = optimize_acqf(
            acq_function=jes,
            bounds=normalized_bounds,
            q=1,
            num_restarts=4,
            raw_samples=256,
        )
        
        # Candidate is in the normalized space [0, 1]^d, so we un-normalize it
        # using the original bounds before returning.
        unnormalized_candidate = self.bounds[0] + candidate * (self.bounds[1] - self.bounds[0])
        point = unnormalized_candidate.squeeze(0) # Return a 1D tensor
        
        end_time = time.time()
        print(f"Active sample selection took {end_time - start_time:.2f} seconds.")

        return point

    def update(self, new_x, new_y):
        """Adds a new data point to the training set."""
        self.train_X = torch.cat([self.train_X, new_x.unsqueeze(0)])
        self.train_Y = torch.cat([self.train_Y, new_y.unsqueeze(0)])


class IIDSampler:
    def __init__(self, bounds):
        self.bounds = bounds
        self.dim = bounds.shape[1]

    def get_next_point(self):
        """Generates a new point by sampling uniformly from the bounds."""
        # Formula: low + (high - low) * rand
        point = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * torch.rand(self.dim, **tkwargs)
        return point
    
    def update(self, new_x, new_y):
        """Does nothing."""
        pass


class PointLoader:
    """
    A 'sampler' that loads points from a CSV file instead of generating them
    e.g. if you want to eval multiple policies on the same points
    """
    def __init__(self, filepath):
        print(f"Loading evaluation points from {filepath}...")
        try:
            df = pd.read_csv(filepath)
            if not {'x', 'y'}.issubset(df.columns):
                raise ValueError("CSV file must contain 'x' and 'y' columns.")
            
            self.points = [
                torch.tensor([row['x'], row['y']], **tkwargs)
                for _, row in df.iterrows()
            ]
        except FileNotFoundError:
            print(f"Error: The file '{filepath}' was not found.")
            exit()
        except Exception as e:
            print(f"Error reading the file: {e}")
            exit()
            
        self.index = 0
        print(f"Successfully loaded {len(self.points)} points.")

    def get_next_point(self):
        """Returns the next point from the pre-loaded list."""
        if self.index >= len(self.points):
            print("Error: Ran out of points to load from the file.")
            print("Adjust --num_evals or provide a file with more points.")
            exit()
        
        point = self.points[self.index]
        self.index += 1
        return point

    def update(self, new_x, new_y):
        """Does nothing."""
        pass
