'''
Active testing (surrogate model + acquisition function) & IID testing (uniform random sampling) for robot policy evaluation.

BoTorch sources:
https://botorch.org/docs/overview
https://botorch.readthedocs.io/en/latest/index.html
'''

import argparse
import pandas as pd
import torch
import time
import math

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
        
        end_time = time.time()
        print(f"Active sample selection took {end_time - start_time:.2f} seconds.")
        
        return unnormalized_candidate.squeeze(0) # Return a 1D tensor

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


def is_valid_point(point):
    """
    Prompts the user to manually confirm if a sampled point is valid.
    """
    x, y = point[0].item(), point[1].item()
    print("-" * 30)
    print(f"📍 Sampled point for validation: (x={x:.3f}, y={y:.3f})")
    
    while True:
        is_valid_str = input("Is this a valid point to evaluate? (y/n): ").lower().strip()
        if is_valid_str in ['y', 'yes']:
            return True
        elif is_valid_str in ['n', 'no']:
            print("Point marked as invalid by user. Resampling...")
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


def run_evaluation(point):
    """
    Simulates a robot evaluation trial for a given point.
    Prompts the user for the outcome.
    """
    x, y = point[0].item(), point[1].item()
    print("-" * 30)
    print(f"🤖 Running trial at position: (x={x:.3f}, y={y:.3f})")
    
    # Get binary success/failure outcome
    while True:
        try:
            binary_outcome = int(input("Enter binary outcome (1 for success, 0 for failure): "))
            if binary_outcome in [0, 1]:
                break
            else:
                print("Invalid input. Please enter 0 or 1.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Get continuous outcome
    while True:
        try:
            print("0=failed completely, 1=moved to block, 2=grasped block, 3=moved to bowl, 4=dropped into bowl")
            continuous_outcome = int(input("Enter continuous outcome (0-4): "))
            if continuous_outcome in [0, 1, 2, 3, 4]:
                break
            else:
                print("Invalid input. Please enter a number between 0 and 4.")
        except ValueError:
            print("Invalid input. Please enter a number.")
            
    return binary_outcome, continuous_outcome


def main(args):
    """Main execution loop."""
    print(f"Starting evaluation with mode: '{args.mode}' for {args.num_evals} trials.")
    
    # Define the search space bounds for our factors [x, y]
    # Format: torch.tensor([[low_x, low_y], [high_x, high_y]])
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], **tkwargs)
    
    results_data = []

    # Initialize Sampler for the Main Loop
    if args.mode == 'active':
        # Initial Random Sampling: active testing needs a few initial points to build the first model.
        print(f"\n--- Collecting {args.num_init_pts} initial random points ---")
        initial_X = []
        initial_Y = []
        for i in range(args.num_init_pts):
            print(f"\nTrial {i+1}/{args.num_evals} (Initial Random Sample)")
            point = IIDSampler(bounds).get_next_point()
            while not is_valid_point(point):
                point = IIDSampler(bounds).get_next_point()

            binary_outcome, continuous_outcome = run_evaluation(point)
            
            initial_X.append(point)
            # Use the binary outcome for the surrogate model
            initial_Y.append(torch.tensor([binary_outcome], **tkwargs))

            results_data.append({
                'trial': i + 1,
                'mode': 'initial_random',
                'x': point[0].item(),
                'y': point[1].item(),
                'orientation': None,  # Placeholder for future use
                'binary_outcome': binary_outcome,
                'continuous_outcome': continuous_outcome
            })
            
        # Convert initial data to tensors
        train_X = torch.stack(initial_X)
        train_Y = torch.stack(initial_Y)
        sampler = ActiveTester(train_X, train_Y, bounds)
    else: # args.mode == 'iid'
        sampler = IIDSampler(bounds)

    # --- Main Evaluation Loop ---
    print(f"\n--- Starting main evaluation loop with '{args.mode}' sampling ---")
    for i in range(args.num_init_pts, args.num_evals):
        print(f"\nTrial {i+1}/{args.num_evals} ({args.mode} sample)")
        
        point = sampler.get_next_point()
        while not is_valid_point(point):
            point = sampler.get_next_point()
            
        binary_outcome, continuous_outcome = run_evaluation(point)
        
        # Convert outcomes to tensors for updating the model
        new_y_tensor = torch.tensor([binary_outcome], **tkwargs)
        
        # Update the sampler with the new data
        sampler.update(point, new_y_tensor)
        
        # Record the results
        results_data.append({
            'trial': i + 1,
            'mode': args.mode,
            'x': point[0].item(),
            'y': point[1].item(),
            'orientation': None,
            'binary_outcome': binary_outcome,
            'continuous_outcome': continuous_outcome
        })

    # --- Save Results ---
    df = pd.DataFrame(results_data)
    df.to_csv(args.output_file, index=False)
    print(f"\n✅ Evaluation complete. Results saved to '{args.output_file}'.")
    print(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run robot policy evaluations using IID or Active Testing."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["iid", "active"],
        required=True,
        help="The sampling strategy to use."
    )
    parser.add_argument(
        "--num_evals",
        type=int,
        default=20,
        help="Total number of evaluations to run."
    )
    parser.add_argument(
        "--num_init_pts",
        type=int,
        default=5,
        help="Number of initial random points for bootstrapping the active learning model."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_results.csv",
        help="Path to save the CSV results file."
    )
    
    args = parser.parse_args()
    if args.num_init_pts >= args.num_evals:
        raise ValueError("`num_init_pts` must be less than `num_evals`.")
        
    main(args)