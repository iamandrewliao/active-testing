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
import os  # Added for checking file existence

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


def is_valid_point(point):
    """Automatically filters out any point past Lightning's reachability (approximated by a straight line)"""
    # Equation of the boundary line: y = 1.5x - 0.75
    # A point is 'invalid' if it's on the right side of the line
    x, y = point[0].item(), point[1].item()
    # Calculate if the point is in the invalid region
    is_invalid = (y <= 1.5*x - 0.75) and (x >= 0.5)
    
    return not is_invalid  # Return True if the point is valid


def run_evaluation(point):
    """
    Simulates a robot evaluation trial for a given point.
    Prompts the user for the outcome.
    """
    x, y = point[0].item(), point[1].item()
    print("-" * 30)
    print(f"🤖 Running trial at position: (x={x:.3f}, y={y:.3f})")
    
    # --- Get continuous outcome ---
    while True:
        try:
            print("Enter continuous outcome (0.5 increments):")
            print("  0=failed completely, 1=moved to block, 2=grasped block, 3=moved to bowl, 4=dropped (success)")
            continuous_outcome = float(input("Enter outcome (0-4): "))
            
            # check if in [0, 0.5, ..., 4]
            if continuous_outcome in [i*0.5 for i in range(9)]:
                break
            else:
                print("Invalid input. Please enter a number between 0 and 4 in 0.5 increments.")
        except ValueError:
            print("Invalid input. Please enter a number.")
            
    # --- Derive binary outcome ---
    binary_outcome = 1.0 if continuous_outcome == 4.0 else 0.0

    # --- Get number of steps taken ---
    if binary_outcome == 0:
        print("Failure reported. Setting steps to max.")
        steps_taken = args.max_steps
    else:
        while True:
            try:
                steps_taken = int(input("Enter number of steps taken to complete the task: "))
                if steps_taken >= 0:
                    break
                else:
                    print("Invalid input. Please enter a non-negative integer.")
            except ValueError:
                print("Invalid input. Please enter a number.")
            
    return binary_outcome, continuous_outcome, steps_taken


def main(args):
    """Main execution loop."""
    print(f"Starting evaluation with mode: '{args.mode}' for {args.num_evals} trials.")
    
    # Define the search space bounds for our factors [x, y]
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], **tkwargs)
    
    results_data = []
    loop_start_index = 0

    # --- Load existing data if available ---
    if os.path.exists(args.output_file):
        print(f"Found existing results file: '{args.output_file}'. Resuming session.")
        try:
            existing_df = pd.read_csv(args.output_file)
            if not existing_df.empty:
                results_data = existing_df.to_dict('records')
                loop_start_index = len(results_data)
                print(f"Loaded {loop_start_index} previous trials. Resuming from trial {loop_start_index + 1}.")
        except pd.errors.EmptyDataError:
            print(f"Warning: Output file '{args.output_file}' is empty. Starting a new session.")
        except Exception as e:
            print(f"Error reading existing results file: {e}. Starting a new session.")
            results_data = []
            loop_start_index = 0
    
    if loop_start_index >= args.num_evals and args.mode != 'loaded':
        print(f"Evaluation already complete with {loop_start_index} trials. Exiting.")
        return

    # --- Initialize Sampler ---
    sampler = None
    if args.mode == 'loaded':
        sampler = PointLoader(args.load_path)
        args.num_evals = len(sampler.points)  # Override num_evals
        if loop_start_index > 0:
            print(f"Skipping the first {loop_start_index} points from the loaded file.")
            sampler.index = loop_start_index
        
        if loop_start_index >= args.num_evals:
            print("All points from the loaded file have already been evaluated. Exiting.")
            return
        
        print(f"Running evaluation for {args.num_evals - loop_start_index} remaining loaded points.")

    elif args.mode == 'iid':
        sampler = IIDSampler(bounds)
        
    elif args.mode == 'active':
        if loop_start_index >= args.num_init_pts:
            print(f"Resuming in 'active' mode with {loop_start_index} points.")
            initial_X_tensors = [torch.tensor([row['x'], row['y']], **tkwargs) for row in results_data]
            initial_Y_tensors = [torch.tensor([row['continuous_outcome']], **tkwargs) for row in results_data]
            train_X = torch.stack(initial_X_tensors)
            train_Y = torch.stack(initial_Y_tensors)
            sampler = ActiveTester(train_X, train_Y, bounds)
        else:
            print("Not enough data for active learning yet. Starting with initial random sampling.")
            sampler = IIDSampler(bounds)

    # --- Main Evaluation Loop ---
    print(f"\n--- Starting main evaluation loop ---")
    for i in range(loop_start_index, args.num_evals):
        
        current_mode = args.mode
        if args.mode == 'active':
            current_mode = 'initial_random' if i < args.num_init_pts else 'active'

        # --- Handle sampler transition for 'active' mode ---
        if args.mode == 'active' and i == args.num_init_pts:
            print("\n" + "="*50)
            print(f"Reached {args.num_init_pts} initial points. Switching to Active Testing.")
            print("="*50 + "\n")
            
            initial_X_tensors = [torch.tensor([row['x'], row['y']], **tkwargs) for row in results_data]
            initial_Y_tensors = [torch.tensor([row['continuous_outcome']], **tkwargs) for row in results_data]
            train_X = torch.stack(initial_X_tensors)
            train_Y = torch.stack(initial_Y_tensors)
            sampler = ActiveTester(train_X, train_Y, bounds)

        print(f"\nTrial {i+1}/{args.num_evals} (mode: {current_mode})")
        
        point = sampler.get_next_point()
        
        # --- Robust validity check ---
        while not is_valid_point(point):
            print(f"Point (x={point[0]:.3f}, y={point[1]:.3f}) is not valid -> handling...")
            if args.mode == 'loaded':
                print(f"Error: The loaded point from trial {i+1} is invalid. Please check your source file.")
                print("Stopping evaluation.")
                exit()
            elif current_mode == 'active':
                # Fall back to a single random sample for this trial to avoid an infinite loop
                print("Warning: Active learner suggested an invalid point. Falling back to IID sampling for this one trial.")
                point = IIDSampler(bounds).get_next_point() 
                # This will loop again if the *random* point is also invalid, which is fine.
            else: 
                # This is 'iid' or 'initial_random'
                print("Resampling...")
                point = sampler.get_next_point()

        binary_outcome, continuous_outcome, steps_taken = run_evaluation(point)
        
        # Use continuous_outcome for the surrogate model, matching initial data collection
        new_y_tensor = torch.tensor([continuous_outcome], **tkwargs)
        
        # Update the sampler with the new data
        sampler.update(point, new_y_tensor)
        
        # Record the results
        results_data.append({
            'trial': i + 1,
            'mode': current_mode,
            'x': point[0].item(),
            'y': point[1].item(),
            'orientation': None,
            'binary_outcome': binary_outcome,
            'continuous_outcome': continuous_outcome,
            'steps_taken': steps_taken,
        })
        
        # Save current results
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(args.output_file, index=False)
        print(f"Saved results for trial {i+1} to '{args.output_file}'")

    print("\nEvaluation complete.")

    # --- Save Generated Points if Requested ---
    if args.save_points:
        final_df = pd.DataFrame(results_data)
        points_df = final_df[['x', 'y']]
        points_df.to_csv(args.save_points, index=False)
        print(f"💾 Evaluation points saved to '{args.save_points}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run robot policy evaluations using IID or Active Testing."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["iid", "active", "loaded"],
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
        default=10,
        help="Number of initial random points for bootstrapping the active learning model."
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=35,
        help="Maximum number of steps allowed per evaluation."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_results.csv",
        help="Path to save the eval results as CSV file."
    )
    parser.add_argument(
        "--save_points",
        type=str,
        default=None,
        help="Path to save only the eval points to a CSV file. Note that eval points are already included in eval results."
    )
    parser.add_argument(
        "--load_path",
        type=str,
        default=None,
        help="Path to a CSV file from which to load evaluation points. Could be the full eval results CSV file or just the eval points CSV file."
    )

    args = parser.parse_args()

    if not args.load_path and args.mode=='loaded':
        parser.error("`load_path` must be specified if loading points")

    if args.num_init_pts >= args.num_evals and args.mode == 'active':
        raise ValueError("`num_init_pts` must be less than `num_evals` for active learning.")
        
    main(args)