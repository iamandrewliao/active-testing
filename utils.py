import torch
import argparse

tkwargs = {"dtype": torch.double, "device": "cpu"}

def get_grid_points(resolution, bounds, tkwargs):
    """
    Creates a tensor of all points on a uniform N_x_N grid.
    
    Returns:
        torch.Tensor: A tensor of shape [resolution*resolution, 2]
    """
    x_lin = torch.linspace(bounds[0, 0], bounds[1, 0], resolution, **tkwargs)
    y_lin = torch.linspace(bounds[0, 1], bounds[1, 1], resolution, **tkwargs)
    
    # Use torch.meshgrid
    grid_y, grid_x = torch.meshgrid(y_lin, x_lin, indexing='ij')
    
    # Stack and reshape to get [N*N, 2]
    grid_tensor = torch.stack([grid_x, grid_y], dim=-1)
    all_points = grid_tensor.reshape(-1, 2)
    
    print(f"Generated {all_points.shape[0]} total grid points ({resolution}x{resolution}).")
    return all_points


def is_valid_point(point):
    """Automatically filters out any point past Lightning's reachability (approximated by a straight line)"""
    # Equation of the boundary line: y = 1.35x - 0.475
    # A point is 'invalid' if it's on the right side of the line
    x, y = point[0].item(), point[1].item()
    # Calculate if the point is in the invalid region
    is_invalid = (y <= 1.35*x - 0.475) and (x >= 0.35)
    # is_invalid = (y <= 1.5*x - 0.75) and (x >= 0.5)
    
    return not is_invalid  # Return True if the point is valid


def run_evaluation(point, max_steps):
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
        steps_taken = max_steps
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

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run robot policy evaluations using IID or Active Testing."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["iid", "active", "loaded", "brute_force"],
        required=True,
        help="The sampling strategy to use."
    )
    parser.add_argument(
        "--grid_resolution",
        type=int,
        default=10,
        help="The resolution (N) for the N_x_N grid in 'brute_force' or 'iid' mode."
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
    
    return args