'''
Main script for robot policy evaluation.
'''
import pandas as pd
import torch
import math
from samplers import ActiveTester, IIDSampler, ListIteratorSampler

from utils import is_valid_point, run_evaluation, parse_args, get_grid_points
import os  # Added for checking file existence

tkwargs = {"dtype": torch.double, "device": "cpu"}

def main(args):
    """Main execution loop."""
    if not math.sqrt(args.num_evals).is_integer():
        raise ValueError("num_evals must be a perfect square for grid generation.")
    
    print(f"Starting evaluation with mode: '{args.mode}' for {args.num_evals} trials.")
    
    # Define the search space bounds for our factors [x, y]
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], **tkwargs)
    
    # --- Generate discrete grid points (if needed) ---
    # This grid is now used by 'iid' and 'brute_force' modes,
    # and for the initial points in 'active' mode.
# --- Generate discrete grid points (if needed) ---
    # This grid is now used by 'iid' and 'brute_force' modes,
    # and for the initial points in 'active' mode.
    grid_points = None
    if args.mode in ['iid', 'brute_force', 'active']:
        all_grid_points = get_grid_points(int(math.sqrt(args.num_evals)), bounds, tkwargs)
        
        # --- Filter grid points based on validity ---
        print("Filtering grid points using is_valid_point()...")
        valid_points_list = [p for p in all_grid_points if is_valid_point(p)]
        
        if not valid_points_list:
            print(f"Error: No valid points found in the {args.grid_resolution}x{args.grid_resolution} grid.")
            print("Check your 'is_valid_point' function or increase --grid_resolution.")
            exit()
            
        grid_points = torch.stack(valid_points_list)
        print(f"Filtered grid from {all_grid_points.shape[0]} total points to {grid_points.shape[0]} valid points.")
        # --- End of filtering ---

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
    
    if loop_start_index >= args.num_evals and args.mode not in ['loaded', 'brute_force']:
        # (Slight change: don't exit if brute_force)
        print(f"Evaluation already complete with {loop_start_index} trials. Exiting.")
        return

    # --- Initialize Sampler ---
    sampler = None
    if args.mode == 'loaded':
        sampler = ListIteratorSampler(args.load_path)
        args.num_evals = len(sampler.points)  # Override num_evals
        if loop_start_index > 0:
            print(f"Skipping the first {loop_start_index} points from the loaded file.")
            sampler.index = loop_start_index
        
        if loop_start_index >= args.num_evals:
            print("All points from the loaded file have already been evaluated. Exiting.")
            return
        
        print(f"Running evaluation for {args.num_evals - loop_start_index} remaining loaded points.")
    elif args.mode == 'brute_force':
        sampler = ListIteratorSampler(grid_points)
        args.num_evals = len(sampler.points)  # Override num_evals
        if loop_start_index > 0:
            print(f"Skipping the first {loop_start_index} points from the grid.")
            sampler.index = loop_start_index
        
        if loop_start_index >= args.num_evals:
            print("All grid points have already been evaluated. Exiting.")
            return
        
        print(f"Running evaluation for {args.num_evals - loop_start_index} remaining grid points.")
    elif args.mode == 'iid':
        sampler = IIDSampler(grid_points)
        
    elif args.mode == 'active':
        if loop_start_index >= args.num_init_pts:
            print(f"Resuming in 'active' mode with {loop_start_index} points.")
            initial_X_tensors = [torch.tensor([row['x'], row['y']], **tkwargs) for row in results_data]
            initial_Y_tensors = [torch.tensor([row['continuous_outcome']], **tkwargs) for row in results_data]
            train_X = torch.stack(initial_X_tensors)
            train_Y = torch.stack(initial_Y_tensors)
            sampler = ActiveTester(train_X, train_Y, grid_points)
        else:
            print("Not enough data for active learning yet. Starting with initial random sampling.")
            sampler = IIDSampler(grid_points)

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
            sampler = ActiveTester(train_X, train_Y, grid_points)

        print(f"\nTrial {i+1}/{args.num_evals} (mode: {current_mode})")
        
        point = sampler.get_next_point()
        
        # --- Robust validity check ---
        while not is_valid_point(point):
            print(f"Point (x={point[0]:.3f}, y={point[1]:.3f}) is not valid -> handling...")
            if args.mode in ['loaded', 'brute_force']:
                print(f"Error: The loaded point from trial {i+1} is invalid. Please check your source file.")
                print("Stopping evaluation.")
                exit()
            elif current_mode == 'active':
                # Fall back to a single random sample for this trial to avoid an infinite loop
                print("Warning: Active learner suggested an invalid point. Falling back to IID sampling for this one trial.")
                point = IIDSampler(grid_points).get_next_point()
                # This will loop again if the *random* point is also invalid, which is fine.
            else: 
                # This is 'iid' or 'initial_random'
                print("Resampling...")
                point = sampler.get_next_point()

        binary_outcome, continuous_outcome, steps_taken = run_evaluation(point, args.max_steps)
        
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
    args = parse_args()
    main(args)