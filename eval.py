'''
Main script for robot policy evaluation.
'''
import pandas as pd
import torch
import math
from testers import ActiveTester, IIDSampler, ListIteratorSampler

from utils import is_valid_point, run_evaluation, parse_args, get_design_points
import os  # Added for checking file existence

tkwargs = {"dtype": torch.double, "device": "cuda" if torch.cuda.is_available() else "cpu"}
# Define the search space bounds for our factors e.g. x, y, table height, etc.
BOUNDS = torch.tensor([[0.0, 0.0], [1.0, 1.0]], **tkwargs)
DIMS = BOUNDS.shape[1]

def main(args):
    """Main execution loop."""
    if args.mode == 'brute_force' and args.num_evals:
        print("Note: --num_evals is ignored in 'brute_force' mode; evaluating full design space instead.")
    
    # --- Generate full design space (if needed) ---
    # This space of evaluation points is used by 'iid', 'brute_force', and 'active' modes.
    points = None
    if args.mode in ['iid', 'brute_force', 'active']:
        # 1. Generate full design space based on --resolution
        all_points = get_design_points(args.resolution, BOUNDS, tkwargs)
        
        # 2. Filter design space to get the valid "brute-force" pool of points
        valid_points_list = [p for p in all_points if is_valid_point(p)]
        
        if not valid_points_list:
            print(f"Error: No valid points found in the {args.resolution}^D design space.")
            print("Check your 'is_valid_point' function or increase --resolution.")
            exit()
            
        points = torch.stack(valid_points_list)
        pool_size = points.shape[0]
        print(f"Filtered design space from {all_points.shape[0]} total points to {pool_size} valid points (this is the 'brute-force' pool).")

        # 3. Set or check num_evals based on the mode
        if args.mode == 'brute_force':
            print(f"Setting num_evals to the full pool size: {pool_size} (brute force mode).")
            args.num_evals = pool_size
        elif args.mode in ['active', 'iid']:
            print(f"{args.mode} mode: Running {args.num_evals} trials, sampling from the {pool_size}-point pool.")
            if args.num_evals > pool_size:
                print(f"Error: num_evals ({args.num_evals}) cannot be larger than the valid point pool size ({pool_size}).")
                print("Increase --resolution or decrease --num_evals.")
                exit()

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
        
        print(f"⏯️  Running evaluation for {args.num_evals - loop_start_index} remaining loaded points.")
    elif args.mode == 'brute_force':
        sampler = ListIteratorSampler(points)

        if loop_start_index > 0:
            print(f"Skipping the first {loop_start_index} points from the design space.")
            sampler.index = loop_start_index
        
        if loop_start_index >= args.num_evals:
            print("All points have already been evaluated. Exiting.")
            return
        
        print(f"⏯️  Running evaluation for {args.num_evals - loop_start_index} remaining points.")
    elif args.mode == 'iid':
        sampler = IIDSampler(points)
        
    elif args.mode == 'active':
        if loop_start_index >= args.num_init_pts:
            print(f"Resuming in 'active' mode with {loop_start_index} points.")
            initial_X_list = []
            for row in results_data:
                if 'factor_0' in row:
                    pt = [row[f'factor_{d}'] for d in range(DIMS)]
                elif 'x' in row and 'y' in row and DIMS == 2: # Legacy support
                    pt = [row['x'], row['y']]
                else:
                    raise ValueError(f"Could not find {DIMS} factor columns in results.")
                initial_X_list.append(torch.tensor(pt, **tkwargs))

            initial_X_tensors = initial_X_list
            initial_Y_tensors = [torch.tensor([row['continuous_outcome']], **tkwargs) for row in results_data]
            train_X = torch.stack(initial_X_tensors)
            train_Y = torch.stack(initial_Y_tensors)
            sampler = ActiveTester(train_X, train_Y, BOUNDS, points, mc_points=None, 
                                   model_name=args.model_name, acq_func_name=args.acq_func_name,
                                   vla_data_path=args.vla_data_path, ood_metric=args.ood_metric)
        else:
            print("Not enough data for active learning yet. Starting with initial random sampling.")
            sampler = IIDSampler(points)

    # --- Main Evaluation Loop ---
    print(f"\n--- Starting main evaluation loop ---")
    print(f"Total trials: {args.num_evals}, Resuming from trial: {loop_start_index + 1}")
    for i in range(loop_start_index, args.num_evals):
        
        current_mode = args.mode
        if args.mode == 'active':
            current_mode = 'initial_random' if i < args.num_init_pts else 'active'

        # --- Handle sampler transition for 'active' mode ---
        if args.mode == 'active' and i == args.num_init_pts:
            print("\n" + "="*50)
            print(f"Reached {args.num_init_pts} initial points. Switching to Active Testing.")
            print("="*50 + "\n")
            
            initial_X_list = []
            for row in results_data:
                if 'factor_0' in row:
                    pt = [row[f'factor_{d}'] for d in range(DIMS)]
                elif 'x' in row and 'y' in row and DIMS == 2: # Legacy support
                    pt = [row['x'], row['y']]
                else:
                    raise ValueError(f"Could not find {DIMS} factor columns in results.")
                initial_X_list.append(torch.tensor(pt, **tkwargs))

            initial_X_tensors = initial_X_list
            initial_Y_tensors = [torch.tensor([row['continuous_outcome']], **tkwargs) for row in results_data]
            train_X = torch.stack(initial_X_tensors)
            train_Y = torch.stack(initial_Y_tensors)
            sampler = ActiveTester(train_X, train_Y, BOUNDS, points, mc_points=None, 
                                   model_name=args.model_name, acq_func_name=args.acq_func_name,
                                   vla_data_path=args.vla_data_path, ood_metric=args.ood_metric)

        print(f"\nTrial {i+1}/{args.num_evals} (mode: {current_mode})")

        # We seed based on 'i' (the number of points currently in the dataset)
        # to ensure the stochastic 'get_optimal_samples' in ActiveTester
        # behaves identically here and in the visualization script.
        if current_mode == 'active':
            torch.manual_seed(i+1)
            print(f"Using seed {i+1}")
        
        point = sampler.get_next_point()
        
        # --- Robust validity check ---
        while not is_valid_point(point):
            print(f"Point {point.tolist()} is not valid -> handling...")
            if args.mode in ['loaded', 'brute_force']:
                print(f"Error: The loaded point from trial {i+1} is invalid. Please check your source file.")
                print("Stopping evaluation.")
                exit()
            elif current_mode == 'active':
                # Fall back to a single random sample for this trial to avoid an infinite loop
                print("Warning: Active learner suggested an invalid point. Falling back to IID sampling for this one trial.")
                point = IIDSampler(points).get_next_point()
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
        entry = {
            'trial': i + 1,
            'mode': current_mode,
            'binary_outcome': binary_outcome,
            'continuous_outcome': continuous_outcome,
            'steps_taken': steps_taken,
        }
        # Dynamically save all factors (e.g. x, y, table height, lighting, etc.)
        for dim_idx in range(point.shape[0]):
            # can add some number->name mapping so it doesn't just say factor_1, factor_2, etc.
            entry[f'factor_{dim_idx}'] = point[dim_idx].item()
        results_data.append(entry)

        # Save current results
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(args.output_file, index=False)
        print(f"Saved results for trial {i+1} to '{args.output_file}'")

    print("\nEvaluation complete.")

    # --- Save generated points if requested ---
    if args.save_points:
        final_df = pd.DataFrame(results_data)
        # Select columns that start with 'factor_' or fallback to x,y
        cols_to_save = [c for c in final_df.columns if c.startswith('factor_')]
        if not cols_to_save and 'x' in final_df.columns:
            cols_to_save = ['x', 'y']
            
        points_df = final_df[cols_to_save]
        points_df.to_csv(args.save_points, index=False)
        print(f"💾 Evaluation points saved to '{args.save_points}'.")


if __name__ == "__main__":
    args = parse_args()
    main(args)