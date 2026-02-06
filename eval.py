'''
Main script for robot policy evaluation.
'''
import pandas as pd
import torch
import math
from testers import ActiveTester, IIDSampler, ListIteratorSampler

from utils import run_evaluation, parse_args, fit_surrogate_model
from factors_config import (
    BOUNDS, DIMS, tkwargs,
    get_design_points_robot, is_valid_point,
    FACTOR_COLUMNS, get_task_config, get_outcome_range, get_success_outcome, get_outcome_descriptions
)
import os  # Added for checking file existence
import uuid
from datetime import datetime

def main(args):
    """Main execution loop."""
    # --- Set up evaluation ID and folder structure ---
    # If output_file is provided and it's in results/{eval_id}/ format, extract eval_id
    if args.eval_id is None and args.output_file:
        # Check if output_file is in the new structure: results/{eval_id}/results.csv
        output_path_parts = os.path.normpath(args.output_file).split(os.sep)
        if len(output_path_parts) >= 2 and output_path_parts[-2] != 'results' and 'results' in output_path_parts:
            # Find 'results' in the path and get the next directory as eval_id
            results_idx = output_path_parts.index('results')
            if results_idx + 1 < len(output_path_parts):
                args.eval_id = output_path_parts[results_idx + 1]
    
    # If still no eval_id, auto-generate it
    if args.eval_id is None:
        # Auto-generate eval_id from timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.eval_id = f"eval_{timestamp}_{uuid.uuid4().hex[:8]}"
    
    # Create results directory structure
    results_base_dir = "results"
    eval_dir = os.path.join(results_base_dir, args.eval_id)
    models_dir = os.path.join(eval_dir, "models")
    
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"📁 Evaluation ID: {args.eval_id}")
    print(f"📁 Results directory: {eval_dir}")
    print(f"📁 Models directory: {models_dir}")
    
    # Set up output file paths
    if args.output_file is None:
        args.output_file = os.path.join(eval_dir, "results.csv")
    else:
        # If output_file is provided, ensure it's in the eval_dir
        # But preserve the original path for resuming if it exists
        if not os.path.exists(args.output_file):
            # New file - put it in eval_dir
            output_filename = os.path.basename(args.output_file)
            args.output_file = os.path.join(eval_dir, output_filename)
        # If file exists, keep the original path (for resuming)
    
    if args.save_points is None:
        # Don't auto-create save_points, but if user wants it, put it in eval_dir
        pass
    elif not os.path.isabs(args.save_points) and not os.path.dirname(args.save_points):
        # If just a filename, put it in eval_dir
        args.save_points = os.path.join(eval_dir, args.save_points)
    
    if args.mode == 'brute_force' and args.num_evals:
        print("Note: --num_evals is ignored in 'brute_force' mode; evaluating full design space instead.")
    
    # --- Generate full design space (if needed) ---
    # This space of evaluation points is used by 'iid', 'brute_force', and 'active' modes.
    points = None
    if args.mode in ['iid', 'brute_force', 'active']:
        # 1. Generate full design space (discrete combinations only)
        # Note: --resolution is ignored since we use fixed discrete values
        all_points = get_design_points_robot()
        
        # 2. Filter design space to get the valid "brute-force" pool of points
        valid_points_list = [p for p in all_points if is_valid_point(p)]
        
        if not valid_points_list:
            print(f"Error: No valid points found in the design space.")
            print("Check your 'is_valid_point' function.")
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
                print("Decrease --num_evals (design space is fixed at 11x11x3x3 = 1089 points).")
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
                if set(FACTOR_COLUMNS).issubset(row.keys()):
                    # Factor format
                    pt = [row[col] for col in FACTOR_COLUMNS]
                else:
                    raise ValueError(f"Could not find factor columns in results. Expected {FACTOR_COLUMNS} columns.")
                initial_X_list.append(torch.tensor(pt, **tkwargs))

            initial_X_tensors = initial_X_list
            initial_Y_tensors = [torch.tensor([row['continuous_outcome']], **tkwargs) for row in results_data]
            train_X = torch.stack(initial_X_tensors)
            train_Y = torch.stack(initial_Y_tensors)
            sampler = ActiveTester(train_X, train_Y, BOUNDS, points, mc_points=None, 
                                   model_name=args.model_name, acq_func_name=args.acq_func_name,
                                   training_data_factors_path=args.training_data_factors_path, ood_metric=args.ood_metric,
                                   use_train_data_for_surrogate=args.use_train_data_for_surrogate,
                                   task_name=args.task)
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
                if set(FACTOR_COLUMNS).issubset(row.keys()):
                    # Factor format
                    pt = [row[col] for col in FACTOR_COLUMNS]
                else:
                    raise ValueError(f"Could not find factor columns in results. Expected {FACTOR_COLUMNS} columns.")
                initial_X_list.append(torch.tensor(pt, **tkwargs))

            initial_X_tensors = initial_X_list
            initial_Y_tensors = [torch.tensor([row['continuous_outcome']], **tkwargs) for row in results_data]
            train_X = torch.stack(initial_X_tensors)
            train_Y = torch.stack(initial_Y_tensors)
            sampler = ActiveTester(train_X, train_Y, BOUNDS, points, mc_points=None, 
                                   model_name=args.model_name, acq_func_name=args.acq_func_name,
                                   training_data_factors_path=args.training_data_factors_path, ood_metric=args.ood_metric,
                                   use_train_data_for_surrogate=args.use_train_data_for_surrogate,
                                   task_name=args.task)

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

        binary_outcome, continuous_outcome, steps_taken = run_evaluation(point, args.max_steps, args.task)
        
        # Use continuous_outcome for the surrogate model, matching initial data collection
        new_y_tensor = torch.tensor([continuous_outcome], **tkwargs)
        
        # Update the sampler with the new data
        sampler.update(point, new_y_tensor)
        
        # Save model at this trial (for both active and IID modes)
        # Only save if model_name is specified (needed for visualization)
        if hasattr(args, 'model_name') and args.model_name is not None:
            import pickle
            model_save_path = os.path.join(models_dir, f'trial_{i+1}_model.pkl')
            try:
                # For active mode, model is already fitted in get_next_point
                # For IID mode, we need to fit a model now
                if args.mode == 'active' and hasattr(sampler, 'model') and sampler.model is not None:
                    # Model already fitted, just save it
                    model_to_save = sampler.model
                    train_X_to_save = sampler.train_X
                    train_Y_to_save = sampler.train_Y
                elif args.mode == 'iid':
                    # Fit model for IID mode
                    torch.manual_seed(i+1)  # Match eval.py seed behavior
                    train_X_list = []
                    for row in results_data:
                        if set(FACTOR_COLUMNS).issubset(row.keys()):
                            pt = [row[col] for col in FACTOR_COLUMNS]
                            train_X_list.append(torch.tensor(pt, **tkwargs))
                    train_X_list.append(point)
                    train_X_to_save = torch.stack(train_X_list)
                    train_Y_to_save = torch.stack([torch.tensor([row['continuous_outcome']], **tkwargs) for row in results_data] + [new_y_tensor])
                    
                    if len(train_X_to_save) >= 2:
                        model_to_save = fit_surrogate_model(train_X_to_save, train_Y_to_save, BOUNDS, model_name=args.model_name)
                    else:
                        model_to_save = None
                else:
                    model_to_save = None
                
                if model_to_save is not None:
                    with open(model_save_path, 'wb') as f:
                        pickle.dump({
                            'model': model_to_save,
                            'model_name': args.model_name,
                            'acq_func_name': args.acq_func_name if args.mode == 'active' else None,
                            'train_X': train_X_to_save,
                            'train_Y': train_Y_to_save,
                            'bounds': BOUNDS,
                            'task_name': args.task,
                            'training_data_factors_path': args.training_data_factors_path,
                            'ood_metric': args.ood_metric,
                            'use_train_data_for_surrogate': args.use_train_data_for_surrogate,
                            'trial': i + 1,
                            'mode': current_mode,
                        }, f)
            except Exception as e:
                print(f"Warning: Could not save model for trial {i+1}: {e}")
        
        # Record the results
        entry = {
            'trial': i + 1,
            'mode': current_mode,
            'binary_outcome': binary_outcome,
            'continuous_outcome': continuous_outcome,
            'steps_taken': steps_taken,
        }
        # Save all factors using named columns from config
        for idx, col_name in enumerate(FACTOR_COLUMNS):
            entry[col_name] = point[idx].item()
        results_data.append(entry)

        # Save current results
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(args.output_file, index=False)
        print(f"Saved results for trial {i+1} to '{args.output_file}'")

    print("\nEvaluation complete.")

    # --- Save final model if in active mode ---
    if args.mode == 'active' and hasattr(sampler, 'model') and sampler.model is not None:
        import pickle
        model_save_path = os.path.join(models_dir, 'final_model.pkl')
        try:
            with open(model_save_path, 'wb') as f:
                pickle.dump({
                    'model': sampler.model,
                    'model_name': args.model_name,
                    'acq_func_name': args.acq_func_name,
                    'train_X': sampler.train_X,
                    'train_Y': sampler.train_Y,
                    'bounds': BOUNDS,
                    'task_name': args.task,
                    'training_data_factors_path': args.training_data_factors_path,
                    'ood_metric': args.ood_metric,
                    'use_train_data_for_surrogate': args.use_train_data_for_surrogate,
                }, f)
            print(f"💾 Final model saved to '{model_save_path}'.")
        except Exception as e:
            print(f"Warning: Could not save model: {e}")

    # --- Save generated points if requested ---
    if args.save_points:
        final_df = pd.DataFrame(results_data)
        # Save only the factor columns from config
        points_df = final_df[FACTOR_COLUMNS]
        points_df.to_csv(args.save_points, index=False)
        print(f"💾 Evaluation points saved to '{args.save_points}'.")


if __name__ == "__main__":
    args = parse_args()
    main(args)