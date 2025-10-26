"""Visualization functions"""

import argparse
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Import necessary BoTorch/GPyTorch components
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.joint_entropy_search import qJointEntropySearch
from botorch.acquisition.utils import get_optimal_samples

from utils import get_grid_points

# Set up torch device and data type
tkwargs = {"dtype": torch.double, "device": "cpu"}
# Define the search space bounds (hardcoded to match eval.py)
BOUNDS = torch.tensor([[0.0, 0.0], [1.0, 1.0]], **tkwargs)


def _load_data(filepath):
    """Loads evaluation data from a CSV file."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        exit()
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading file: {e}")
        exit()

def _get_tensors_from_df(df):
    """Extracts and formats training tensors from a DataFrame."""
    train_X = torch.tensor(df[['x', 'y']].values, **tkwargs)
    train_Y = torch.tensor(df['continuous_outcome'].values, **tkwargs).unsqueeze(-1) # Shape [N, 1]
    return train_X, train_Y

def _fit_gp_model(train_X, train_Y):
    """Fits a SingleTaskGP to the provided data."""
    # Need at least 2 points to fit outcome transform reliably
    if train_X.shape[0] < 2:
        # Fit without outcome transform if only 1 point
        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            input_transform=Normalize(d=train_X.shape[-1], bounds=BOUNDS),
        )
    else:
        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            input_transform=Normalize(d=train_X.shape[-1], bounds=BOUNDS),
            outcome_transform=Standardize(m=1), # Standardize Y
        )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def plot_tested_points(df, output_file):
    """
    Plots a simple scatter plot of tested points, colored by their outcome.
    """
    print(f"Generating tested points plot -> {output_file}...")
    plt.figure(figsize=(10, 8))

    # Scatter plot colored by continuous outcome
    sc = plt.scatter(
        df['x'],
        df['y'],
        c=df['continuous_outcome'],
        cmap='viridis_r',  # Use _r for "reversed" (low=bad, high=good)
        vmin=0,
        vmax=4,
        edgecolors='k',
        alpha=0.8
    )

    plt.colorbar(sc, label='Continuous Outcome (0-4)')
    plt.title(f"Tested Points and Outcomes (N={len(df)})")
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.xlim(BOUNDS[0, 0].item(), BOUNDS[1, 0].item()) # Use .item() for scalar bounds
    plt.ylim(BOUNDS[0, 1].item(), BOUNDS[1, 1].item())
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print("Done.")


def plot_active_learning(df, output_file, grid_resolution):
    """
    Plots the surrogate model mean and acquisition function landscape
    for an active testing run.
    """
    print(f"Generating active learning plots -> {output_file}...")

    # 1. Fit the model
    train_X, train_Y = _get_tensors_from_df(df)
    if train_X.shape[0] < 1:
        print("Error: No data points found in the results file. Cannot generate plots.")
        return
    model = _fit_gp_model(train_X, train_Y)

    # 2. Get the acquisition function (JES) - requires fitted model
    normalized_bounds = torch.tensor([[0.0] * 2, [1.0] * 2], **tkwargs) # Bounds for normalized space
    # JES requires >1 point for outcome standardization to work properly
    if train_X.shape[0] >= 2:
        try:
            optimal_inputs, optimal_outputs = get_optimal_samples(
                model, bounds=normalized_bounds, num_optima=16
            )
            jes = qJointEntropySearch(
                model=model,
                optimal_inputs=optimal_inputs,
                optimal_outputs=optimal_outputs,
                estimation_type="LB",
            )
        except Exception as e:
            print(f"Warning: Could not compute JES: {e}. Acquisition plot will be empty.")
            jes = None # Flag that JES is unavailable
    else:
        print("Warning: Need at least 2 data points to compute JES. Acquisition plot will be empty.")
        jes = None

    # 3. Create grid and evaluate
    grid_tensor = get_grid_points(grid_resolution, BOUNDS, tkwargs)
    grid_shape = (grid_resolution, grid_resolution)

    # Get model predictions
    mean_values = np.full(grid_shape, np.nan)
    acq_values = np.full(grid_shape, np.nan)

    with torch.no_grad():
        posterior = model.posterior(grid_tensor)
        mean_values = posterior.mean.numpy().reshape(grid_shape)

        # Get acquisition values if JES was computed
        if jes is not None:
             acq_values = jes(grid_tensor.reshape(-1, 1, 2)).numpy().reshape(grid_shape)

    # 4. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    plot_extent = [BOUNDS[0, 0].item(), BOUNDS[1, 0].item(), BOUNDS[0, 1].item(), BOUNDS[1, 1].item()] # Use .item()

    # --- Plot 1: Surrogate Model Mean ---
    ax = axes[0]
    im = ax.imshow(mean_values, origin='lower', extent=plot_extent, cmap='viridis_r', vmin=0, vmax=4, aspect='equal')
    fig.colorbar(im, ax=ax, label='Predicted Outcome (Mean)')
    # Overlay tested points
    ax.scatter(df['x'], df['y'], c='red', edgecolors='k', s=50, label='Tested Points')
    ax.set_title(f'Surrogate Model (GP Mean) (N={len(df)})')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()

    # --- Plot 2: Acquisition Function ---
    ax = axes[1]
    # Display NaN as gray or a specific color if desired
    cmap_acq = plt.cm.magma.copy()
    cmap_acq.set_bad(color='gray') # Show NaN values as gray
    im = ax.imshow(acq_values, origin='lower', extent=plot_extent, cmap=cmap_acq, aspect='equal')
    fig.colorbar(im, ax=ax, label='Acquisition Value (JES)')
    # Overlay tested points
    ax.scatter(df['x'], df['y'], c='cyan', edgecolors='k', s=50, label='Tested Points')
    ax.set_title('Acquisition Function Landscape')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()

    plt.suptitle('Active Testing Diagnostics', fontsize=16, y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])

    # Save the combined plot
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print("Done.")


def animate_active_learning(df, output_file, grid_resolution, interval=500):
    """
    Generates an MP4 animation showing the evolution of the surrogate model
    and acquisition function during an active testing run, using the 'mode' column.
    """
    print(f"Generating active learning animation -> {output_file}...")
    total_trials = len(df)
    if total_trials < 1: # Need at least 1 point to show anything
        print("Error: Need at least 1 data point to create an animation.")
        return

    # --- Setup Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    plot_extent = [BOUNDS[0, 0].item(), BOUNDS[1, 0].item(), BOUNDS[0, 1].item(), BOUNDS[1, 1].item()] # Use .item()
    grid_tensor = get_grid_points(grid_resolution, BOUNDS, tkwargs)
    grid_shape = (grid_resolution, grid_resolution)
    normalized_bounds = torch.tensor([[0.0] * 2, [1.0] * 2], **tkwargs) # For JES

    # Placeholder artists for updating
    im_mean = axes[0].imshow(np.zeros(grid_shape), origin='lower', extent=plot_extent, cmap='viridis_r', vmin=0, vmax=4, aspect='equal')
    sc_mean = axes[0].scatter([], [], c='red', edgecolors='k', s=50, label='Tested Points')
    sc_next_mean = axes[0].scatter([], [], c='lime', edgecolors='k', s=150, marker='*', label='Next Acquired', zorder=10) # Star for last point
    cb_mean = fig.colorbar(im_mean, ax=axes[0], label='Predicted Outcome (Mean)')
    axes[0].set_xlabel('X Position')
    axes[0].set_ylabel('Y Position')
    axes[0].legend(loc='upper right')
    axes[0].set_xlim(plot_extent[0], plot_extent[1])
    axes[0].set_ylim(plot_extent[2], plot_extent[3])


    cmap_acq = plt.cm.magma.copy()
    cmap_acq.set_bad(color='gray') # Show NaN values as gray
    im_acq = axes[1].imshow(np.zeros(grid_shape), origin='lower', extent=plot_extent, cmap=cmap_acq, aspect='equal')
    sc_acq = axes[1].scatter([], [], c='cyan', edgecolors='k', s=50, label='Tested Points')
    sc_next_acq = axes[1].scatter([], [], c='yellow', edgecolors='k', s=150, marker='*', label='Next Acquired', zorder=10) # Star for last point
    cb_acq = fig.colorbar(im_acq, ax=axes[1], label='Acquisition Value (JES)')
    axes[1].set_xlabel('X Position')
    axes[1].set_ylabel('Y Position')
    axes[1].legend(loc='upper right')
    axes[1].set_xlim(plot_extent[0], plot_extent[1])
    axes[1].set_ylim(plot_extent[2], plot_extent[3])


    # --- Animation Update Function ---
    def update(frame):
        # Frame 0 shows state *after* trial 1, frame 1 after trial 2, etc.
        current_trial_index = frame + 1 # 1-based index for display
        print(f"  Processing frame {current_trial_index}/{total_trials}...") # Print correct frame number

        # Get data up to and including the point acquired in this frame's trial
        current_df = df.iloc[:current_trial_index]
        train_X, train_Y = _get_tensors_from_df(current_df)

        # Get the point acquired in this trial (the last point in current_df)
        last_point = current_df.iloc[-1]
        next_point_x, next_point_y = last_point['x'], last_point['y']
        # --- Get mode directly from the dataframe ---
        current_mode = last_point['mode']

        # Clear previous next point highlight only
        sc_next_mean.set_offsets(np.empty((0, 2)))
        sc_next_acq.set_offsets(np.empty((0, 2)))

        # Handle fitting the model (need >= 1 point)
        mean_values = np.full(grid_shape, np.nan) # Default to NaN
        acq_values = np.full(grid_shape, np.nan)

        # Always fit the model if we have at least one point
        model = _fit_gp_model(train_X, train_Y)
        with torch.no_grad():
            posterior = model.posterior(grid_tensor)
            mean_values = posterior.mean.numpy().reshape(grid_shape)

        # Try to calculate JES if we have enough points (>= 2)
        if train_X.shape[0] >= 2:
            try:
                optimal_inputs, optimal_outputs = get_optimal_samples(
                    model, bounds=normalized_bounds, num_optima=16
                )
                jes = qJointEntropySearch(
                    model=model,
                    optimal_inputs=optimal_inputs,
                    optimal_outputs=optimal_outputs,
                    estimation_type="LB",
                )
                with torch.no_grad():
                     acq_values = jes(grid_tensor.reshape(-1, 1, 2)).numpy().reshape(grid_shape)
            except Exception as e:
                # Keep acq_values as NaN if JES calculation fails
                if current_trial_index == 2: # Only print warning once if it happens early
                     print(f"    Warning: Could not compute JES at trial {current_trial_index}: {e}")

        # Update Image plots
        im_mean.set_data(mean_values)
        if not np.all(np.isnan(mean_values)): # Avoid error if all NaN
             im_mean.set_clim(vmin=np.nanmin(mean_values), vmax=np.nanmax(mean_values))
        else:
             im_mean.set_clim(vmin=0, vmax=4) # Default if all NaN

        im_acq.set_data(acq_values)
        if not np.all(np.isnan(acq_values)):
             im_acq.set_clim(vmin=np.nanmin(acq_values), vmax=np.nanmax(acq_values))
        # No else needed, cmap handles NaN color


        # Update Scatter plots - more efficient update
        points_so_far = current_df[['x', 'y']].values
        sc_mean.set_offsets(points_so_far)
        sc_acq.set_offsets(points_so_far)

        # Highlight the point acquired in *this* trial
        sc_next_mean.set_offsets([[next_point_x, next_point_y]])
        sc_next_acq.set_offsets([[next_point_x, next_point_y]])

        # Update Titles
        axes[0].set_title(f'Surrogate Model (GP Mean) (After Trial {current_trial_index})')
        axes[1].set_title(f'Acquisition Function (JES) (Mode: {current_mode})') # Show mode here

        fig.suptitle(f'Active Testing Evolution (Trial {current_trial_index}/{total_trials})', fontsize=16, y=0.95)

        # Return *all* artists that were modified
        return [im_mean, sc_mean, sc_next_mean, im_acq, sc_acq, sc_next_acq]


    # --- Create and Save Animation ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.92]) # Apply layout once before starting animation
    # We need total_trials frames (frame 0 = after trial 1, ..., frame N-1 = after trial N)
    # Use blit=False for potentially more robust updates, especially with text/titles changing
    ani = animation.FuncAnimation(fig, update, frames=total_trials, interval=interval, blit=True)

    # Save the animation
    try:
        # Increase dpi for better quality if needed
        ani.save(output_file, writer='ffmpeg', fps=max(1, 1000//interval), dpi=150)
        print(f"Animation saved successfully to '{output_file}'.")
    except Exception as e:
        print(f"\nError saving animation: {e}")
        print("Ensure ffmpeg is installed and accessible in your system's PATH.")
        print("Alternatively, try installing matplotlib with a movie writer: 'uv pip install matplotlib[animation]'")

    plt.close(fig) # Close the figure after saving


def plot_comparison(results_list, gt_df, output_file, grid_resolution):
    """
    Generates a plot comparing model errors from one or more runs
    (e.g., Active, IID) against a ground truth model.

    Args:
        results_list (list): A list of tuples, e.g., [('Active', active_df), ('IID', iid_df)]
        gt_df (pd.DataFrame): The ground truth data.
        output_file (str): Path to save the plot.
    """
    print(f"Generating comparison plot -> {output_file}...")

    # 1. Fit Ground Truth model and get grid predictions
    X_gt, Y_gt = _get_tensors_from_df(gt_df)
    print(f"  Fitting Ground Truth model (N={len(X_gt)})...")
    if len(X_gt) < 1:
         print("Error: Ground truth data is empty. Cannot generate comparison plot.")
         return
    model_gt = _fit_gp_model(X_gt, Y_gt)

    grid_tensor = get_grid_points(grid_resolution, BOUNDS, tkwargs)
    grid_shape = (grid_resolution, grid_resolution)
    with torch.no_grad():
        mean_gt = model_gt.posterior(grid_tensor).mean.numpy().reshape(grid_shape)

    # 2. Fit each model, get predictions, and calculate error
    errors_and_stats = []
    for label, df in results_list:
        X_model, Y_model = _get_tensors_from_df(df)
        if len(X_model) < 1:
            print(f"  Skipping {label} model (N=0).")
            continue
        print(f"  Fitting {label} model (N={len(X_model)})...")
        model = _fit_gp_model(X_model, Y_model)

        with torch.no_grad():
            mean_model = model.posterior(grid_tensor).mean.numpy().reshape(grid_shape)

        error_map = np.abs(mean_model - mean_gt)
        # Calculate MAE only over valid (non-NaN) error values if GT or model had issues
        mae = np.nanmean(error_map)
        print(f"    Overall MAE ({label}): {mae:.4f}")

        errors_and_stats.append({
            'label': label,
            'n': len(X_model),
            'error_map': error_map,
            'mae': mae
        })

    # 3. Plotting
    num_models = len(errors_and_stats)
    num_plots = num_models + 1 # One plot for each model error + one for GT

    fig, axes = plt.subplots(1, num_plots, figsize=(9 * num_plots, 9))
    # Ensure 'axes' is always an array/list for consistent indexing
    if num_plots == 0:
        print("Error: No models provided or fit successfully for comparison.")
        return
    elif num_plots == 1:
        axes = [axes] # Make it a list if only GT plot
    else:
        axes = axes.flatten() # Make sure axes is 1D array if multiple plots

    plot_extent = [BOUNDS[0, 0].item(), BOUNDS[1, 0].item(), BOUNDS[0, 1].item(), BOUNDS[1, 1].item()] # Use .item()

    # Determine a common error scale (if there are any error plots)
    vmax_error = 0
    if errors_and_stats:
        # Filter out potential all-NaN maps before finding max
        valid_maxes = [np.nanmax(stats['error_map']) for stats in errors_and_stats if not np.all(np.isnan(stats['error_map']))]
        if valid_maxes:
            vmax_error = max(valid_maxes)
        if vmax_error == 0 or not valid_maxes: vmax_error=1.0 # Avoid range issue if error is zero or no valid errors

    # --- Plot 1...N: Model Errors ---
    for i, stats in enumerate(errors_and_stats):
        ax = axes[i]
        cmap_err = plt.cm.hot.copy()
        cmap_err.set_bad(color='gray')
        im = ax.imshow(stats['error_map'], origin='lower', extent=plot_extent, cmap=cmap_err, vmin=0, vmax=vmax_error, aspect='equal')
        fig.colorbar(im, ax=ax, label='Absolute Error')
        ax.set_title(f"{stats['label']} Model Error (N={stats['n']})\nGrid MAE = {stats['mae']:.4f}")
        ax.set_xlabel('X Position')
        if i == 0:
            ax.set_ylabel('Y Position') # Only set Y label on the first plot

    # --- Plot N+1: Ground Truth Model ---
    ax_gt = axes[num_models] # Last axis is always GT
    cmap_gt = plt.cm.viridis_r.copy()
    cmap_gt.set_bad(color='gray')
    im_gt = ax_gt.imshow(mean_gt, origin='lower', extent=plot_extent, cmap=cmap_gt, vmin=0, vmax=4, aspect='equal')
    fig.colorbar(im_gt, ax=ax_gt, label='Predicted Outcome (Mean)')
    ax_gt.set_title(f'Ground Truth Model (N={len(X_gt)})')
    ax_gt.set_xlabel('X Position')
    if num_models == 0: # If only GT is plotted
        ax_gt.set_ylabel('Y Position')

    plt.suptitle(f'Model Comparison vs. Ground Truth (N={len(X_gt)})', fontsize=16, y=0.96)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Visualization script for evaluation results.")
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', required=True, help='Visualization command to run')

    # --- Command: plot-points ---
    parser_points = subparsers.add_parser('plot-points', help='Plot tested points colored by outcome')
    parser_points.add_argument('--results_file', type=str, required=True, help='Path to the evaluation_results.csv file')
    parser_points.add_argument('--output_file', type=str, default='visualizations/tested_points.png', help='Output file path for the plot')

    # --- Command: plot-active ---
    parser_active = subparsers.add_parser('plot-active', help='Plot active learning diagnostics (model mean, acqf)')
    parser_active.add_argument(
        "--grid_resolution", type=int, default=10, help="The resolution (N) for the N_x_N grid."
    )
    parser_active.add_argument('--results_file', type=str, required=True, help='Path to the active_results.csv file')
    parser_active.add_argument('--output_file', type=str, default='visualizations/active_plots.png', help='Output file path for the plot')

    # --- Command: animate-active ---
    parser_animate = subparsers.add_parser('animate-active', help='Animate active learning diagnostics over trials')
    parser_animate.add_argument(
        "--grid_resolution", type=int, default=10, help="The resolution (N) for the N_x_N grid."
    )
    parser_animate.add_argument('--results_file', type=str, required=True, help='Path to the active_results.csv file')
    parser_animate.add_argument('--interval', type=int, default=500, help='Delay between frames in milliseconds.')
    parser_animate.add_argument('--output_file', type=str, default='visualizations/active_animation.mp4', help='Output file path for the animation (.mp4)')


    # --- Command: plot-comparison ---
    parser_compare = subparsers.add_parser('plot-comparison', help='Compare model error against ground truth')
    parser_compare.add_argument(
        "--grid_resolution", type=int, default=10, help="The resolution (N) for the N_x_N grid."
    )
    parser_compare.add_argument(
        '--gt', type=str, required=True, help='Path to the ground_truth_results.csv file (M samples)'
    )
    parser_compare.add_argument(
        '--model', action='append', nargs=2, metavar=('LABEL', 'FILEPATH'),
        help='Add a model to compare. Can be used multiple times. E.g., --model Active active.csv'
    )
    parser_compare.add_argument(
        '--output_file', type=str, default='visualizations/comparison_plot.png', help='Output file path for the plot'
    )

    args = parser.parse_args()

    # --- Create output directory if it doesn't exist ---
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: '{output_dir}'")

    # --- Execute the chosen command ---
    if args.command == 'plot-points':
        df = _load_data(args.results_file)
        plot_tested_points(df, args.output_file)

    elif args.command == 'plot-active':
        df = _load_data(args.results_file)
        plot_active_learning(df, args.output_file, args.grid_resolution)

    elif args.command == 'animate-active': # Modified command execution
        df = _load_data(args.results_file)
        animate_active_learning(df, args.output_file, args.grid_resolution, args.interval)

    elif args.command == 'plot-comparison':
        gt_df = _load_data(args.gt)

        results_list = []
        if not args.model:
            print("Warning: No --model files provided for comparison.")
        else:
            for label, filepath in args.model:
                df = _load_data(filepath)
                results_list.append((label, df))

        plot_comparison(results_list, gt_df, args.output_file, args.grid_resolution)


if __name__ == "__main__":
    main()