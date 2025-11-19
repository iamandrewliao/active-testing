"""Visualization functions"""

import argparse
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Import necessary BoTorch/GPyTorch components
from botorch.utils.sampling import draw_sobol_samples

from utils import get_grid_points, is_valid_point, fit_surrogate_model, get_acquisition_function, optimize_acq_func

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
        return pd.read_csv(filepath, dtype={'mode': str})
    except Exception as e:
        print(f"Error reading file: {e}")
        exit()


def _get_tensors_from_df(df):
    """Extracts and formats training tensors from a DataFrame."""
    train_X = torch.tensor(df[['x', 'y']].values, **tkwargs)
    train_Y = torch.tensor(df['continuous_outcome'].values, **tkwargs).unsqueeze(-1) # Shape [N, 1]
    return train_X, train_Y


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
    print(f"Saved figure to {output_file}.")


def plot_active_learning(df, output_file, grid_resolution, model_name, acq_func_name):
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
    # Set seed for deterministic 'get_optimal_samples'
    # Use the number of data points as the seed
    if train_X.shape[0] >= 2:
        torch.manual_seed(train_X.shape[0]+1)
        print(f"Using seed {train_X.shape[0]}")
    print(f"  Fitting {model_name} model...")
    model = fit_surrogate_model(train_X, train_Y, BOUNDS, model_name=model_name)

    # 2. Get the acquisition function
    print(f"  Instantiating {acq_func_name} acquisition function.")
    mc_points = None
    if acq_func_name == "qNIPV":
        mc_points = draw_sobol_samples(
            bounds=BOUNDS, n=128, q=1
        ).squeeze(1).to(**tkwargs)
    acq_func = get_acquisition_function(model=model, acq_func_name=acq_func_name, mc_points=mc_points)

    # 3. Create grid and evaluate
    grid_tensor = get_grid_points(grid_resolution, BOUNDS, tkwargs)
    grid_shape = (grid_resolution, grid_resolution)
    
    # --- Create the validity mask ---
    valid_mask_flat = np.array([is_valid_point(p) for p in grid_tensor])
    valid_grid_tensor = grid_tensor[valid_mask_flat]

    # Get model predictions
    mean_values_flat = np.full(grid_tensor.shape[0], np.nan)
    acq_values_flat = np.full(grid_tensor.shape[0], np.nan)

    with torch.no_grad():
        # --- Apply mask to mean values ---
        posterior = model.posterior(grid_tensor)
        # Get the mean, handling 2D [S, N] output for Bayesian models
        mean_tensor = posterior.mean.squeeze(-1) # Shape is [N] or [S, N]
        if mean_tensor.ndim == 2:
            mean_tensor = mean_tensor.mean(dim=0) # Average over S samples to get [N]
        mean_values_flat = mean_tensor.numpy() # shape [N]
        mean_values_flat[~valid_mask_flat] = np.nan # Set invalid to NaN
        mean_values = mean_values_flat.reshape(grid_shape)

        # Calculate acq values only for valid points
        if acq_func is not None and valid_grid_tensor.shape[0] > 0:
            try:
                valid_acq_values = acq_func(valid_grid_tensor.reshape(-1, 1, 2)).numpy()
                acq_values_flat[valid_mask_flat] = valid_acq_values
            except Exception as e:
                 print(f"Warning: {acq_func_name} evaluation failed: {e}")
        acq_values = acq_values_flat.reshape(grid_shape)

    # 4. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    # Calculate extent to center pixels
    x_min, y_min = BOUNDS[0, 0].item(), BOUNDS[0, 1].item()
    x_max, y_max = BOUNDS[1, 0].item(), BOUNDS[1, 1].item()
    
    if grid_resolution > 1:
        x_step = (x_max - x_min) / (grid_resolution - 1)
        y_step = (y_max - y_min) / (grid_resolution - 1)
        half_x_step, half_y_step = x_step / 2.0, y_step / 2.0
        plot_extent = [
            x_min - half_x_step, x_max + half_x_step,
            y_min - half_y_step, y_max + half_y_step
        ]
    else:
        plot_extent = [x_min - 0.5, x_max + 0.5, y_min - 0.5, y_max + 0.5]


    # --- Plot 1: Surrogate Model Mean ---
    ax = axes[0]
    cmap_mean = plt.cm.viridis_r.copy()
    cmap_mean.set_bad(color='gray')
    im = ax.imshow(mean_values, origin='lower', extent=plot_extent, cmap=cmap_mean, vmin=0, vmax=4, aspect='equal')
    fig.colorbar(im, ax=ax, label='Predicted Outcome (Mean)')
    # Scatter plot colored by outcome with white edge
    ax.scatter(
        df['x'], df['y'], c=df['continuous_outcome'], cmap='viridis_r', 
        vmin=0, vmax=4, edgecolors='w', s=50, label='Tested Points', zorder=2
    )
    ax.set_title(f'Surrogate Model ({model_name}) (N={len(df)})')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()
    ax.set_xlim(plot_extent[0], plot_extent[1])
    ax.set_ylim(plot_extent[2], plot_extent[3])


    # --- Plot 2: Acquisition Function ---
    ax = axes[1]
    cmap_acq = plt.cm.magma.copy()
    cmap_acq.set_bad(color='gray') # Show NaN values (invalid points) as gray
    im = ax.imshow(acq_values, origin='lower', extent=plot_extent, cmap=cmap_acq, aspect='equal')
    fig.colorbar(im, ax=ax, label=f'Acquisition Value ({acq_func_name})')
    # Scatter plot colored by outcome with white edge
    ax.scatter(
        df['x'], df['y'], c=df['continuous_outcome'], cmap='viridis_r', 
        vmin=0, vmax=4, edgecolors='w', s=50, label='Tested Points', zorder=2
    )
    ax.set_title('Acquisition Function Landscape')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()
    ax.set_xlim(plot_extent[0], plot_extent[1])
    ax.set_ylim(plot_extent[2], plot_extent[3])

    plt.suptitle('Active Testing Diagnostics', fontsize=16, y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])

    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {output_file}.")


def animate_active_learning(df, output_file, grid_resolution, model_name, acq_func_name, interval=500):
    """
    Generates an MP4 animation showing the evolution of the surrogate model
    and acquisition function during an active testing run, using the 'mode' column.
    """
    print(f"Generating active learning animation (Model: {model_name}, Acqf: {acq_func_name}) -> {output_file}...")
    total_trials = len(df)
    if total_trials < 2:
        print("Error: Need at least 2 data point to show the next acquired points.")
        return

    # --- Setup Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    # Calculate extent to center pixels
    x_min, y_min = BOUNDS[0, 0].item(), BOUNDS[0, 1].item()
    x_max, y_max = BOUNDS[1, 0].item(), BOUNDS[1, 1].item()
    
    if grid_resolution > 1:
        x_step = (x_max - x_min) / (grid_resolution - 1)
        y_step = (y_max - y_min) / (grid_resolution - 1)
        half_x_step, half_y_step = x_step / 2.0, y_step / 2.0
        plot_extent = [
            x_min - half_x_step, x_max + half_x_step,
            y_min - half_y_step, y_max + half_y_step
        ]
    else:
        plot_extent = [x_min - 0.5, x_max + 0.5, y_min - 0.5, y_max + 0.5]
    
    grid_tensor = get_grid_points(grid_resolution, BOUNDS, tkwargs)
    grid_shape = (grid_resolution, grid_resolution)
    normalized_bounds = torch.tensor([[0.0] * 2, [1.0] * 2], **tkwargs)
    
    # Pre-calculate the valid mask
    valid_mask_flat = np.array([is_valid_point(p) for p in grid_tensor])
    valid_grid_tensor = grid_tensor[valid_mask_flat]

    # --- Setup placeholder artists with new styling ---
    # Surrogate model plot
    cmap_mean = plt.cm.viridis_r.copy()
    cmap_mean.set_bad(color='gray')
    im_mean = axes[0].imshow(np.zeros(grid_shape), origin='lower', extent=plot_extent, cmap=cmap_mean, vmin=0, vmax=4, aspect='equal')
    sc_mean = axes[0].scatter([], [], c=[], cmap='viridis_r', vmin=0, vmax=4, edgecolors='w', s=50, label='Tested Points', zorder=2)
    sc_next_mean = axes[0].scatter([], [], c='red', edgecolors='w', s=150, marker='*', label='Next Acquired', zorder=10) 
    cb_mean = fig.colorbar(im_mean, ax=axes[0], label='Predicted Outcome (Mean)')
    axes[0].set_xlabel('X Position')
    axes[0].set_ylabel('Y Position')
    axes[0].legend(loc='upper right')
    axes[0].set_xlim(plot_extent[0], plot_extent[1])
    axes[0].set_ylim(plot_extent[2], plot_extent[3])
    # Acquistion function plot
    cmap_acq = plt.cm.magma.copy()
    cmap_acq.set_bad(color='gray')
    im_acq = axes[1].imshow(np.zeros(grid_shape), origin='lower', extent=plot_extent, cmap=cmap_acq, aspect='equal')
    sc_acq = axes[1].scatter([], [], c=[], cmap='viridis_r', vmin=0, vmax=4, edgecolors='w', s=50, label='Tested Points', zorder=2)
    sc_next_acq = axes[1].scatter([], [], c='red', edgecolors='w', s=150, marker='*', label='Next Acquired', zorder=10) 
    cb_acq = fig.colorbar(im_acq, ax=axes[1], label=f'Acquisition Value ({acq_func_name})')
    axes[1].set_xlabel('X Position')
    axes[1].set_ylabel('Y Position')
    axes[1].legend(loc='upper right')
    axes[1].set_xlim(plot_extent[0], plot_extent[1])
    axes[1].set_ylim(plot_extent[2], plot_extent[3])


    # --- Animation Update Function ---
    def update(frame):
        t = frame + 1 
        print(f"  Processing frame {frame+1}/{total_trials - 1} (Showing model state at t={t}, predicting t={t+1})...")
        # Get data up to and including trial t
        current_df = df.iloc[:t]

        train_X, train_Y = _get_tensors_from_df(current_df)

        # Get the point for the *next* trial (t+1)
        next_point_row = df.iloc[t] # iloc[t] is the (t+1)th row
        next_point_x, next_point_y = next_point_row['x'], next_point_row['y']
        current_mode = next_point_row['mode'] # Mode used to pick *this* next point

        # Clear previous next point highlight
        sc_next_mean.set_offsets(np.empty((0, 2)))
        sc_next_acq.set_offsets(np.empty((0, 2)))
    
        # Initialize plots with NaN
        mean_values_flat = np.full(grid_tensor.shape[0], np.nan)
        acq_values_flat = np.full(grid_tensor.shape[0], np.nan)
        # Set seed based on 't' (the number of points used for fitting) to match eval.py
        if train_X.shape[0] >= 1: # Always set seed before fitting
            torch.manual_seed(t+1)
            print(f"Using seed {t}")
        # Fit model on data 1...t
        model = fit_surrogate_model(train_X, train_Y, BOUNDS, model_name=model_name)
        with torch.no_grad():
            posterior = model.posterior(grid_tensor)
            # Get the mean, handling 2D [S, N] output for Bayesian models
            mean_tensor = posterior.mean.squeeze(-1) # Shape is [N] or [S, N]
            if mean_tensor.ndim == 2:
                mean_tensor = mean_tensor.mean(dim=0) # Average over S samples to get [N]
            mean_values_flat = mean_tensor.numpy() # shape [N]
            mean_values_flat[~valid_mask_flat] = np.nan # Apply mask
        mean_values = mean_values_flat.reshape(grid_shape)
    
        # Calculate acquisition function based on model 1...t
        acq_func = None
        if train_X.shape[0] >= 2:
            mc_points = None
            if acq_func_name == "qNIPV":
                mc_points = draw_sobol_samples(
                    bounds=BOUNDS, n=128, q=1
                ).squeeze(1).to(**tkwargs)
            acq_func = get_acquisition_function(model=model, acq_func_name=acq_func_name, mc_points=mc_points)

        with torch.no_grad():
            if acq_func is not None and valid_grid_tensor.shape[0] > 0:
                try:
                    valid_acq_values = acq_func(valid_grid_tensor.reshape(-1, 1, 2)).numpy()
                    acq_values_flat[valid_mask_flat] = valid_acq_values
                except Exception as e:
                    if t == 2: # Print warning on first attempt
                        print(f"    Warning: Could not compute {acq_func_name} at t={t}: {e}")
        
        acq_values = acq_values_flat.reshape(grid_shape)
    
        # Update Image plots
        im_mean.set_data(mean_values)
        if not np.all(np.isnan(mean_values)):
             im_mean.set_clim(vmin=np.nanmin(mean_values), vmax=np.nanmax(mean_values))
        else:
             im_mean.set_clim(vmin=0, vmax=4)
    
        im_acq.set_data(acq_values)
        if not np.all(np.isnan(acq_values)):
             im_acq.set_clim(vmin=np.nanmin(acq_values), vmax=np.nanmax(acq_values))
    
    
        # Update scatter offsets AND colors (Points 1...t)
        points_so_far = current_df[['x', 'y']].values
        colors_so_far = current_df['continuous_outcome'].values
        sc_mean.set_offsets(points_so_far)
        sc_mean.set_array(colors_so_far)
        sc_acq.set_offsets(points_so_far)
        sc_acq.set_array(colors_so_far)
    
        # Highlight the *next* point to acquire (Point t+1)
        sc_next_mean.set_offsets([[next_point_x, next_point_y]])
        sc_next_acq.set_offsets([[next_point_x, next_point_y]])
    
        # Update Titles
        axes[0].set_title(f'Surrogate Model ({model_name} Mean) (State at N={t})')
        axes[1].set_title(f'Acquisition Function ({acq_func_name}) (Mode: {current_mode})')
    
        fig.suptitle(f'Active Testing: State at Trial {t}, Predicting Trial {t+1} (Out of {total_trials})', fontsize=16, y=0.95)
    
        return [im_mean, sc_mean, sc_next_mean, im_acq, sc_acq, sc_next_acq]
    

    # --- Create and Save Animation ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    ani = animation.FuncAnimation(fig, update, frames=total_trials-1, interval=interval, blit=True)

    try:
        ani.save(output_file, writer='ffmpeg', fps=max(1, 1000//interval), dpi=150)
        print(f"Animation saved successfully to '{output_file}'.")
    except Exception as e:
        print(f"\nError saving animation: {e}")
        print("Ensure ffmpeg is installed and accessible in your system's PATH.")

    plt.close(fig)


def plot_comparison(
    results_list, gt_df, output_file, 
    grid_resolution, model_name, plot_mode='error'
):
    """
    Generates a plot comparing model errors or model means from one or more runs
    (e.g., Active, IID) against a ground truth dataset.
    """
    print(f"Generating comparison plot (Mode: {plot_mode}, Model: {model_name}) -> {output_file}...")

    # --- Create grid and validity mask ---
    grid_tensor = get_grid_points(grid_resolution, BOUNDS, tkwargs)
    grid_shape = (grid_resolution, grid_resolution)
    # This mask is still needed for the *model* plots
    valid_mask_flat = np.array([is_valid_point(p) for p in grid_tensor])

    # 1. Load Ground Truth data (FIXED LOGIC)
    #    We now map each (x, y) point to its grid index.
    X_gt, Y_gt = _get_tensors_from_df(gt_df)
    print(f"  Mapping Ground Truth data (N={len(X_gt)})...")
    if len(X_gt) < 1:
         print("Error: Ground truth data is empty. Cannot generate comparison plot.")
         return

    # Create a nan-filled flat array for the GT map
    mean_gt_flat = np.full(grid_tensor.shape[0], np.nan)
    
    # Create a fast lookup map from "(x_str, y_str)" -> grid_index
    # We use formatted strings to handle floating point comparisons
    grid_map = {
        f"{p[0].item():.6f},{p[1].item():.6f}": i 
        for i, p in enumerate(grid_tensor)
    }

    # Iterate through the GT dataframe and place outcomes in the correct grid slot
    points_mapped = 0
    for _, row in gt_df.iterrows():
        x, y = row['x'], row['y']
        outcome = row['continuous_outcome']
        key = f"{x:.6f},{y:.6f}"
        
        if key in grid_map:
            index = grid_map[key]
            mean_gt_flat[index] = outcome
            points_mapped += 1
        else:
            # This can happen if the GT data was made with a different grid
            print(f"    Warning: GT point (x={x}, y={y}) not found in {grid_resolution}x{grid_resolution} grid.")

    print(f"    Successfully mapped {points_mapped} GT points to the grid.")
    if points_mapped == 0:
        print("Error: No GT points matched the grid. Check --grid_resolution.")
        return

    # Reshape the flat array to the 2D grid
    mean_gt = mean_gt_flat.reshape(grid_shape)


    # 2. Fit each model and calculate error/stats (This block is unchanged)
    model_data_list = []
    for label, df in results_list:
        X_model, Y_model = _get_tensors_from_df(df)
        if len(X_model) < 1:
            print(f"  Skipping {label} model (N=0).")
            continue
        print(f"  Fitting {label} {model_name} model (N={len(X_model)})...")
        model = fit_surrogate_model(X_model, Y_model, BOUNDS, model_name=model_name)

        with torch.no_grad():
            # --- Apply mask to model mean ---
            mean_tensor_model = model.posterior(grid_tensor).mean.squeeze(-1)
            if mean_tensor_model.ndim == 2:
                mean_tensor_model = mean_tensor_model.mean(dim=0)
                
            mean_model_flat = mean_tensor_model.numpy()
            # We use the valid_mask_flat here to mask out the model's predictions
            # in the same invalid regions as the GT.
            mean_model_flat[~valid_mask_flat] = np.nan
            mean_model = mean_model_flat.reshape(grid_shape)

        # Calculate error metrics regardless of plot_mode
        # This works because both mean_model and mean_gt have NaNs
        # in the same invalid locations.
        error_map = np.abs(mean_model - mean_gt)
        mae = np.nanmean(error_map) # MAE only over valid areas
        print(f"    Valid Area MAE ({label}): {mae:.4f}")
        
        # Store the data
        stats_dict = {
            'label': label,
            'n': len(X_model),
            'mean_map': mean_model,
            'error_map': error_map,
            'mae': mae
        }
        
        model_data_list.append(stats_dict)

    # 3. Plotting (This block is unchanged)
    num_models = len(model_data_list)
    num_plots = num_models + 1 # +1 for GT

    fig, axes = plt.subplots(1, num_plots, figsize=(9 * num_plots, 9))
    if num_plots == 0:
        print("Error: No models provided or fit successfully for comparison.")
        return
    elif num_plots == 1:
        axes = [axes] # Make it iterable
    else:
        axes = axes.flatten()

    # Calculate extent (same as before)
    x_min, y_min = BOUNDS[0, 0].item(), BOUNDS[0, 1].item()
    x_max, y_max = BOUNDS[1, 0].item(), BOUNDS[1, 1].item()
    
    if grid_resolution > 1:
        x_step = (x_max - x_min) / (grid_resolution - 1)
        y_step = (y_max - y_min) / (grid_resolution - 1)
        half_x_step, half_y_step = x_step / 2.0, y_step / 2.0
        plot_extent = [
            x_min - half_x_step, x_max + half_x_step,
            y_min - half_y_step, y_max + half_y_step
        ]
    else:
        plot_extent = [x_min - 0.5, x_max + 0.5, y_min - 0.5, y_max + 0.5]

    # Determine a common error scale (only if needed)
    vmax_error = 1.0
    if plot_mode == 'error':
        if model_data_list:
            valid_maxes = [
                np.nanmax(stats['error_map']) for stats in model_data_list 
                if 'error_map' in stats and not np.all(np.isnan(stats['error_map']))
            ]
            if valid_maxes:
                vmax_error = max(valid_maxes)
            if vmax_error == 0 or not valid_maxes: 
                vmax_error = 1.0

    # Define the colormaps
    cmap_gt = plt.cm.viridis_r.copy()
    cmap_gt.set_bad(color='gray')
    cmap_err = plt.cm.hot.copy()
    cmap_err.set_bad(color='gray')

    # --- Plot 1...N: Model Plots (Error or Mean) ---
    for i, stats in enumerate(model_data_list):
        ax = axes[i]
        
        if plot_mode == 'error':
            im = ax.imshow(
                stats['error_map'], origin='lower', extent=plot_extent, 
                cmap=cmap_err, vmin=0, vmax=vmax_error, aspect='equal'
            )
            fig.colorbar(im, ax=ax, label='Absolute Error')
            ax.set_title(
                f"{stats['label']} Model Error (N={stats['n']})\n"
                f"Valid Area MAE = {stats['mae']:.4f}"
            )
        
        elif plot_mode == 'mean':
            im = ax.imshow(
                stats['mean_map'], origin='lower', extent=plot_extent, 
                cmap=cmap_gt, vmin=0, vmax=4, aspect='equal' # Use GT cmap and scale
            )
            fig.colorbar(im, ax=ax, label='Predicted Outcome (Mean)')
            ax.set_title(
                f"{stats['label']} Model Mean (N={stats['n']})\n"
                f"Valid Area MAE = {stats['mae']:.4f}"
            )
            
        # Common axis setup
        ax.set_xlabel('X Position')
        if i == 0:
            ax.set_ylabel('Y Position')
        ax.set_xlim(plot_extent[0], plot_extent[1])
        ax.set_ylim(plot_extent[2], plot_extent[3])


    # --- Plot N+1: Ground Truth Model ---
    ax_gt = axes[num_models]
    im_gt = ax_gt.imshow(
        mean_gt, origin='lower', extent=plot_extent, 
        cmap=cmap_gt, vmin=0, vmax=4, aspect='equal'
    )
    fig.colorbar(im_gt, ax=ax_gt, label='Actual Outcome (Mean)')
    ax_gt.set_title(f'Ground Truth Data (N={points_mapped})') # Use points_mapped
    ax_gt.set_xlabel('X Position')
    if num_models == 0:
        ax_gt.set_ylabel('Y Position')
    ax_gt.set_xlim(plot_extent[0], plot_extent[1])
    ax_gt.set_ylim(plot_extent[2], plot_extent[3])

    plt.suptitle(f'Model Comparison vs. Ground Truth (N={points_mapped})', fontsize=16, y=0.96)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {output_file}.")


def main():
    parser = argparse.ArgumentParser(description="Visualization script for evaluation results.")
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', required=True, help='Visualization command to run')

    # --- Command: plot-points ---
    parser_points = subparsers.add_parser('plot-points', help='Plot tested points colored by outcome')
    parser_points.add_argument('--results_file', type=str, required=True, help='Path to the evaluation_results.csv file')
    parser_points.add_argument('--output_file', type=str, default='visualizations/robo_eval/tested_points.png', help='Output file path for the plot')

    # --- Command: plot-active ---
    parser_active = subparsers.add_parser('plot-active', help='Plot active learning diagnostics (model mean, acqf)')
    parser_active.add_argument(
        "--grid_resolution", type=int, default=10, help="The resolution (N) for the N_x_N grid."
    )
    parser_active.add_argument('--results_file', type=str, required=True, help='Path to the active_results.csv file')
    parser_active.add_argument('--output_file', type=str, default='visualizations/robo_eval/active_plots.png', help='Output file path for the plot')
    parser_active.add_argument(
        '--model_name', type=str, default='SingleTaskGP', 
        help='Name of surrogate model (e.g. SingleTaskGP, SaasFullyBayesianSingleTaskGP)'
    )
    parser_active.add_argument(
        '--acq_func_name', type=str, default='PSD', help='Name of acq func (e.g. PSD, qNIPV, qBALD)'
    )

    # --- Command: animate-active ---
    parser_animate = subparsers.add_parser('animate-active', help='Animate active learning diagnostics over trials')
    parser_animate.add_argument(
        "--grid_resolution", type=int, default=10, help="The resolution (N) for the N_x_N grid."
    )
    parser_animate.add_argument('--results_file', type=str, required=True, help='Path to the active_results.csv file')
    parser_animate.add_argument('--interval', type=int, default=500, help='Delay between frames in milliseconds.')
    parser_animate.add_argument('--output_file', type=str, default='visualizations/robo_eval/active_animation.mp4', help='Output file path for the animation (.mp4)')
    parser_animate.add_argument(
        '--model_name', type=str, default='SingleTaskGP', 
        help='Name of surrogate model (e.g. SingleTaskGP, SaasFullyBayesianSingleTaskGP)'
    )
    parser_animate.add_argument(
        '--acq_func_name', type=str, default='PSD', help='Name of acq func (e.g. PSD, qNIPV, qBALD)'
    )

    # --- Command: plot-comparison ---
    parser_compare = subparsers.add_parser('plot-comparison', help='Compare model error against ground truth')
    parser_compare.add_argument(
        "--grid_resolution", type=int, default=10, help="The resolution (N) for the N_x_N grid."
    )
    parser_compare.add_argument(
        '--gt_results_file', type=str, required=True, help='Path to the ground_truth_results.csv file (M samples)'
    )
    parser_compare.add_argument(
        '--add_results_file', action='append', nargs=2, metavar=('LABEL', 'FILEPATH'),
        help='Add a results file to compare. E.g., --add_results_file Active active.csv'
    )
    parser_compare.add_argument(
        '--output_file', type=str, default='visualizations/robo_eval/comparison_plot.png', help='Output file path for the plot'
    )
    parser_compare.add_argument(
        '--model_name', type=str, default='SingleTaskGP', 
        help='Name of surrogate model (e.g. SingleTaskGP, SaasFullyBayesianSingleTaskGP)'
    )
    parser_compare.add_argument(
        '--plot_mode', type=str, choices=['error', 'mean'], default='mean',
        help="Type of plot to generate: 'error' (vs GT) or 'mean' (model output)"
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
        plot_active_learning(df, args.output_file, args.grid_resolution, args.model_name, args.acq_func_name)

    elif args.command == 'animate-active':
        df = _load_data(args.results_file)
        animate_active_learning(df, args.output_file, args.grid_resolution, args.model_name, args.acq_func_name, args.interval)

    elif args.command == 'plot-comparison':
        gt_df = _load_data(args.gt_results_file)
        results_list = []
        if not args.add_results_file:
            print("Warning: No --add_results_file files provided for comparison.")
        else:
            for label, filepath in args.add_results_file:
                df = _load_data(filepath)
                results_list.append((label, df))
        plot_comparison(results_list, gt_df, args.output_file, args.grid_resolution, args.model_name, args.plot_mode)


if __name__ == "__main__":
    main()