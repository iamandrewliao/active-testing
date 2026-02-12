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

from utils import get_design_points_test, fit_surrogate_model, get_acquisition_function, optimize_acq_func, calculate_rmse, calculate_log_likelihood
from factors_config import (
    BOUNDS,
    FACTOR_COLUMNS,
    get_outcome_range,
    get_success_outcome,
    OBJECT_POS_X_VALUES,
    OBJECT_POS_Y_VALUES,
    TABLE_HEIGHT_VALUES,
    VIEWPOINT_VALUES,
    get_viewpoint_name,
    get_viewpoint_params,
    get_viewpoint_index_from_params,
    is_valid_point,
    VIEWPOINT_REPRESENTATION,
)

# Set up torch device and data type
tkwargs = {"dtype": torch.double, "device": "cpu"}

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
    # Use FACTOR_COLUMNS from config to handle 4D factors
    train_X = torch.tensor(df[FACTOR_COLUMNS].values, **tkwargs)
    train_Y = torch.tensor(df['continuous_outcome'].values, **tkwargs).unsqueeze(-1) # Shape [N, 1]
    return train_X, train_Y


def plot_tested_points_xy(df, output_file, task_name=None):
    """
    Plots a simple scatter plot of tested points, colored by their outcome.
    Only shows factors x and y (2D).
    """
    print(f"Generating tested points plot -> {output_file}...")
    plt.figure(figsize=(10, 8))

    # Get outcome range from task config (with fallback)
    min_outcome, max_outcome, _ = get_outcome_range(task_name)

    # Scatter plot colored by continuous outcome (2D projection: x vs y)
    sc = plt.scatter(
        df['x'],
        df['y'],
        c=df['continuous_outcome'],
        cmap='viridis_r',  # Use _r for "reversed" (low=bad, high=good)
        vmin=min_outcome,
        vmax=max_outcome,
        edgecolors='k',
        alpha=0.8
    )

    plt.colorbar(sc, label=f'Continuous Outcome ({min_outcome}-{max_outcome})')
    plt.title(f"Tested Points and Outcomes (N={len(df)})")
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.xlim(BOUNDS[0, 0].item(), BOUNDS[1, 0].item()) # Use .item() for scalar bounds
    plt.ylim(BOUNDS[0, 1].item(), BOUNDS[1, 1].item())
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {output_file}.")


def plot_active_learning_xy(df, output_file, grid_resolution, model_name, acq_func_name, table_height, camera_viewpoint, results_file=None, task_name=None):
    """
    Plots the surrogate model mean and acquisition function landscape
    for an active testing run.
    Only shows factors x and y (2D).
    
    Args:
        df: DataFrame with evaluation results
        output_file: Output path for the plot
        grid_resolution: Resolution for the grid (for 2D projection)
        model_name: Model name (for fallback if model not saved)
        acq_func_name: Acquisition function name
        table_height: table height value to use for visualization
        camera_viewpoint: viewpoint index value (e.g. 0, 1, 2) to use for visualization
        results_file: Path to results CSV (used to find saved model)
        task_name: Task name for outcome range
    """
    print(f"Generating active learning plots -> {output_file}...")
    import pickle

    # Try to load saved model from eval.py
    model = None
    if results_file:
        # Check if results file is in new structure: results/{eval_id}/results.csv
        # If so, look for models in results/{eval_id}/models/
        results_dir = os.path.dirname(results_file)
        results_basename = os.path.basename(results_file)
        
        # Try new structure first: results/{eval_id}/models/final_model.pkl
        if os.path.basename(results_dir) and os.path.exists(os.path.join(results_dir, 'models')):
            model_path = os.path.join(results_dir, 'models', 'final_model.pkl')
        else:
            # Fall back to old structure: same directory as CSV
            model_path = results_file.replace('.csv', '_model.pkl')
        
        if os.path.exists(model_path):
            try:
                print(f"  Loading saved model from '{model_path}'...")
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    model = model_data['model']
                    # Verify model name matches
                    if model_data.get('model_name') != model_name:
                        print(f"  Warning: Saved model name ({model_data.get('model_name')}) doesn't match requested ({model_name}). Using saved model.")
                    print(f"  Successfully loaded model (trained on {model_data['train_X'].shape[0]} points).")
            except Exception as e:
                print(f"  Warning: Could not load saved model: {e}. Will retrain.")
    
    # If model not loaded, retrain (fallback)
    if model is None:
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

    # 3. Create 2D grid for visualization (x, y) and expand to the full factor space
    # For visualization, we fix table_height and viewpoint to default values
    x_grid = torch.linspace(BOUNDS[0, 0].item(), BOUNDS[1, 0].item(), grid_resolution, **tkwargs)
    y_grid = torch.linspace(BOUNDS[0, 1].item(), BOUNDS[1, 1].item(), grid_resolution, **tkwargs)
    xx, yy = torch.meshgrid(x_grid, y_grid, indexing='ij')
    grid_2d = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    
    # Expand to full factor space: use given values for table_height and viewpoint
    if VIEWPOINT_REPRESENTATION == "index":
        # Factors: [x, y, table_height, viewpoint_index]
        grid_tensor = torch.cat(
            [
                grid_2d,
                torch.full((grid_2d.shape[0], 1), table_height, **tkwargs),
                torch.full((grid_2d.shape[0], 1), camera_viewpoint, **tkwargs),
            ],
            dim=1,
        )
    else:
        # Factors: [x, y, table_height, camera_azimuth, camera_elevation, camera_distance]
        # Use camera params for viewpoint 0 ('back') for 2D visualization
        vp = get_viewpoint_params(camera_viewpoint)
        cam_az = float(vp["azimuth"])
        cam_el = float(vp["elevation"])
        cam_dist = float(vp["distance"])
        grid_tensor = torch.cat(
            [
                grid_2d,
                torch.full((grid_2d.shape[0], 1), table_height, **tkwargs),
                torch.full((grid_2d.shape[0], 1), cam_az, **tkwargs),
                torch.full((grid_2d.shape[0], 1), cam_el, **tkwargs),
                torch.full((grid_2d.shape[0], 1), cam_dist, **tkwargs),
            ],
            dim=1,
        )
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
                # Reshape for acquisition function: (N, 1, D) where D is the factor dimensionality
                valid_acq_values = acq_func(valid_grid_tensor.unsqueeze(1)).numpy()
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
    # Get outcome range for colorbar
    min_outcome, max_outcome, _ = get_outcome_range(task_name)
    
    cmap_mean = plt.cm.viridis_r.copy()
    cmap_mean.set_bad(color='gray')
    im = ax.imshow(mean_values, origin='lower', extent=plot_extent, cmap=cmap_mean, vmin=min_outcome, vmax=max_outcome, aspect='equal')
    fig.colorbar(im, ax=ax, label='Predicted Outcome (Mean)')
    # Scatter plot colored by outcome with white edge
    ax.scatter(
        df['x'], df['y'], c=df['continuous_outcome'], cmap='viridis_r', 
        vmin=min_outcome, vmax=max_outcome, edgecolors='w', s=50, label='Tested Points', zorder=2
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
        vmin=min_outcome, vmax=max_outcome, edgecolors='w', s=50, label='Tested Points', zorder=2
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


def animate_active_learning_xy(df, output_file, grid_resolution, model_name, acq_func_name, table_height, camera_viewpoint, interval=500):
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
    
    # Create 2D grid for visualization and expand to full factor space
    x_grid = torch.linspace(BOUNDS[0, 0].item(), BOUNDS[1, 0].item(), grid_resolution, **tkwargs)
    y_grid = torch.linspace(BOUNDS[0, 1].item(), BOUNDS[1, 1].item(), grid_resolution, **tkwargs)
    xx, yy = torch.meshgrid(x_grid, y_grid, indexing='ij')
    grid_2d = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    
    # Expand to full factor space: use given values for table_height and viewpoint
    if VIEWPOINT_REPRESENTATION == "index":
        grid_tensor = torch.cat(
            [
                grid_2d,
                torch.full((grid_2d.shape[0], 1), table_height, **tkwargs),
                torch.full((grid_2d.shape[0], 1), camera_viewpoint, **tkwargs),
            ],
            dim=1,
        )
    else:
        vp = get_viewpoint_params(0)
        cam_az = float(vp["azimuth"])
        cam_el = float(vp["elevation"])
        cam_dist = float(vp["distance"])
        grid_tensor = torch.cat(
            [
                grid_2d,
                torch.full((grid_2d.shape[0], 1), table_height, **tkwargs),
                torch.full((grid_2d.shape[0], 1), cam_az, **tkwargs),
                torch.full((grid_2d.shape[0], 1), cam_el, **tkwargs),
                torch.full((grid_2d.shape[0], 1), cam_dist, **tkwargs),
            ],
            dim=1,
        )
    grid_shape = (grid_resolution, grid_resolution)
    
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
                    valid_acq_values = acq_func(valid_grid_tensor.unsqueeze(1)).numpy()
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


def plot_comparison_xy(
    results_list, gt_df, output_file, 
    grid_resolution, model_name, table_height, camera_viewpoint, plot_mode
):
    """
    Generates a plot comparing model errors or model means from one or more runs
    (e.g., Active, IID) against a ground truth dataset.
    Only shows factors x and y (2D projection).
    
    Args:
        results_list: List of (label, df) tuples for model results to compare
        gt_df: DataFrame with ground truth results
        output_file: Output path for the plot
        grid_resolution: Resolution for the 2D grid (x, y)
        model_name: Model name to use
        table_height: Table height value to use for visualization
        camera_viewpoint: Viewpoint index to use for visualization
        plot_mode: 'error' or 'mean'
    """
    print(f"Generating comparison plot (Mode: {plot_mode}, Model: {model_name}) -> {output_file}...")

    # 1. Create 2D grid for visualization (x, y) and expand to full factor space
    x_grid = torch.linspace(BOUNDS[0, 0].item(), BOUNDS[1, 0].item(), grid_resolution, **tkwargs)
    y_grid = torch.linspace(BOUNDS[0, 1].item(), BOUNDS[1, 1].item(), grid_resolution, **tkwargs)
    xx, yy = torch.meshgrid(x_grid, y_grid, indexing='ij')
    grid_2d = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    
    # Expand to full factor space: use given values for table_height and viewpoint
    if VIEWPOINT_REPRESENTATION == "index":
        # Factors: [x, y, table_height, viewpoint_index]
        grid_tensor = torch.cat(
            [
                grid_2d,
                torch.full((grid_2d.shape[0], 1), table_height, **tkwargs),
                torch.full((grid_2d.shape[0], 1), float(camera_viewpoint), **tkwargs),
            ],
            dim=1,
        )
    else:
        # Factors: [x, y, table_height, camera_azimuth, camera_elevation, camera_distance]
        vp = get_viewpoint_params(camera_viewpoint)
        cam_az = float(vp["azimuth"])
        cam_el = float(vp["elevation"])
        cam_dist = float(vp["distance"])
        grid_tensor = torch.cat(
            [
                grid_2d,
                torch.full((grid_2d.shape[0], 1), table_height, **tkwargs),
                torch.full((grid_2d.shape[0], 1), cam_az, **tkwargs),
                torch.full((grid_2d.shape[0], 1), cam_el, **tkwargs),
                torch.full((grid_2d.shape[0], 1), cam_dist, **tkwargs),
            ],
            dim=1,
        )
    grid_shape = (grid_resolution, grid_resolution)
    
    # Create validity mask
    valid_mask_flat = np.array([is_valid_point(p) for p in grid_tensor])

    # 2. Load Ground Truth data and aggregate by (x, y)
    #    Since GT may have multiple table_height/viewpoint values per (x,y), we aggregate
    print(f"  Mapping Ground Truth data (N={len(gt_df)})...")
    if len(gt_df) < 1:
         print("Error: Ground truth data is empty. Cannot generate comparison plot.")
         return

    # Create a lookup: (x, y) -> list of outcomes (for aggregation)
    gt_lookup = {}
    for _, row in gt_df.iterrows():
        x, y = row['x'], row['y']
        outcome = row['continuous_outcome']
        # Round to match grid precision
        x_rounded = round(float(x), 6)
        y_rounded = round(float(y), 6)
        key = (x_rounded, y_rounded)
        if key not in gt_lookup:
            gt_lookup[key] = []
        gt_lookup[key].append(outcome)
    
    # Create a nan-filled flat array for the GT map
    mean_gt_flat = np.full(grid_tensor.shape[0], np.nan)
    
    # Map GT outcomes to grid (aggregate multiple outcomes per (x,y) by taking mean)
    points_mapped = 0
    for i, p in enumerate(grid_tensor):
        x_val = round(p[0].item(), 6)
        y_val = round(p[1].item(), 6)
        key = (x_val, y_val)
        if key in gt_lookup:
            # Aggregate multiple outcomes (if any) by taking mean
            mean_gt_flat[i] = np.mean(gt_lookup[key])
            points_mapped += 1
    
    print(f"    Successfully mapped {points_mapped} GT grid points (from {len(gt_df)} GT data points).")
    if points_mapped == 0:
        print("Error: No GT points matched the grid. Check --grid_resolution.")
        return

    # Reshape the flat array to the 2D grid
    mean_gt = mean_gt_flat.reshape(grid_shape)


    # 3. Fit each model and calculate error/stats
    model_data_list = []
    for label, df in results_list:
        X_model, Y_model = _get_tensors_from_df(df)
        if len(X_model) < 1:
            print(f"  Skipping {label} model (N=0).")
            continue
        print(f"  Fitting {label} {model_name} model (N={len(X_model)})...")
        model = fit_surrogate_model(X_model, Y_model, BOUNDS, model_name=model_name)

        with torch.no_grad():
            # Move grid_tensor to model's device
            device = next(model.parameters()).device
            grid_tensor_device = grid_tensor.to(device)
            
            # --- Apply mask to model mean ---
            posterior = model.posterior(grid_tensor_device)
            mean_tensor_model = posterior.mean.squeeze(-1)
            if mean_tensor_model.ndim == 2:
                mean_tensor_model = mean_tensor_model.mean(dim=0)
                
            mean_model_flat = mean_tensor_model.cpu().numpy()
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


def plot_metrics_vs_trials(active_dfs=None, iid_dfs=None, gt_df=None, output_file=None, model_name=None, task_name=None, 
                           active_results_files=None, iid_results_files=None):
    """
    Plots log-likelihood (LL) and RMSE on ground truth test set vs. trials.
    Compares active testing, IID testing, and ground truth (line at RMSE=0).
    
    Supports multiple runs with variance shading. Uses saved models from eval.py at each trial instead of retraining.
    
    Args:
        active_dfs: List of DataFrames with active testing results (or single DataFrame for backward compatibility)
        iid_dfs: List of DataFrames with IID testing results (or single DataFrame for backward compatibility)
        gt_df: DataFrame with ground truth test set
        output_file: Output path for the plot
        model_name: Surrogate model name to use (if trained models aren't found)
        task_name: Task name for configuration
        active_results_files: List of paths to active results CSV files (used to find saved models)
        iid_results_files: List of paths to IID results CSV files (used to find saved models)
    """
    print(f"Generating metrics vs trials plot -> {output_file}...")
    import pickle
    
    # Prepare ground truth test set
    X_gt, Y_gt = _get_tensors_from_df(gt_df)
    if len(X_gt) < 1:
        print("Error: Ground truth test set is empty.")
        return

    # Normalize inputs to lists for consistent processing
    if active_dfs is None:
        active_dfs = []
    elif not isinstance(active_dfs, list):
        active_dfs = [active_dfs]
    
    if iid_dfs is None:
        iid_dfs = []
    elif not isinstance(iid_dfs, list):
        iid_dfs = [iid_dfs]
    
    if active_results_files is None:
        active_results_files = []
    elif not isinstance(active_results_files, list):
        active_results_files = [active_results_files]
    
    if iid_results_files is None:
        iid_results_files = []
    elif not isinstance(iid_results_files, list):
        iid_results_files = [iid_results_files]
    
    # Ensure we have matching numbers of DataFrames and file paths
    while len(active_results_files) < len(active_dfs):
        active_results_files.append(None)
    while len(iid_results_files) < len(iid_dfs):
        iid_results_files.append(None)
    
    def process_runs(dfs, results_files, run_type):
        """Process multiple runs and return metrics arrays."""
        all_rmse_runs = []
        all_ll_runs = []
        all_trials_runs = []
        
        print(f"  Processing {run_type} testing results ({len(dfs)} runs)...")
        
        for run_idx, (df, results_file) in enumerate(zip(dfs, results_files)):
            rmse_values = []
            ll_values = []
            trials_values = []
            
            results_file_base = results_file.replace('.csv', '') if results_file else None
            
            for trial_num in range(1, len(df) + 1):
                # Try to load saved model from eval.py
                model = None
                if results_file_base:
                    # Check if results file is in new structure: results/{eval_id}/results.csv
                    # If so, look for models in results/{eval_id}/models/
                    results_file_full = results_files[run_idx] if run_idx < len(results_files) else None
                    if results_file_full:
                        results_dir = os.path.dirname(results_file_full)
                        # Try new structure first: results/{eval_id}/models/trial_{trial_num}_model.pkl
                        if os.path.basename(results_dir) and os.path.exists(os.path.join(results_dir, 'models')):
                            model_path = os.path.join(results_dir, 'models', f'trial_{trial_num}_model.pkl')
                        else:
                            # Fall back to old structure
                            model_path = f"{results_file_base}_trial_{trial_num}_model.pkl"
                    else:
                        # Fall back to old structure
                        model_path = f"{results_file_base}_trial_{trial_num}_model.pkl"
                    
                    if os.path.exists(model_path):
                        try:
                            with open(model_path, 'rb') as f:
                                model_data = pickle.load(f)
                                model = model_data['model']
                        except Exception as e:
                            print(f"    Warning: Could not load model for run {run_idx+1}, trial {trial_num}: {e}")
                
                # If model not loaded, fall back to retraining
                if model is None:
                    current_df = df.iloc[:trial_num]
                    if len(current_df) < 1:
                        continue
                    
                    train_X, train_Y = _get_tensors_from_df(current_df)
                    if len(train_X) < 2:  # Need at least 2 points to fit a model
                        continue
                    
                    # Set seed to match eval.py behavior
                    torch.manual_seed(trial_num)
                    
                    # Fit model
                    model = fit_surrogate_model(train_X, train_Y, BOUNDS, model_name=model_name)
                
                # Calculate metrics on GT test set
                rmse = calculate_rmse(model, X_gt, Y_gt)
                ll = calculate_log_likelihood(model, X_gt, Y_gt)  # Log-likelihood
                
                rmse_values.append(rmse)
                ll_values.append(ll)
                trials_values.append(trial_num)
            
            all_rmse_runs.append(rmse_values)
            all_ll_runs.append(ll_values)
            all_trials_runs.append(trials_values)
        
        return all_rmse_runs, all_ll_runs, all_trials_runs
    
    # Process active and IID runs
    active_rmse_runs, active_ll_runs, active_trials_runs = process_runs(active_dfs, active_results_files, "active")
    iid_rmse_runs, iid_ll_runs, iid_trials_runs = process_runs(iid_dfs, iid_results_files, "IID")
    
    def compute_mean_std(runs_data, trials_runs):
        """Compute mean and std across runs, handling different trial lengths."""
        # Find max trial number across all runs
        max_trial = max([max(trials) for trials in trials_runs] + [0]) if trials_runs else 0
        
        if max_trial == 0:
            return [], [], []
        
        # Initialize arrays for mean and std
        mean_values = []
        std_values = []
        trial_numbers = []
        
        for trial_num in range(1, max_trial + 1):
            # Collect values for this trial across all runs
            trial_values = []
            for run_idx, trials in enumerate(trials_runs):
                if trial_num in trials:
                    trial_idx = trials.index(trial_num)
                    trial_values.append(runs_data[run_idx][trial_idx])
            
            if len(trial_values) > 0:
                mean_values.append(np.mean(trial_values))
                std_values.append(np.std(trial_values))
                trial_numbers.append(trial_num)
        
        return mean_values, std_values, trial_numbers
    
    # Compute statistics across runs
    active_rmse_mean, active_rmse_std, active_trials = compute_mean_std(active_rmse_runs, active_trials_runs)
    active_ll_mean, active_ll_std, _ = compute_mean_std(active_ll_runs, active_trials_runs)
    
    iid_rmse_mean, iid_rmse_std, iid_trials = compute_mean_std(iid_rmse_runs, iid_trials_runs)
    iid_ll_mean, iid_ll_std, _ = compute_mean_std(iid_ll_runs, iid_trials_runs)
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: RMSE vs Trials
    ax = axes[0]
    
    # Plot active testing with variance shading
    if len(active_rmse_mean) > 0:
        ax.plot(active_trials, active_rmse_mean, 'b-o', label='Active Testing', markersize=4, linewidth=1.5)
        if len(active_dfs) > 1:  # Only show shading if multiple runs
            ax.fill_between(active_trials, 
                          np.array(active_rmse_mean) - np.array(active_rmse_std),
                          np.array(active_rmse_mean) + np.array(active_rmse_std),
                          alpha=0.2, color='blue')
    
    # Plot IID testing with variance shading
    if len(iid_rmse_mean) > 0:
        ax.plot(iid_trials, iid_rmse_mean, 'r-s', label='IID Testing', markersize=4, linewidth=1.5)
        if len(iid_dfs) > 1:  # Only show shading if multiple runs
            ax.fill_between(iid_trials,
                          np.array(iid_rmse_mean) - np.array(iid_rmse_std),
                          np.array(iid_rmse_mean) + np.array(iid_rmse_std),
                          alpha=0.2, color='red')
    
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, label='Ground Truth (RMSE=0)')
    ax.set_xlabel('Number of Trials', fontsize=12)
    ax.set_ylabel('RMSE on Ground Truth Test Set', fontsize=12)
    ax.set_title('RMSE vs. Trials', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Log-Likelihood vs Trials
    ax = axes[1]
    
    # Plot active testing with variance shading
    if len(active_ll_mean) > 0:
        ax.plot(active_trials, active_ll_mean, 'b-o', label='Active Testing', markersize=4, linewidth=1.5)
        if len(active_dfs) > 1:  # Only show shading if multiple runs
            ax.fill_between(active_trials,
                          np.array(active_ll_mean) - np.array(active_ll_std),
                          np.array(active_ll_mean) + np.array(active_ll_std),
                          alpha=0.2, color='blue')
    
    # Plot IID testing with variance shading
    if len(iid_ll_mean) > 0:
        ax.plot(iid_trials, iid_ll_mean, 'r-s', label='IID Testing', markersize=4, linewidth=1.5)
        if len(iid_dfs) > 1:  # Only show shading if multiple runs
            ax.fill_between(iid_trials,
                          np.array(iid_ll_mean) - np.array(iid_ll_std),
                          np.array(iid_ll_mean) + np.array(iid_ll_std),
                          alpha=0.2, color='red')
    
    ax.set_xlabel('Number of Trials', fontsize=12)
    ax.set_ylabel('Log-Likelihood on Ground Truth Test Set', fontsize=12)
    ax.set_title('Log-Likelihood vs. Trials', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    runs_info = ""
    if len(active_dfs) > 1 or len(iid_dfs) > 1:
        runs_info = f" ({len(active_dfs)} active, {len(iid_dfs)} IID runs)"
    
    plt.suptitle(f'Model Performance Comparison (Model: {model_name}){runs_info}', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved figure to {output_file}.")


def create_rmse_summary_table(eval_df, gt_df, output_file, model_name, task_name=None, eval_results_file=None):
    """
    Creates a summary table of RMSE (prediction error) across all factor value combinations.
    
    Table structure:
    - Rows: Camera viewpoint (0, 1, 2)
    - Columns: Nested structure - Table heights (1, 2, 3), then for each table height,
               position x values (0-1), then for each position x, position y values (0-1)
    
    Uses the final saved model from eval.py instead of retraining.
    
    Args:
        eval_df: DataFrame with evaluation results (used to train the model)
        gt_df: DataFrame with ground truth results (used to calculate errors)
        output_file: Output path for the table (CSV file)
        model_name: Surrogate model name to use
        task_name: Task name for configuration
        eval_results_file: Path to evaluation results CSV (used to find saved final model)
    """
    print(f"Generating RMSE summary table -> {output_file}...")
    import pickle
    
    # Prepare ground truth data - create a lookup dictionary
    # Handle both "index" and "params" viewpoint representations
    gt_lookup = {}
    for _, row in gt_df.iterrows():
        x_val = round(row['x'], 1)
        y_val = round(row['y'], 1)
        th_val = round(row['table_height'])
        
        if VIEWPOINT_REPRESENTATION == "index":
            # Use viewpoint index directly
            vp_val = round(row['viewpoint'])
            key = (x_val, y_val, th_val, vp_val)
        else:
            # Convert camera params to viewpoint index for lookup
            cam_az = row['camera_azimuth']
            cam_el = row['camera_elevation']
            cam_dist = row['camera_distance']
            vp_idx = get_viewpoint_index_from_params(cam_az, cam_el, cam_dist)
            if vp_idx is None:
                continue  # Skip if viewpoint params don't match any known viewpoint
            key = (x_val, y_val, th_val, vp_idx)
        
        gt_lookup[key] = row['continuous_outcome']
    
    # Get all factor values
    x_vals = OBJECT_POS_X_VALUES.cpu().numpy()
    y_vals = OBJECT_POS_Y_VALUES.cpu().numpy()
    table_heights = TABLE_HEIGHT_VALUES.cpu().numpy()
    viewpoints = VIEWPOINT_VALUES.cpu().numpy()
    
    # Try to load final saved model from eval.py
    model = None
    if eval_results_file:
        # Check if results file is in new structure: results/{eval_id}/results.csv
        # If so, look for models in results/{eval_id}/models/
        results_dir = os.path.dirname(eval_results_file)
        
        # Try new structure first: results/{eval_id}/models/final_model.pkl
        if os.path.basename(results_dir) and os.path.exists(os.path.join(results_dir, 'models')):
            final_model_path = os.path.join(results_dir, 'models', 'final_model.pkl')
        else:
            # Fall back to old structure: same directory as CSV
            final_model_path = eval_results_file.replace('.csv', '_model.pkl')
        
        if os.path.exists(final_model_path):
            try:
                print(f"  Loading final saved model from '{final_model_path}'...")
                with open(final_model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    model = model_data['model']
                    print(f"  Successfully loaded final model (trained on {model_data['train_X'].shape[0]} points).")
            except Exception as e:
                print(f"  Warning: Could not load final model: {e}")
    
    # If model not loaded, fall back to retraining
    if model is None:
        if len(eval_df) < 2:
            print("Error: Need at least 2 evaluation points to fit a model.")
            return None
        
        print(f"  Fitting model on {len(eval_df)} evaluation points (model not found)...")
        train_X, train_Y = _get_tensors_from_df(eval_df)
        model = fit_surrogate_model(train_X, train_Y, BOUNDS, model_name=model_name)
    
    # Build the table structure in a hierarchical format:
    # Each row represents one factor combination: camera_viewpoint, table_height, x, y, RMSE
    # Include ALL factor combinations in the design space
    # For each combination, get model prediction and calculate RMSE vs ground truth where available
    all_rows = []
    
    # Move model to appropriate device
    device = next(model.parameters()).device
    
    for viewpoint_idx in viewpoints:
        for table_height in table_heights:
            for x in x_vals:
                for y in y_vals:
                    # Construct test point based on viewpoint representation
                    if VIEWPOINT_REPRESENTATION == "index":
                        # Factors: [x, y, table_height, viewpoint_index]
                        test_point = torch.tensor([[x, y, table_height, viewpoint_idx]], **tkwargs)
                    else:
                        # Factors: [x, y, table_height, camera_azimuth, camera_elevation, camera_distance]
                        vp_params = get_viewpoint_params(int(viewpoint_idx))
                        if vp_params is None:
                            continue
                        test_point = torch.tensor([[
                            x, y, table_height,
                            vp_params['azimuth'],
                            vp_params['elevation'],
                            vp_params['distance']
                        ]], **tkwargs)
                    
                    test_point = test_point.to(device)
                    with torch.no_grad():
                        posterior = model.posterior(test_point)
                        pred_mean = posterior.mean
                        # Handle shape
                        while pred_mean.ndim > 1 and pred_mean.shape[-1] == 1:
                            pred_mean = pred_mean.squeeze(-1)
                        if pred_mean.ndim == 2:
                            pred_mean = pred_mean.mean(dim=0)
                        pred_value = pred_mean.item()
                    
                    # Initialize row data
                    row_data = {
                        'camera_viewpoint': int(viewpoint_idx),
                        'table_height': int(table_height),
                        'x': x,
                        'y': y,
                        'RMSE': np.nan  # Default to NaN if no ground truth
                    }
                    
                    # Calculate RMSE if ground truth is available
                    key = (round(float(x), 1), round(float(y), 1), round(float(table_height)), round(float(viewpoint_idx)))
                    if key in gt_lookup:
                        gt_value = gt_lookup[key]
                        # Calculate absolute error (RMSE for a single point is just absolute error)
                        error = abs(pred_value - gt_value)
                        row_data['RMSE'] = error
                    
                    all_rows.append(row_data)
    
    # Create DataFrame
    summary_df = pd.DataFrame(all_rows)
    
    # Save to CSV
    summary_df.to_csv(output_file, index=False)
    print(f"Saved RMSE summary table to {output_file}.")
    print(f"Table shape: {summary_df.shape}")
    print(f"Columns: camera_viewpoint, table_height, x, y, RMSE")
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Visualization script for evaluation results.")
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', required=True, help='Visualization command to run')

    # --- Command: plot-points-xy ---
    parser_points = subparsers.add_parser('plot-points-xy', help='Plot tested points colored by outcome')
    parser_points.add_argument('--results_file', type=str, required=True, help='Path to the evaluation_results.csv file')
    parser_points.add_argument('--output_file', type=str, default='visualizations/robo_eval/tested_points.png', help='Output file path for the plot')
    parser_points.add_argument('--task', type=str, default=None, help='Task name for configuration')

    # --- Command: plot-active-xy ---
    parser_active = subparsers.add_parser('plot-active-xy', help='Plot active learning diagnostics (model mean, acqf)')
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
    parser_active.add_argument(
        '--table_height', type=float,
        help='Table height value to use for visualization'
    )
    parser_active.add_argument(
        '--camera_viewpoint', type=int,
        help='Camera viewpoint index to use for visualization'
    )
    parser_active.add_argument('--task', type=str, default=None, help='Task name for configuration')

    # --- Command: animate-active-xy ---
    parser_animate = subparsers.add_parser('animate-active-xy', help='Animate active learning diagnostics over trials')
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
    parser_animate.add_argument(
        '--table_height', type=float,
        help='Table height value to use for visualization'
    )
    parser_animate.add_argument(
        '--camera_viewpoint', type=int,
        help='Camera viewpoint index to use for visualization'
    )

    # --- Command: plot-comparison-xy ---
    parser_compare = subparsers.add_parser('plot-comparison-xy', help='Compare model error against ground truth')
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
        '--table_height', type=float,
        help='Table height value to use for visualization'
    )
    parser_compare.add_argument(
        '--camera_viewpoint', type=int,
        help='Camera viewpoint index to use for visualization'
    )
    parser_compare.add_argument(
        '--plot_mode', type=str, choices=['error', 'mean'], default='mean',
        help="Type of plot to generate: 'error' (vs GT) or 'mean' (model output)"
    )
    
    # --- Command: plot-metrics-vs-trials ---
    parser_metrics = subparsers.add_parser('plot-metrics-vs-trials', help='Plot log-likelihood and RMSE vs trials for active and IID testing')
    parser_metrics.add_argument('--active_results_file', type=str, default=None, help='Path to active testing results CSV (single run, for backward compatibility)')
    parser_metrics.add_argument('--iid_results_file', type=str, default=None, help='Path to IID testing results CSV (single run, for backward compatibility)')
    parser_metrics.add_argument('--add_active_results_file', action='append', help='Add an active testing results CSV file (can be used multiple times for multiple runs)')
    parser_metrics.add_argument('--add_iid_results_file', action='append', help='Add an IID testing results CSV file (can be used multiple times for multiple runs)')
    parser_metrics.add_argument('--gt_results_file', type=str, required=True, help='Path to ground truth test set CSV')
    parser_metrics.add_argument('--output_file', type=str, default='visualizations/robo_eval/metrics_vs_trials.png', help='Output file path')
    parser_metrics.add_argument('--model_name', type=str, default='SingleTaskGP', help='Surrogate model name')
    parser_metrics.add_argument('--task', type=str, default=None, help='Task name for configuration')

    # --- Command: create-rmse-table ---
    parser_table = subparsers.add_parser('create-rmse-table', help='Create RMSE summary table across all factor combinations')
    parser_table.add_argument('--eval_results_file', type=str, required=True, help='Path to evaluation results CSV (used to train model)')
    parser_table.add_argument('--gt_results_file', type=str, required=True, help='Path to ground truth results CSV (used to calculate errors)')
    parser_table.add_argument('--output_file', type=str, default='visualizations/robo_eval/rmse_summary_table.csv', help='Output CSV file path')
    parser_table.add_argument('--model_name', type=str, default='SingleTaskGP', help='Surrogate model name')
    parser_table.add_argument('--task', type=str, default=None, help='Task name for configuration')

    args = parser.parse_args()

    # --- Create output directory if it doesn't exist ---
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: '{output_dir}'")

    # --- Execute the chosen command ---
    if args.command == 'plot-points-xy':
        df = _load_data(args.results_file)
        plot_tested_points_xy(df, args.output_file, task_name=args.task)

    elif args.command == 'plot-active-xy':
        df = _load_data(args.results_file)
        plot_active_learning_xy(df, args.output_file, args.grid_resolution, args.model_name, args.acq_func_name, 
                            args.table_height, args.camera_viewpoint, results_file=args.results_file, task_name=args.task)

    elif args.command == 'animate-active-xy':
        df = _load_data(args.results_file)
        animate_active_learning_xy(df, args.output_file, args.grid_resolution, args.model_name, args.acq_func_name, 
                            args.table_height, args.camera_viewpoint, args.interval)

    elif args.command == 'plot-comparison-xy':
        gt_df = _load_data(args.gt_results_file)
        results_list = []
        if not args.add_results_file:
            print("Warning: No --add_results_file files provided for comparison.")
        else:
            for label, filepath in args.add_results_file:
                df = _load_data(filepath)
                results_list.append((label, df))
        plot_comparison_xy(results_list, gt_df, args.output_file, args.grid_resolution, args.model_name,
                       args.table_height, args.camera_viewpoint, args.plot_mode)

    elif args.command == 'plot-metrics-vs-trials':
        gt_df = _load_data(args.gt_results_file)
        
        # Collect active results files
        active_results_files = []
        active_dfs = []
        if args.active_results_file:
            active_results_files.append(args.active_results_file)
            active_dfs.append(_load_data(args.active_results_file))
        if args.add_active_results_file:
            for filepath in args.add_active_results_file:
                active_results_files.append(filepath)
                active_dfs.append(_load_data(filepath))
        
        # Collect IID results files
        iid_results_files = []
        iid_dfs = []
        if args.iid_results_file:
            iid_results_files.append(args.iid_results_file)
            iid_dfs.append(_load_data(args.iid_results_file))
        if args.add_iid_results_file:
            for filepath in args.add_iid_results_file:
                iid_results_files.append(filepath)
                iid_dfs.append(_load_data(filepath))
        
        if len(active_dfs) == 0 and len(iid_dfs) == 0:
            print("Error: At least one active or IID results file must be provided.")
            return
        
        plot_metrics_vs_trials(active_dfs=active_dfs if active_dfs else None,
                              iid_dfs=iid_dfs if iid_dfs else None,
                              gt_df=gt_df,
                              output_file=args.output_file,
                              model_name=args.model_name,
                              task_name=args.task,
                              active_results_files=active_results_files if active_results_files else None,
                              iid_results_files=iid_results_files if iid_results_files else None)

    elif args.command == 'create-rmse-table':
        eval_df = _load_data(args.eval_results_file)
        gt_df = _load_data(args.gt_results_file)
        create_rmse_summary_table(eval_df, gt_df, args.output_file, args.model_name, args.task, 
                                 eval_results_file=args.eval_results_file)


if __name__ == "__main__":
    main()