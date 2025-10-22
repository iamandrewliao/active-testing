"""Visualization functions"""

import argparse
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
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
    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        input_transform=Normalize(d=train_X.shape[-1], bounds=BOUNDS),
        outcome_transform=Standardize(m=1),
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
    plt.xlim(BOUNDS[0, 0], BOUNDS[1, 0])
    plt.ylim(BOUNDS[0, 1], BOUNDS[1, 1])
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
    model = _fit_gp_model(train_X, train_Y)
    
    # 2. Get the acquisition function (JES)
    normalized_bounds = torch.tensor([[0.0] * 2, [1.0] * 2], **tkwargs)
    optimal_inputs, optimal_outputs = get_optimal_samples(
        model, bounds=normalized_bounds, num_optima=16
    )
    jes = qJointEntropySearch(
        model=model,
        optimal_inputs=optimal_inputs,
        optimal_outputs=optimal_outputs,
        estimation_type="LB",
    )
    
    # 3. Create grid and evaluate
    grid_tensor = get_grid_points(grid_resolution, BOUNDS, tkwargs)
    grid_shape = (grid_resolution, grid_resolution)
    
    # Get model predictions
    with torch.no_grad():
        posterior = model.posterior(grid_tensor)
        mean = posterior.mean.numpy().reshape(grid_shape)
        
        # Get acquisition values
        # Note: acqf expects input shape [N, q, D], so we add a 'q' dim
        # e.g. for a 100x100x2 grid, input shape should be [10000, 1, 2]
        # https://botorch.readthedocs.io/en/latest/acquisition.html#botorch.acquisition.joint_entropy_search.qJointEntropySearch
        acq_values = jes(grid_tensor.reshape(-1, 1, 2)).numpy().reshape(grid_shape)

    # 4. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    plot_extent = [BOUNDS[0, 0], BOUNDS[1, 0], BOUNDS[0, 1], BOUNDS[1, 1]]

    # --- Plot 1: Surrogate Model Mean ---
    ax = axes[0]
    im = ax.imshow(mean, origin='lower', extent=plot_extent, cmap='viridis_r', vmin=0, vmax=4, aspect='equal')
    fig.colorbar(im, ax=ax, label='Predicted Outcome (Mean)')
    # Overlay tested points
    ax.scatter(df['x'], df['y'], c='red', edgecolors='k', s=50, label='Tested Points')
    ax.set_title(f'Surrogate Model (GP Mean) (N={len(df)})')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()

    # --- Plot 2: Acquisition Function ---
    ax = axes[1]
    im = ax.imshow(acq_values, origin='lower', extent=plot_extent, cmap='magma', aspect='equal')
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
    model_gt = _fit_gp_model(X_gt, Y_gt)

    grid_tensor = get_grid_points(grid_resolution, BOUNDS, tkwargs)
    grid_shape = (grid_resolution, grid_resolution)
    with torch.no_grad():
        mean_gt = model_gt.posterior(grid_tensor).mean.numpy().reshape(grid_shape)

    # 2. Fit each model, get predictions, and calculate error
    errors_and_stats = []
    for label, df in results_list:
        X_model, Y_model = _get_tensors_from_df(df)
        print(f"  Fitting {label} model (N={len(X_model)})...")
        model = _fit_gp_model(X_model, Y_model)
        
        with torch.no_grad():
            mean_model = model.posterior(grid_tensor).mean.numpy().reshape(grid_shape)
        
        error_map = np.abs(mean_model - mean_gt)
        mae = np.mean(error_map)
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
    # Ensure 'axes' is always an array for consistent indexing, even if num_plots=1
    if num_plots == 1:
        axes = [axes]
        
    plot_extent = [BOUNDS[0, 0], BOUNDS[1, 0], BOUNDS[0, 1], BOUNDS[1, 1]]
    
    # Determine a common error scale (if there are any error plots)
    vmax = 0
    if errors_and_stats:
        vmax = max(stats['error_map'].max() for stats in errors_and_stats)

    # --- Plot 1...N: Model Errors ---
    for i, stats in enumerate(errors_and_stats):
        ax = axes[i]
        im = ax.imshow(stats['error_map'], origin='lower', extent=plot_extent, cmap='hot', vmin=0, vmax=vmax, aspect='equal')
        fig.colorbar(im, ax=ax, label='Absolute Error')
        ax.set_title(f"{stats['label']} Model Error (N={stats['n']})\nMAE = {stats['mae']:.4f}")
        ax.set_xlabel('X Position')
        if i == 0:
            ax.set_ylabel('Y Position') # Only set Y label on the first plot

    # --- Plot N+1: Ground Truth Model ---
    ax = axes[-1] # Always the last plot
    im = ax.imshow(mean_gt, origin='lower', extent=plot_extent, cmap='viridis_r', vmin=0, vmax=4, aspect='equal')
    fig.colorbar(im, ax=ax, label='Predicted Outcome (Mean)')
    ax.set_title(f'Ground Truth Model (N={len(X_gt)})')
    ax.set_xlabel('X Position')
    
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
        "--grid_resolution",
        type=int,
        default=10,
        help="The resolution (N) for the N_x_N grid."
    )    
    parser_active.add_argument('--results_file', type=str, required=True, help='Path to the active_results.csv file')
    parser_active.add_argument('--output_file', type=str, default='visualizations/active_plots.png', help='Output file path for the plot')

    # --- Command: plot-comparison ---
    parser_compare = subparsers.add_parser('plot-comparison', help='Compare model error against ground truth')
    parser_compare.add_argument(
        "--grid_resolution",
        type=int,
        default=10,
        help="The resolution (N) for the N_x_N grid."
    )
    parser_compare.add_argument(
        '--gt', 
        type=str, 
        required=True, 
        help='Path to the ground_truth_results.csv file (M samples)'
    )
    parser_compare.add_argument(
        '--model', 
        action='append', 
        nargs=2, 
        metavar=('LABEL', 'FILEPATH'), 
        help='Add a model to compare. Can be used multiple times. E.g., --model Active active.csv'
    )
    parser_compare.add_argument(
        '--output_file', 
        type=str, 
        default='visualizations/comparison_plot.png', 
        help='Output file path for the plot'
    )

    args = parser.parse_args()

    # --- Execute the chosen command ---
    if args.command == 'plot-points':
        df = _load_data(args.results_file)
        plot_tested_points(df, args.output_file)
        
    elif args.command == 'plot-active':
        df = _load_data(args.results_file)
        plot_active_learning(df, args.output_file, args.grid_resolution)
        
    elif args.command == 'plot-comparison':
        gt_df = _load_data(args.gt)
        
        results_list = []
        if not args.model:
            print("Warning: No --model files provided. Only plotting the Ground Truth model.")
        else:
            for label, filepath in args.model:
                df = _load_data(filepath)
                results_list.append((label, df))
        
        plot_comparison(results_list, gt_df, args.output_file, args.grid_resolution)


if __name__ == "__main__":
    main()