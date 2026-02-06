'''
Based on active testing results, determine what data to collect next.
'''

import torch
from viz import _load_data, _get_tensors_from_df, _fit_gp_model
from utils import get_design_points_test, is_valid_point


# --- Define Constants ---
tkwargs = {"dtype": torch.double, "device": "cpu"}
# Define the search space bounds (hardcoded from viz.py)
BOUNDS = torch.tensor([[0.0, 0.0], [1.0, 1.0]], **tkwargs)

def find_certain_failures(results_file, grid_resolution, mean_pct, variance_pct):
    """
    Finds points on a grid where the surrogate model, trained on
    results_file, predicts failure with high certainty.
    
    Args:
        results_file (str): Path to the active testing results CSV.
        grid_resolution (int): The N for the N_x_N grid.
        mean_pct (float): Predicted mean outcome in this percentile is a "failure".
        variance_pct (float): Predicted variance in this percentile is "certain".
    """
    
    # 1. Load data
    print(f"Loading training data from {results_file}...")
    df = _load_data(results_file)
    if df is None or df.empty:
        print(f"Could not load data from {results_file}. Make sure the file exists.")
        return

    # 2. Fit the model
    train_X, train_Y = _get_tensors_from_df(df)
    model = _fit_gp_model(train_X, train_Y)
    model.eval() # Set model to evaluation mode

    # 3. Generate the target grid
    print(f"Generating {grid_resolution}x{grid_resolution} analysis grid...")
    grid_points = get_design_points_test(grid_resolution, BOUNDS, tkwargs)

    print("Checking grid point validity...")
    # We check all points on the grid for validity
    valid_mask = torch.tensor(
        [is_valid_point(p) for p in grid_points], 
        dtype=torch.bool, 
        device=tkwargs["device"]
    )
    print(f"Found {valid_mask.sum()} valid points out of {len(grid_points)} total grid points.")

    # 4. Get model predictions on the grid
    print("Getting model predictions on the grid...")
    with torch.no_grad():
        posterior = model.posterior(grid_points)
        mean_preds = posterior.mean.squeeze()
        variance_preds = posterior.variance.squeeze()

    # 5. Filter for certain failures
    mean_threshold = torch.quantile(mean_preds, mean_pct)
    var_threshold = torch.quantile(variance_preds, variance_pct)
    print(f"Filtering points where mean < {mean_threshold} and variance < {var_threshold}...")
    
    # Create boolean masks
    is_failure = mean_preds < mean_threshold
    is_certain = variance_preds < var_threshold
    
    # Combine masks
    is_certain_failure_and_valid = is_failure & is_certain & valid_mask
    
    # Get the grid points that satisfy the condition
    certain_failure_points = grid_points[is_certain_failure_and_valid]
    
    # Also get the corresponding predictions for verification
    certain_failure_means = mean_preds[is_certain_failure_and_valid]
    certain_failure_vars = variance_preds[is_certain_failure_and_valid]

    # 6. Print results
    print("\n--- Results ---")
    if certain_failure_points.shape[0] == 0:
        print("No grid points matched the criteria.")
    else:
        print(f"Found {certain_failure_points.shape[0]} grid points matching the criteria:")
        print(" (x, y) \t\t | Predicted Mean | Predicted Variance")
        print("-" * 50)
        for i in range(certain_failure_points.shape[0]):
            point = certain_failure_points[i]
            mean = certain_failure_means[i].item()
            var = certain_failure_vars[i].item()
            print(f" ({point[0].item():.3f}, {point[1].item():.3f}) \t | {mean:.4f} \t | {var:.4f}")

# --- Main execution ---
if __name__ == "__main__":
    # --- Parameters ---
    
    # Use the active_40 results file, as it's a good dataset
    RESULTS_FILE_PATH = 'results/active_40_results.csv'
    
    # User specified a 11x11 grid
    GRID_RESOLUTION = 11

    # Collect data for predictions with low variance and low mean outcomes (bottom n %)
    # 0.05 is a good starting point (std dev ~0.22)
    OUTCOME_MEAN_PERCENTILE = 0.7
    OUTCOME_VAR_PERCENTILE = 0.5

    find_certain_failures(
        results_file=RESULTS_FILE_PATH,
        grid_resolution=GRID_RESOLUTION,
        mean_pct=OUTCOME_MEAN_PERCENTILE,
        variance_pct=OUTCOME_VAR_PERCENTILE
    )