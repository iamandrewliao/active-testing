"""Test active learning/testing components on simulated data from test functions."""

import torch
import pandas as pd
from tqdm import tqdm

import argparse
from testers import ActiveTester, IIDSampler
from utils import fit_surrogate_model, calculate_log_likelihood, calculate_rmse, calculate_wasserstein_distance

# Import BoTorch stuff
from botorch.test_functions import Branin # test function
from botorch.utils.sampling import draw_sobol_samples # quasi-random sampling

import seaborn as sns
import matplotlib.pyplot as plt

# Set up torch device and data type
tkwargs = {"dtype": torch.double, "device": "cpu"}

# --- Parameters ---
N_POINTS = 256
N_INIT = 10 # Number of initial random points for both samplers
N_TRIALS = 30 # Number of new points each sampler will select (IID will continue to sample randomly)

# We'll use the Branin test function (negated, so we maximize)
true_function = Branin(negate=True)
# The Branin function is defined on bounds: [-5, 10] x [0, 15]
BOUNDS = torch.tensor([[-5.0, 0.0], [10.0, 15.0]], **tkwargs)


def simulate_data():
    """
    Samples N_POINTS points, evaluates test function on them

    Returns:
        gt_df: DataFrame of simulated points (X) and function outputs (Y)
        init_df: DataFrame of randomly selected simulated data (X, Y) to start samplers (e.g. initial training data for ActiveTester)
        sim_pts: N_POINTS x D tensor of all simulated points (X)
    """
    # 1. Generate quasi-random points
    sim_pts = draw_sobol_samples(
        bounds=BOUNDS,
        n=N_POINTS,
        q=1,  # q=1 means we get N points of 1 sample each
        seed=0  # Make it reproducible
    ).squeeze(1) # Squeeze from [N, q, D] to shape [N, D]
    
    # 2. Evaluate the true function on all grid points
    with torch.no_grad():
        true_Y = true_function(sim_pts).view(-1) # shape [N]

    # 3. Create the ground truth data
    gt_df = pd.DataFrame({
        'x1': sim_pts[:, 0].tolist(),
        'x2': sim_pts[:, 1].tolist(),
        'y': true_Y.tolist(),
    })
    
    # 4. Select initial points (same for all samplers)
    torch.manual_seed(0) # for reproducible starting points
    init_indices = torch.randperm(sim_pts.shape[0])[:N_INIT]
    
    init_df = gt_df.iloc[init_indices].copy()
    
    # Add dummy columns to match real data format
    init_df['trial'] = range(1, N_INIT + 1)
    init_df['mode'] = 'initial_random'
    
    print(f"Generated ground truth data ({len(gt_df)} points) and initial data ({len(init_df)} points) to start samplers.")
    return gt_df, init_df, sim_pts


def create_test_set():
    """Creates a fixed, held-out test set for evaluation."""
    X_test = draw_sobol_samples(
        bounds=BOUNDS,
        n=N_POINTS,
        q=1,
        seed=12345
    ).squeeze(1)
    with torch.no_grad():
        Y_test = true_function(X_test).view(-1)
    test_df = pd.DataFrame({
        'x1': X_test[:, 0].tolist(),
        'x2': X_test[:, 1].tolist(),
        'y': Y_test.tolist(),
    })
    print(f"Generated {N_POINTS} test points...")
    return test_df


# def run_loop(sampler, sampler_mode, true_function, num_trials):
#     """
#     Runs a mock evaluation loop, "testing" points against the true function
#     and feeding the results back into the sampler.
#     Returns a DataFrame of the newly sampled points.
#     """    
#     sampled_data = []
    
#     # Run the main loop
#     for i in tqdm(range(num_trials), desc=f"Running {sampler_mode} trials"):
#         # 1. Get the next point to test
#         next_pt = sampler.get_next_point()
#         # 2. Evaluate the point w/ the test function
#         outcome = true_function(next_pt).view(-1)
#         # 3. Update the sampler with the new data
#         sampler.update(next_pt, outcome) # e.g. adds to active sampler's training data
#         # 4. Store the result
#         sampled_data.append({
#             'trial': N_INIT + i + 1,
#             'mode': sampler_mode,
#             'x1': next_pt[0].item(),
#             'x2': next_pt[1].item(),
#             'y': outcome.item(),
#         })
        
#     return pd.DataFrame(sampled_data)


def calculate_metrics(model, X_test, Y_test):
    rmse = calculate_rmse(model, X_test, Y_test)
    ll = calculate_log_likelihood(model, X_test, Y_test)
    wass_dist = calculate_wasserstein_distance(model, X_test, Y_test)
    # print(f"RMSE: {rmse:.4f}")
    # print(f"Test log-likelihood: {ll:.4f}")
    # print(f"Wasserstein distance: {wass_dist:.4f}")
    return rmse, ll, wass_dist


def compare_results(results_df):
    """Compares active and IID results against the test set using various metrics (RMSE, log-likelihood, Wasserstein distance)."""
    print("\n--- Plotting Results ---")
    
    sns.set(style="whitegrid", font_scale=1.2)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharex=True)
    
    # 1. RMSE Plot (Lower is better)
    sns.lineplot(data=results_df, x='trial', y='rmse', hue='sampler', ax=axes[0])
    axes[0].set_title('Model RMSE vs. Trials')
    axes[0].set_ylabel('RMSE (on Test Set)')
    axes[0].set_xlabel('Trial Number')

    # 2. MLL Plot (Higher is better)
    sns.lineplot(data=results_df, x='trial', y='mll', hue='sampler', ax=axes[1])
    axes[1].set_title('Mean Log-Likelihood vs. Trials')
    axes[1].set_ylabel('Mean Log-Likelihood (on Test Set)')
    axes[1].set_xlabel('Trial Number')

    # 3. Wasserstein Plot (Lower is better)
    sns.lineplot(data=results_df, x='trial', y='w_dist', hue='sampler', ax=axes[2])
    axes[2].set_title('Wasserstein Distance (Y_pred vs Y_true)')
    axes[2].set_ylabel('Wasserstein-1D Distance')
    axes[2].set_xlabel('Trial Number')
    
    plt.suptitle(f'Sampler Comparison (N_init={N_INIT}, N_trials={N_TRIALS})', fontsize=16, y=1.03)
    plt.tight_layout()
    plt.savefig(f"{args.save_path}", dpi=300, bbox_inches='tight')
    print(f"Saved plot to {args.save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_path',
        type=str,
        default='./visualizations/test/sampler_comparison.png',
        help='Path where the visualization will be saved.'
    )
    args = parser.parse_args()

    print("--- Starting Active Learning Test on Simulated Data ---")
    
    # Generate all data
    gt_df, init_df, sim_pts = simulate_data()
    test_df = create_test_set()
    
    # Get initial tensors for sampler constructors
    X_init = torch.tensor(init_df[['x1', 'x2']].values, **tkwargs)
    Y_init = torch.tensor(init_df['y'].values, **tkwargs).unsqueeze(1)

    # Get test tensors
    X_test = torch.tensor(test_df[['x1', 'x2']].values, **tkwargs)
    Y_test = torch.tensor(test_df['y'].values, **tkwargs)

    active_results = []
    iid_results = []
    
    # Fit initial model (same for both IID and active; use SingleTaskGP because it's much faster)
    init_model = fit_surrogate_model(X_init, Y_init, BOUNDS, model_name="SingleTaskGP")
    
    rmse, ll, wd = calculate_metrics(init_model, X_test, Y_test)
    active_results.append({
        "sampler": 'active',
        "trial": 0,
        "rmse": rmse,
        "mll": ll,
        "w_dist": wd,
    })
    iid_results.append({
        "sampler": 'iid',
        "trial": 0,
        "rmse": rmse,
        "mll": ll,
        "w_dist": wd,
    })

    # Initialize the testers
    # Sample points "to use for MC-integrating the posterior variance. Usually, these are qMC samples on the whole design space" (required by qNIPV)
    mc_integration_points = draw_sobol_samples(
        bounds=BOUNDS, n=128, q=1
    ).squeeze(1).to(**tkwargs)
    active_sampler = ActiveTester(X_init, Y_init, BOUNDS, sim_pts, mc_integration_points)
    iid_sampler = IIDSampler(sim_pts)

    # Active test
    for i in tqdm(range(N_TRIALS), desc="Active Trials"):
        # 1. Fit model and get next point.
        next_pt = active_sampler.get_next_point() # Shape [D]
        # 2. Evaluate the point w/ the test function
        outcome = true_function(next_pt).view(-1) # Shape [1]
        # 3. Update the sampler's data (does not refit)
        active_sampler.update(next_pt, outcome)
        # 4. Calculate metrics using the model that was just fit inside get_next_point()
        rmse, ll, wd = calculate_metrics(active_sampler.model, X_test, Y_test)
        active_results.append({
            "sampler": 'active',
            "trial": i+1,
            "rmse": rmse,
            "mll": ll,
            "w_dist": wd,
        })

    # IID test
    # We must manually track the training data for the IID model to update our model
    X_train_iid = X_init.clone()
    Y_train_iid = Y_init.clone()
    for i in tqdm(range(N_TRIALS), desc="IID Trials"):
        # 1. Get the next point
        next_pt = iid_sampler.get_next_point() # Shape [D]
        # 2. Evaluate the point w/ the test function
        outcome = true_function(next_pt).view(-1) # Shape [1]
        # 3. Update the sampler (does nothing, but good practice)
        iid_sampler.update(next_pt, outcome)
        # 4. Manually add data to our IID training set
        X_train_iid = torch.cat([X_train_iid, next_pt.unsqueeze(0)], dim=0) # [D] -> [1, D]
        Y_train_iid = torch.cat([Y_train_iid, outcome.unsqueeze(-1)], dim=0) # [1] -> [1, 1]
        # 5. Fit a *new* GP model on all IID data so far (SingleTaskGP b/c much faster)
        current_iid_model = fit_surrogate_model(X_train_iid, Y_train_iid, BOUNDS, model_name="SingleTaskGP")
        # 6. Calculate metrics
        rmse, ll, wd = calculate_metrics(current_iid_model, X_test, Y_test)
        iid_results.append({
            "sampler": 'iid',
            "trial": i+1,
            "rmse": rmse,
            "mll": ll,
            "w_dist": wd,
        })

    # Compare results
    results_df = pd.concat([pd.DataFrame(active_results), pd.DataFrame(iid_results)], ignore_index=True)
    print("\n--- Final Metrics ---")
    compare_results(results_df)