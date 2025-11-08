"""Test active learning/testing components on simulated data from test functions."""

import torch
import pandas as pd
from tqdm import tqdm

# Import from your project files
from samplers import ActiveTester, IIDSampler
from viz import _get_tensors_from_df  # Import your visualization script as a module

# Import BoTorch stuff
from botorch.test_functions import Branin # test function
from botorch.utils.sampling import draw_sobol_samples # quasi-random sampling

# Set up torch device and data type
tkwargs = {"dtype": torch.double, "device": "cpu"}

# --- Test Parameters ---
N_POINTS = 1024
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
    print(f"Generating {N_POINTS} points for the data pool...")
    # 1. Generate quasi-random points
    sim_pts = draw_sobol_samples(
        bounds=BOUNDS,
        n=N_POINTS,
        q=1,  # q=1 means we get N points of 1 sample each
        seed=0  # Make it reproducible
    ).squeeze(1) # Squeeze from [N, q, D] to shape [N, D]
    print(f"Sim points (first 5): {sim_pts[:5]}")
    
    # 2. Evaluate the true function on all grid points
    with torch.no_grad():
        true_Y = true_function(sim_pts).view(-1, 1) # shape [N, 1]
        print(f"True function outputs (first 5): {true_Y[:5]}")

    # 3. Create the ground truth data
    gt_data = []
    for i in range(sim_pts.shape[0]):
        gt_data.append({
            'x1': sim_pts[i, 0].item(),
            'x2': sim_pts[i, 1].item(),
            'y': true_Y[i, 0].item(),
        })
    gt_df = pd.DataFrame(gt_data)
    
    # 4. Select initial points (same for all samplers)
    torch.manual_seed(0) # for reproducible starting points
    init_indices = torch.randperm(sim_pts.shape[0])[:N_INIT]
    
    init_df = gt_df.iloc[init_indices].copy()
    
    # Add dummy columns to match real data format
    init_df['trial'] = range(1, N_INIT + 1)
    init_df['mode'] = 'initial_random'
    
    print(f"Generated ground truth data ({len(gt_df)} points) and initial data ({len(init_df)} points) to start samplers.")
    return gt_df, init_df, sim_pts


def run_loop(sampler, sampler_mode, true_function, init_df, num_trials):
    """
    Runs a mock evaluation loop, "testing" points against the true function
    and feeding the results back into the sampler.
    Returns a DataFrame of the newly sampled points.
    """
    # All samplers must be updated with the initial data
    # (ActiveTester's __init__ already does this)
    
    sampled_data = []
    
    # Run the main loop

    for i in tqdm(range(num_trials), desc=f"Running {sampler_mode} trials"):
        # 1. Get the next point to test
        next_pt = sampler.get_next_point()
        # 2. Evaluate the point w/ the test function
        outcome = true_function(next_pt).view(-1)
        # 3. Update the sampler with the new data
        sampler.update(next_pt, outcome) # e.g. adds to active sampler's training data
        # 4. Store the result
        sampled_data.append({
            'trial': N_INIT + i + 1,
            'mode': sampler_mode,
            'x1': next_pt[0].item(),
            'x2': next_pt[1].item(),
            'y': outcome.item(),
        })
        
    return pd.DataFrame(sampled_data)


def compare_results(active_df, iid_df):
    pass


if __name__ == "__main__":
    print("--- Starting Active Learning Test on Simulated Data ---")
    
    # 1. Generate all data
    gt_df, init_df, sim_pts = simulate_data()
    print(gt_df.head())
    print(init_df.head())
    
    # Get initial tensors for sampler constructors
    init_X = torch.tensor(init_df[['x1', 'x2']].values, **tkwargs)
    init_Y = torch.tensor(init_df['y'].values, **tkwargs).unsqueeze(1)

    # 2. Run Active Test
    # print("\n" + "="*50)
    # print(f"RUNNING ACTIVE ({N_TRIALS} trials)")
    # print("="*50)
    active_sampler = ActiveTester(init_X, init_Y, BOUNDS, sim_pts)
    active_new_df = run_loop(active_sampler, 'active', true_function, init_df, N_TRIALS)
    active_results_df = pd.concat([init_df, active_new_df]).reset_index(drop=True)

    # 3. Run IID Test
    # print("\n" + "="*50)
    # print(f"Running IID test ({N_TRIALS} trials)")
    # print("="*50)
    iid_sampler = IIDSampler(sim_pts)
    iid_new_df = run_loop(iid_sampler, 'iid', true_function, init_df, N_TRIALS)
    iid_results_df = pd.concat([init_df, iid_new_df]).reset_index(drop=True)

    # 4. Compare results
    compare_results(active_results_df, iid_results_df)