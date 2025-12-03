import torch
import argparse
import pandas as pd

from botorch.models.transforms import Normalize, Standardize
from botorch.models import SingleTaskGP
from botorch.models.kernels import InfiniteWidthBNNKernel
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize, Standardize
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP, FullyBayesianSingleTaskGP
from botorch.fit import fit_fully_bayesian_model_nuts

from botorch.acquisition.analytic import PosteriorStandardDeviation
from botorch.acquisition.active_learning import qNegIntegratedPosteriorVariance
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.bayesian_active_learning import qBayesianActiveLearningByDisagreement
from botorch.optim import optimize_acqf, optimize_acqf_discrete

from scipy.stats import wasserstein_distance
from scipy.stats import gaussian_kde

from EfficientEval import fit_mdn_model, MDN_BALD


def fit_surrogate_model(train_X, train_Y, bounds, model_name="SingleTaskGP"):
    """
    Fits and returns surrogate model.
    """
    # print(f"Fitting {model_name} model with {train_X.shape[0]} points...")
    input_transform = Normalize(d=train_X.shape[-1], bounds=bounds) # normalizes X to [0, 1]^d
    outcome_transform = Standardize(m=1) # standardizes Y to have zero mean and unit variance
    if model_name=="SingleTaskGP" or model_name == "I-BNN": # infinite-width BNN is just a GP with a special kernel
        if model_name == "SingleTaskGP":
            kernel = None
        elif model_name == "I-BNN":
            kernel = InfiniteWidthBNNKernel(depth=3)
        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            covar_module=kernel
            )
        mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
        fit_gpytorch_mll(mll)
    elif model_name=="FullyBayesianSingleTaskGP":
        model = FullyBayesianSingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            input_transform=input_transform,
            outcome_transform=outcome_transform
        )
        fit_fully_bayesian_model_nuts(model)
    elif model_name=="SaasFullyBayesianSingleTaskGP":
        model = SaasFullyBayesianSingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            input_transform=input_transform,
            outcome_transform=outcome_transform
        )
        fit_fully_bayesian_model_nuts(model)
    elif model_name == "MDN":
        model = fit_mdn_model(train_X, train_Y, bounds)
    return model


def get_acquisition_function(model, acq_func_name, mc_points=None):
    if acq_func_name == "PSD":
        acq_func = PosteriorStandardDeviation(model=model)
    elif acq_func_name == "qNIPV":
        if mc_points is None:
             raise ValueError("mc_points must be provided to run_acquisition for qNIPV.")
        acq_func = qNegIntegratedPosteriorVariance(
            model=model, 
            mc_points=mc_points
        )
    elif acq_func_name == "qBALD":
        acq_func = qBayesianActiveLearningByDisagreement(model=model)
    elif acq_func_name == "MDN_BALD":
        acq_func = MDN_BALD(model=model)
    else:
        raise ValueError(f"Unknown acq_func_name: {acq_func_name}")
    
    return acq_func


def optimize_acq_func(acq_func, design_space, discrete=True, normalized_bounds=None):
    if discrete:
        candidate, _ = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,
            choices=design_space,
        )
    else:
        if normalized_bounds is None:
            raise ValueError("normalized_bounds must be provided for continuous optimization.")
        candidate, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=normalized_bounds,
            q=1,
            num_restarts=4,
            raw_samples=256,
        )
    point = candidate.squeeze(0) # Return a 1D tensor
    return point


@torch.no_grad()
def calculate_rmse(model, X_test, Y_test):
    """Calculates the Root Mean Squared Error on the test set."""  
    model.eval()
    posterior = model.posterior(X_test)
    mean = posterior.mean.squeeze(-1) # Shape is [N] or [S, N]
    
    if mean.ndim == 2: # If it's Bayesian (S, N)
        mean = mean.mean(dim=0) # Average over samples to get [N]
    
    return torch.sqrt(torch.mean((mean - Y_test)**2)).item()


@torch.no_grad()
def calculate_log_likelihood(model, X_test, Y_test):
    """Calculates the Mean Log-Likelihood (Log Predictive Density) on the test set."""
    model.eval()
    posterior = model.posterior(X_test)
    pred_dist = posterior.distribution
    
    # log_prob will be [S, N] for Bayesian, or [N] for SingleTaskGP
    log_probs = pred_dist.log_prob(Y_test)
    
    if log_probs.ndim == 2: # Bayesian case [S, N]
        # We want log( E[p(y|x)] ) = log( 1/S * sum( exp(log_p(y|x, theta_s)) ) )
        # This is equivalent to logsumexp(log_p) - log(S)
        num_samples = log_probs.shape[0]
        # Average over the N test points
        mean_log_pred_density = torch.logsumexp(log_probs, dim=0).mean() - torch.log(torch.tensor(num_samples))
        return mean_log_pred_density.item()
    else: # SingleTaskGP case [N]
        return log_probs.mean().item() # Just average over test points


@torch.no_grad()
def calculate_wasserstein_distance(model, X_test, Y_test):
    """Calculates the Wasserstein-1D distance between true and predicted Y-distributions."""
    model.eval()
    posterior = model.posterior(X_test)
    mean = posterior.mean.squeeze(-1) # Shape is [N] or [S, N]
    
    if mean.ndim == 2: # If it's Bayesian (S, N)
        mean = mean.mean(dim=0) # Average over samples to get [N]
    
    y_true_np = Y_test.cpu().numpy()
    y_pred_np = mean.cpu().numpy() # No .detach() needed due to decorator
    
    return wasserstein_distance(y_true_np, y_pred_np)

def get_design_points(resolution, bounds, tkwargs):
    """
    Creates a tensor of all points for arbitrary D dimensions.
    bounds: [2, D] tensor
    """
    dims = bounds.shape[1]
    
    # Generate linspace for each dimension
    # bounds[0, d] is the min of dimension d, bounds[1, d] is the max
    linspaces = [torch.linspace(bounds[0, d], bounds[1, d], resolution, **tkwargs) for d in range(dims)]
    
    # Create meshgrid (indexing='ij' ensures correct order for n-dims)
    dims_linspaces = torch.meshgrid(*linspaces, indexing='ij')
    
    # Stack and reshape to get [N^D, D]
    design_space_tensor = torch.stack(dims_linspaces, dim=-1)
    all_points = design_space_tensor.reshape(-1, dims)
    
    print(f"Generated {all_points.shape[0]} total design points across {dims} dimensions.")
    return all_points

# CHANGE FOR YOUR SETUP
def is_valid_point(point):
    """Automatically filters out any point past my UR5 arm's reachability (approximated by a straight line)"""
    # Equation of the boundary line: y = 1.35x - 0.475
    # A point is 'invalid' if it's on the right side of the line
    x, y = point[0].item(), point[1].item()
    # Calculate if the point is in the invalid region
    # is_invalid = (y <= 1.35*x - 0.475) and (x >= 0.35)
    is_invalid = (y <= 1.5*x - 0.75) and (x >= 0.5) # TO DO: comment this out and replace with the more accurate one above
    
    return not is_invalid  # Return True if the point is valid


def run_evaluation(point, max_steps):
    """
    Simulates a robot evaluation trial for a given point.
    Prompts the user for the outcome.
    """
    # Create a string representation of the point coordinates
    coords_str = ", ".join([f"{val:.3f}" for val in point.tolist()])
    print("-" * 30)
    print(f"🤖 Running trial at position: ({coords_str})")
    
    # --- Get continuous outcome ---
    while True:
        try:
            print("Enter continuous outcome (0.5 increments):")
            print("  0=failed completely, 1=moved to block, 2=grasped block, 3=moved to bowl, 4=dropped (success)")
            continuous_outcome = float(input("Enter outcome (0-4): "))
            
            # check if in [0, 0.5, ..., 4]
            if continuous_outcome in [i*0.5 for i in range(9)]:
                break
            else:
                print("Invalid input. Please enter a number between 0 and 4 in 0.5 increments.")
        except ValueError:
            print("Invalid input. Please enter a number.")
            
    # --- Derive binary outcome ---
    binary_outcome = 1.0 if continuous_outcome == 4.0 else 0.0

    # --- Get number of steps taken ---
    if binary_outcome == 0:
        print("Failure reported. Setting steps to max.")
        steps_taken = max_steps
    else:
        while True:
            try:
                steps_taken = int(input("Enter number of steps taken to complete the task: "))
                if steps_taken >= 0:
                    break
                else:
                    print("Invalid input. Please enter a non-negative integer.")
            except ValueError:
                print("Invalid input. Please enter a number.")
            
    return binary_outcome, continuous_outcome, steps_taken


def load_vla_data(vla_data_path):
    """
    Loads VLA training data from a CSV file.
    Returns a (N, D) tensor.
    """
    if vla_data_path is None:
        return None
    print(f"Loading VLA training distribution from {vla_data_path}...")
    try:
        df = pd.read_csv(vla_data_path)
        
        # Try to find factor columns
        factor_cols = [c for c in df.columns if c.startswith('factor_')]
        if not factor_cols:
            if {'x', 'y'}.issubset(df.columns):
                factor_cols = ['x', 'y']
            else:
                # If no standard names, assume all columns are factors? 
                # Or raise error. Let's assume all numeric columns for flexibility or error.
                # Safer to raise error to enforce schema.
                raise ValueError("VLA CSV must have 'factor_N' or 'x','y' columns.")

        vla_tensor = torch.tensor(df[factor_cols].values, dtype=torch.double)
        return vla_tensor
    except Exception as e:
        print(f"Error loading VLA data: {e}")
        return None


def compute_knn_distance(target_points, reference_points, k=1):
    """
    Computes the distance from each point in target_points to its 
    k-th nearest neighbor in reference_points.
    
    Args:
        target_points: (N, 2) tensor of points to evaluate
        reference_points: (M, 2) tensor of VLA training data
        k: Which neighbor to use (default 1 = nearest neighbor)
        
    Returns:
        (N, 1) tensor of distances
    """
    # Compute pairwise distances: (N, M) matrix
    # efficient distance calculation: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
    # or just use cdist for simplicity and robustness
    dists = torch.cdist(target_points, reference_points)
    
    # Get the k-th smallest distance for each target point
    # topk returns the largest, so we negate or sort. 
    # For small k, topk(largest=False) is best.
    values, _ = torch.topk(dists, k=k, dim=1, largest=False)
    
    # The last column of values is the k-th nearest neighbor distance
    # shape (N, k), we want the k-th one (index k-1)
    kth_dist = values[:, -1].unsqueeze(1) # Shape (N, 1)
    
    return kth_dist

# Sklearn version
# def compute_knn_distance(target_points, reference_points, k=1):
#     """
#     Computes the distance from each point in target_points to its 
#     k-th nearest neighbor in reference_points.
    
#     Args:
#         target_points: (N, 2) tensor of points to evaluate
#         reference_points: (M, 2) tensor of VLA training data
#         k: Which neighbor to use (default 1 = nearest neighbor)
        
#     Returns:
#         (N, 1) tensor of distances
#     """
#     from sklearn.neighbors import NearestNeighbors
#     # sklearn expects numpy arrays
#     # Ensure inputs are on CPU before converting
#     target_np = target_points.detach().cpu().numpy()
#     ref_np = reference_points.detach().cpu().numpy()
    
#     # Fit the NearestNeighbors model on the reference points (VLA data)
#     # algorithm='auto' will choose the best algorithm (KDTree, BallTree, or Brute)
#     nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(ref_np)
    
#     # Find k-nearest neighbors for the target points
#     # distances will be shape (N, k), indices will be (N, k)
#     distances, _ = nbrs.kneighbors(target_np)
    
#     # We want the distance to the k-th neighbor (the last column)
#     kth_dist_np = distances[:, -1]
    
#     # Convert back to tensor, matching the device/dtype of the input
#     kth_dist = torch.tensor(
#         kth_dist_np, 
#         dtype=target_points.dtype, 
#         device=target_points.device
#     ).unsqueeze(1) # Shape (N, 1)
    
#     return kth_dist


def compute_kde_density(target_points, reference_points):
    """
    Computes KDE density estimates for target_points based on reference_points.
    """    
    # Convert to numpy for scipy and transpose because gaussian_kde expects (dims, N)
    ref_np = reference_points.cpu().numpy().T 
    target_np = target_points.cpu().numpy().T
    
    kde = gaussian_kde(ref_np)
    densities = kde(target_np)
    # Convert back to tensor (N, 1)
    return torch.tensor(densities, dtype=target_points.dtype, device=target_points.device).unsqueeze(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run robot policy evaluations using IID or Active Testing."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["iid", "active", "loaded", "brute_force"],
        required=True,
        help="The sampling strategy to use."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=10,
        help="The resolution (N) for the N^D design space in 'brute_force' or 'iid' mode."
    )
    parser.add_argument(
        "--num_evals",
        type=int,
        default=20,
        help="Total number of evaluations to run."
    )
    parser.add_argument(
        "--num_init_pts",
        type=int,
        default=10,
        help="Number of initial random points for bootstrapping the active learning model."
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=35,
        help="Maximum number of steps allowed per evaluation."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_results.csv",
        help="Path to save the eval results as CSV file."
    )
    parser.add_argument(
        "--save_points",
        type=str,
        default=None,
        help="Path to save only the eval points to a CSV file. Note that eval points are already included in eval results."
    )
    parser.add_argument(
        "--load_path",
        type=str,
        default=None,
        help="Path to a CSV file from which to load evaluation points. Could be the full eval results CSV file or just the eval points CSV file."
    )
    parser.add_argument(
        '--model_name',
        type=str,
        help="Surrogate model to use for active testing (e.g. 'FullyBayesianSingleTaskGP')."
    )
    parser.add_argument(
        '--acq_func_name',
        type=str,
        help="Acquisition function to use (e.g. 'qBALD')."
    )
    parser.add_argument(
        '--vla_data_path',
        type=str,
        default=None,
        help="Path to CSV containing VLA training data (factor values) that help compute OOD metrics on eval data."
    )
    parser.add_argument(
        '--ood_metric',
        type=str,
        choices=['knn', 'kde'],
        default='knn',
        help="Metric to use for the OOD feature (distance to nearest neighbor or density)."
    )

    args = parser.parse_args()

    if not args.load_path and args.mode=='loaded':
        parser.error("`load_path` must be specified if loading points")

    if args.num_init_pts >= args.num_evals and args.mode == 'active':
        raise ValueError("`num_init_pts` must be less than `num_evals` for active learning.")
    
    return args