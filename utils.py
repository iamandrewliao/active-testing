import torch
import argparse
import pandas as pd

# Import factor configuration
from factors_config import (
    get_viewpoint_name,
    FACTOR_COLUMNS,
    VIEWPOINT_REPRESENTATION,
    get_viewpoint_index_from_params,
)

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

from MDN import MDN, MDNWrapper, train_mdn
from DeepEnsemble import MLP, DeepEnsembleWrapper, train_ensemble
from BALD import BALD

def fit_surrogate_model(train_X, train_Y, bounds, model_name="SingleTaskGP", use_mc_dropout=None):
    """
    Fits and returns surrogate model.
    
    Args:
        train_X: Training inputs
        train_Y: Training outputs
        bounds: Input bounds for normalization
        model_name: Name of the model to use
        use_mc_dropout: For MDN, whether to use MC dropout (None = True for MDN, False otherwise)
    """
    # print(f"Fitting {model_name} model with {train_X.shape[0]} points...")
    input_transform = Normalize(d=train_X.shape[-1], bounds=bounds) # normalizes X to unit cube [0, 1]^d
    outcome_transform = Standardize(m=1) # standardizes Y to have zero mean and unit variance
    if model_name=="SingleTaskGP" or model_name == "I-BNN" or model_name == "IBNN": # infinite-width BNN is just a GP with a special kernel
        if model_name == "SingleTaskGP":
            kernel = None
        elif model_name == "I-BNN" or model_name == "IBNN":
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
    elif model_name == "MDN" or model_name == "DeepEnsemble":
        # Calculate statistics
        y_mean = train_Y.mean()
        y_std = train_Y.std()
        if y_std < 1e-6: # Avoid division by zero if all Y are identical
            y_std = torch.tensor(1.0, device=train_Y.device, dtype=train_Y.dtype)
        outcome_stats = (y_mean, y_std)
        train_Y_norm = (train_Y - y_mean) / y_std
        if model_name == "MDN":
            input_dim = train_X.shape[-1]
            raw_model = MDN(input_dim=input_dim, hidden_dim=32, n_components=3, n_hidden_layers=5, dropout_prob=0.1).to(device=train_X.device, dtype=train_X.dtype)
            train_mdn(raw_model, train_X, train_Y_norm, bounds, epochs=500)
            # MC dropout: default to True (paper method), but can be overridden
            if use_mc_dropout is None:
                use_mc_dropout = True  # Default: use MC dropout as in paper
            model = MDNWrapper(raw_model, bounds, outcome_stats=outcome_stats, 
                             use_mc_dropout=use_mc_dropout, num_mc_samples=10)
        elif model_name == "DeepEnsemble":
            models = [MLP(input_dim=train_X.shape[-1], hidden_dim=32, n_hidden_layers=5, dropout_prob=0.1).to(device=train_X.device, dtype=train_X.dtype) for _ in range(5)]
            train_ensemble(models, train_X, train_Y_norm, bounds, epochs=200)
            model = DeepEnsembleWrapper(models, bounds, outcome_stats=outcome_stats)        
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
    elif acq_func_name == "BALD":
        acq_func = BALD(model=model)
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
    # Move inputs to model's device
    device = next(model.parameters()).device
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)
    posterior = model.posterior(X_test)
    mean = posterior.mean
    
    # Handle shape: could be (N,), (N, 1), (N, q, 1), (S, N, q, 1) etc.
    # Squeeze all trailing singleton dimensions to get (N,) or (S, N)
    while mean.ndim > 1 and mean.shape[-1] == 1:
        mean = mean.squeeze(-1)
    
    if mean.ndim == 2: # If it's Bayesian (S, N)
        mean = mean.mean(dim=0) # Average over samples to get [N]
    
    # Ensure Y_test is 1D for comparison
    if Y_test.ndim > 1:
        Y_test = Y_test.squeeze()
    
    return torch.sqrt(torch.mean((mean - Y_test)**2)).item()


@torch.no_grad()
def calculate_log_likelihood(model, X_test, Y_test):
    """Calculates the Mean Log-Likelihood (Log Predictive Density) on the test set."""
    model.eval()
    # Move inputs to model's device
    device = next(model.parameters()).device
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)
    posterior = model.posterior(X_test)
    pred_dist = posterior.distribution
    
    # Ensure Y_test has correct shape for log_prob
    # For distributions with batch_shape (N,), Y_test should be (N,)
    # For distributions with batch_shape (N, q), Y_test should be (N, q) or broadcastable
    Y_test_flat = Y_test.squeeze() if Y_test.ndim > 1 else Y_test
    
    # Check if the distribution is a GPyTorch MultivariateNormal
    # GPs need this special handling to avoid O(N^3) inversion crashes on large test sets
    is_gp = hasattr(pred_dist, "lazy_covariance_matrix") or hasattr(pred_dist, "covariance_matrix")
    if is_gp:
        # Construct independent normals using marginal mean and variance
        mean = posterior.mean
        var = posterior.variance
        # Squeeze dimensions to match Y_test
        while mean.ndim > 1 and mean.shape[-1] == 1:
            mean = mean.squeeze(-1)
        while var.ndim > 1 and var.shape[-1] == 1:
            var = var.squeeze(-1)
        
        # Clip variance to prevent numerical instability (very small variances lead to extreme log-likelihoods)
        # Minimum variance threshold: 1e-6 (in original outcome space)
        # This prevents log-probability from becoming extremely negative when variance is near zero
        min_var = torch.tensor(1e-6, device=var.device, dtype=var.dtype)
        var = torch.clamp(var, min=min_var)
        
        # Handle Bayesian case
        if mean.ndim == 2:  # (S, N)
            # For each sample, compute log_prob and average
            log_probs_list = []
            for s in range(mean.shape[0]):
                marginal_dist = torch.distributions.Normal(mean[s], var[s].sqrt())
                log_probs_list.append(marginal_dist.log_prob(Y_test_flat))
            log_probs = torch.stack(log_probs_list, dim=0)  # (S, N)
        else:
            marginal_dist = torch.distributions.Normal(mean, var.sqrt())
            log_probs = marginal_dist.log_prob(Y_test_flat)
    else:
        # MDN and DeepEnsemble use MixtureSameFamily, which supports batch log_prob 
        # Correctly handles multi-modal distributions
        # MixtureSameFamily.log_prob expects input with shape matching batch_shape
        # Distribution has batch_shape (N,), so Y_test should be (N,)
        log_probs = pred_dist.log_prob(Y_test_flat)

    if log_probs.ndim == 2: # Shape [S, N] (for FullyBayesian models)
        num_samples = log_probs.shape[0]
        # Average in log space: log(1/S * sum(exp(log_p)))
        mean_log_pred_density = torch.logsumexp(log_probs, dim=0).mean() - torch.log(torch.tensor(num_samples, device=log_probs.device, dtype=log_probs.dtype))
        return mean_log_pred_density.item()
    else: # Shape [N]
        return log_probs.mean().item()


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



def get_design_points_test(resolution, bounds, tkwargs):
    """
    Creates a tensor of all points for arbitrary D dimensions.
    bounds: [2, D] tensor
    
    NOTE: This function is for test_active.py (test functions).
    For robot evaluation, use get_design_points_robot() from factors_config.py instead.
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



def run_evaluation(point, max_steps, task_name=None, extra_factors=None):
    """
    Simulates a robot evaluation trial for a given point (a combination of factor values).
    Prompts the user for the outcome.

    Args:
        point: Factor values for the evaluation
        max_steps: Maximum number of steps allowed
        task_name: Name of the task (uses default if None)
        extra_factors: Optional dict of task-specific non-design factors (e.g. for
            putgreeninpot: {'lid_x', 'lid_y', 'lid_on'}). Printed so the evaluator can set up the scene.
    """
    from factors_config import (
        DIMS,
        FACTOR_COLUMNS,
        VIEWPOINT_REPRESENTATION,
        get_outcome_range,
        get_success_outcome,
        get_outcome_descriptions,
    )

    # Create a string representation of the point using factor names from config
    if point.shape[0] == DIMS and DIMS == len(FACTOR_COLUMNS):
        # factors used in this work; change if needed
        x = point[0].item()
        y = point[1].item()
        table_height = point[2].item()

        print("-" * 30)
        print(f"🤖 Running trial:")
        print(f"   Object position (block): ({x:.1f}, {y:.1f})")
        print(f"   Table height: {table_height:.0f} inches")

        if VIEWPOINT_REPRESENTATION == "index":
            # Last dimension is the discrete viewpoint index {0,1,2}
            viewpoint_idx = int(point[3].item())
            viewpoint_name = get_viewpoint_name(viewpoint_idx)
            print(f"   Camera viewpoint: {viewpoint_name} (index {viewpoint_idx})")
        elif VIEWPOINT_REPRESENTATION == "params":
            # Last three dimensions are (azimuth, elevation, distance)
            cam_az = point[3].item()
            cam_el = point[4].item()
            cam_dist = point[5].item()
            viewpoint_idx = get_viewpoint_index_from_params(cam_az, cam_el, cam_dist)
            if viewpoint_idx is not None:
                viewpoint_name = get_viewpoint_name(viewpoint_idx)
                print(
                    f"   Camera viewpoint: {viewpoint_name} "
                    f"(index {viewpoint_idx}, az={cam_az:.1f}, el={cam_el:.1f}, dist={cam_dist:.1f})"
                )
            else:
                # Fallback: parameters don't match any known viewpoint (should not happen if is_valid_point is used)
                print(
                    f"   Camera viewpoint: unknown "
                    f"(az={cam_az:.1f}, el={cam_el:.1f}, dist={cam_dist:.1f})"
                )

        # Task-specific extra factors (e.g. putgreeninpot: lid position for scene setup)
        if extra_factors:
            lid_x = extra_factors.get("lid_x")
            lid_y = extra_factors.get("lid_y")
            lid_on = extra_factors.get("lid_on")
            if lid_x is not None and lid_y is not None and lid_on is not None:
                if lid_on:
                    print(f"   Lid: on pot (0.5, 0.5)")
                else:
                    print(f"   Lid position: ({lid_x:.1f}, {lid_y:.1f})")
        elif task_name == "putgreeninpot":
            # Lid is always on in current task setup
            print(f"   Lid: on pot (0.5, 0.5)")
    else:
        raise ValueError(f"Point dimension {point.shape[0]} does not match expected DIMS={DIMS}")

    # Get task-specific outcome configuration
    min_outcome, max_outcome, increment = get_outcome_range(task_name)
    success_outcome = get_success_outcome(task_name)
    descriptions = get_outcome_descriptions(task_name)
    
    # --- Get continuous outcome ---
    valid_outcomes = [min_outcome + i * increment for i in range(int((max_outcome - min_outcome) / increment) + 1)]
    
    while True:
        try:
            print(f"Enter continuous outcome ({increment} increments):")
            for outcome_val in valid_outcomes:
                desc = descriptions.get(outcome_val, "")
                print(f"  {outcome_val}={desc}")
            continuous_outcome = float(input(f"Enter outcome ({min_outcome}-{max_outcome}): "))
            
            if continuous_outcome in valid_outcomes:
                break
            else:
                print(f"Invalid input. Please enter a number in {valid_outcomes}")
        except ValueError:
            print("Invalid input. Please enter a number.")
            
    # --- Derive binary outcome ---
    binary_outcome = 1.0 if continuous_outcome == success_outcome else 0.0

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


def load_training_data(training_data_factors_path):
    """
    Loads training data from a CSV file.
    Returns a (N, D) tensor.
    """
    if training_data_factors_path is None:
        return None
    print(f"Loading training distribution from {training_data_factors_path}...")
    try:
        df = pd.read_csv(training_data_factors_path)
        
        # Check for factor columns from config
        if set(FACTOR_COLUMNS).issubset(df.columns):
            factor_cols = FACTOR_COLUMNS
        else:
            raise ValueError(f"CSV must have {FACTOR_COLUMNS} columns.")

        training_tensor = torch.tensor(df[factor_cols].values, dtype=torch.double)
        return training_tensor
    except Exception as e:
        print(f"Error loading training data: {e}")
        return None


def compute_knn_distance(target_points, reference_points, k=1):
    """
    Computes the distance from each point in target_points to its 
    k-th nearest neighbor in reference_points.
    
    Args:
        target_points: (N, 2) tensor of points to evaluate
        reference_points: (M, D) tensor of training data
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
#         reference_points: (M, D) tensor of training data
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
        help="The sampling strategy to use."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=10,
        help="The resolution (N) for a continuous N^D design space in 'brute_force' or 'iid' mode."
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
        help="Maximum number of steps allowed per evaluation."
    )
    parser.add_argument(
        "--eval_id",
        type=str,
        default=None,
        help="Evaluation ID. If not provided, will be auto-generated from timestamp. All outputs will be saved in results/{eval_id}/"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save the eval results as CSV file. If not provided, will be saved as results/{eval_id}/results.csv"
    )
    parser.add_argument(
        "--save_points",
        type=str,
        default=None,
        help="Path to save only the eval points to a CSV file. If not provided and --eval_id is set, will be saved as results/{eval_id}/points.csv. Note that eval points are already included in eval results."
    )
    parser.add_argument(
        "--load_path",
        type=str,
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
        '--training_data_factors_path',
        type=str,
        help="Path to CSV containing robot policy training data (factor values) that help compute OOD metrics on eval data."
    )
    parser.add_argument(
        '--task',
        type=str,
        help="Task name for outcome configuration. If not specified, uses default task from factors_config.py."
    )
    parser.add_argument(
        '--ood_metric',
        type=str,
        choices=['knn', 'kde'],
        help="Metric to use for the OOD feature (distance to nearest neighbor or density)."
    )
    parser.add_argument(
        '--use_train_data_for_surrogate',
        action='store_true',
        help="If True, adds robot policy training points to the surrogate model training set."
    )

    args = parser.parse_args()

    if not args.load_path and args.mode=='loaded':
        parser.error("`load_path` must be specified if loading points")

    if args.num_init_pts >= args.num_evals and args.mode == 'active':
        raise ValueError("`num_init_pts` must be less than `num_evals` for active learning.")
    
    return args