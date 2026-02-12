"""
Offline sampling script for existing evaluation results.

Given a CSV of previously collected results (e.g. a brute-force sweep)
containing factor columns and outcomes, this script simulates new
evaluation runs using either:

- Active testing (surrogate + acquisition function), or
- IID testing (uniform random sampling),

but instead of querying a real robot, it *looks up* outcomes from the
provided results file.

The CLI flags mirror those in `eval.py` / `utils.parse_args` wherever
the semantics are the same:
- `--mode`          : "active" or "iid"
- `--num_evals`     : total number of (offline) evaluations
- `--num_init_pts`  : number of initial random points (only for active)
- `--eval_id`       : evaluation ID; results saved to `results/{eval_id}/`
- `--output_file`   : override default `results/{eval_id}/results.csv`
- `--load_path`     : path to the *source* results CSV to sample from
- `--task`          : task name (for consistency / logging)

Example:
    uv run offline_eval.py \\
        --mode active \\
        --num_evals 40 \\
        --num_init_pts 10 \\
        --load_path results/uprightcup_bruteforce/results.csv \\
        --task uprightcup \\
        --eval_id uprightcup_active_offline
"""

import os
import uuid
from datetime import datetime

import pandas as pd
import torch

from utils import parse_args, fit_surrogate_model
from testers import ActiveTester, IIDSampler
from factors_config import FACTOR_COLUMNS, BOUNDS, tkwargs


def _ensure_eval_id_and_paths(args):
    """Set up eval_id and results/{eval_id}/ output paths (similar to eval.py)."""
    # If eval_id not provided, auto-generate from timestamp
    if args.eval_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.eval_id = f"offline_{timestamp}_{uuid.uuid4().hex[:8]}"

    results_base_dir = "results"
    eval_dir = os.path.join(results_base_dir, args.eval_id)
    models_dir = os.path.join(eval_dir, "models")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Output file
    if args.output_file is None:
        args.output_file = os.path.join(eval_dir, "results.csv")
    else:
        # If user provided a relative filename, place it under eval_dir
        if not os.path.isabs(args.output_file) and not os.path.dirname(args.output_file):
            args.output_file = os.path.join(eval_dir, args.output_file)

    return eval_dir, models_dir


def _load_source_results(load_path):
    """Load the source results CSV to sample from."""
    if not load_path:
        raise ValueError(
            "--load_path must be specified and point to a results CSV "
            "to use as the offline ground-truth pool."
        )

    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Source results file '{load_path}' not found.")

    df = pd.read_csv(load_path)
    if df.empty:
        raise ValueError(f"Source results file '{load_path}' is empty.")

    # Ensure the factor columns exist
    if not set(FACTOR_COLUMNS).issubset(df.columns):
        raise ValueError(
            f"Source results file must contain factor columns {FACTOR_COLUMNS}, "
            f"but has columns {list(df.columns)}."
        )

    required_cols = {"continuous_outcome", "binary_outcome", "steps_taken"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Source results file must contain columns {sorted(required_cols)}, "
            f"but has columns {list(df.columns)}."
        )

    return df


def _build_design_tensors(df):
    """
    From a results DataFrame, build tensors for factors and outcomes.

    Returns:
        X_all : [N, D] tensor of factor values
        Y_cont: [N, 1] tensor of continuous outcomes
        Y_bin : [N, 1] tensor of binary outcomes
        steps : [N]    array of steps_taken (as numpy array)
    """
    X_all = torch.tensor(df[FACTOR_COLUMNS].values, **tkwargs)
    Y_cont = torch.tensor(df[["continuous_outcome"]].values, **tkwargs)
    Y_bin = torch.tensor(df[["binary_outcome"]].values, **tkwargs)
    steps = df["steps_taken"].values
    return X_all, Y_cont, Y_bin, steps


def _find_index_in_tensor_rows(X, x):
    """
    Find the index of row x in tensor X by exact comparison.

    Assumes x is one of the rows in X (as constructed from the same
    underlying discrete factor values).
    """
    matches = torch.all(X == x.unsqueeze(0), dim=1)
    idxs = torch.nonzero(matches, as_tuple=False)
    if idxs.numel() == 0:
        raise RuntimeError("Point not found in candidate tensor.")
    return idxs[0, 0].item()


def run_offline_iid(args, df_source, models_dir):
    """Offline IID sampling from a source results CSV (with per-trial models, like eval.py)."""
    X_all, Y_cont, Y_bin, steps = _build_design_tensors(df_source)
    N = X_all.shape[0]

    print(f"Offline IID sampling from {N} candidate points in '{args.load_path}'.")
    print(f"Writing results to '{args.output_file}'.")

    sampler = IIDSampler(X_all)
    results_data = []
    train_X_list = []
    train_Y_list = []

    for i in range(args.num_evals):
        x_next = sampler.get_next_point()
        idx = _find_index_in_tensor_rows(X_all, x_next)

        entry = {
            "trial": i + 1,
            "mode": "iid",
            "binary_outcome": float(Y_bin[idx].item()),
            "continuous_outcome": float(Y_cont[idx].item()),
            "steps_taken": int(steps[idx]),
        }
        for d, col_name in enumerate(FACTOR_COLUMNS):
            entry[col_name] = float(x_next[d].item())

        results_data.append(entry)

        # --- Fit and save model at this trial (mirrors eval.py IID branch) ---
        if hasattr(args, "model_name") and args.model_name is not None:
            import pickle

            # Append current point/outcome to training data
            train_X_list.append(x_next)
            train_Y_list.append(torch.tensor([entry["continuous_outcome"]], **tkwargs))

            model_to_save = None
            if len(train_X_list) >= 2:
                torch.manual_seed(i + 1)  # match eval.py seeding style
                train_X_to_save = torch.stack(train_X_list)
                train_Y_to_save = torch.stack(train_Y_list)
                model_to_save = fit_surrogate_model(
                    train_X_to_save, train_Y_to_save, BOUNDS, model_name=args.model_name
                )

            if model_to_save is not None:
                model_save_path = os.path.join(models_dir, f"trial_{i+1}_model.pkl")
                with open(model_save_path, "wb") as f:
                    pickle.dump(
                        {
                            "model": model_to_save,
                            "model_name": args.model_name,
                            "acq_func_name": None,
                            "train_X": train_X_to_save,
                            "train_Y": train_Y_to_save,
                            "bounds": BOUNDS,
                            "task_name": args.task,
                            "training_data_factors_path": args.training_data_factors_path,
                            "ood_metric": args.ood_metric,
                            "use_train_data_for_surrogate": args.use_train_data_for_surrogate,
                            "trial": i + 1,
                            "mode": "iid",
                        },
                        f,
                    )

    results_df = pd.DataFrame(results_data)
    results_df.to_csv(args.output_file, index=False)
    print("Offline IID sampling complete.")


def run_offline_active(args, df_source, models_dir):
    """Offline active sampling from a source results CSV (with per-trial + final models)."""
    if args.num_init_pts is None or args.num_init_pts <= 0:
        raise ValueError("For active mode, --num_init_pts must be a positive integer.")
    if args.num_init_pts >= args.num_evals:
        raise ValueError("`num_init_pts` must be less than `num_evals` for active mode.")

    X_all, Y_cont, Y_bin, steps = _build_design_tensors(df_source)
    N = X_all.shape[0]

    print(f"Offline ACTIVE sampling from {N} candidate points in '{args.load_path}'.")
    print(f"Total offline evals: {args.num_evals} "
          f"(initial_random={args.num_init_pts}, active={args.num_evals - args.num_init_pts}).")
    print(f"Writing results to '{args.output_file}'.")

    # --- Choose initial random points without replacement ---
    perm = torch.randperm(N)
    init_idx = perm[: args.num_init_pts]
    remaining_idx = perm[args.num_init_pts :]

    initial_X = X_all[init_idx]
    initial_Y = Y_cont[init_idx]

    # Design space for subsequent active sampling (exclude initial points)
    design_space = X_all[remaining_idx]

    # --- Log initial_random points ---
    results_data = []
    for j, idx in enumerate(init_idx.tolist()):
        x0 = X_all[idx]
        entry = {
            "trial": len(results_data) + 1,
            "mode": "initial_random",
            "binary_outcome": float(Y_bin[idx].item()),
            "continuous_outcome": float(Y_cont[idx].item()),
            "steps_taken": int(steps[idx]),
        }
        for d, col_name in enumerate(FACTOR_COLUMNS):
            entry[col_name] = float(x0[d].item())
        results_data.append(entry)

    # --- Instantiate ActiveTester with initial data and remaining design space ---
    active_tester = ActiveTester(
        initial_X=initial_X,
        initial_Y=initial_Y,
        bounds=BOUNDS,
        full_design_space=design_space,
        mc_points=None,
        model_name=args.model_name,
        acq_func_name=args.acq_func_name,
        training_data_factors_path=args.training_data_factors_path,
        ood_metric=args.ood_metric,
        use_train_data_for_surrogate=args.use_train_data_for_surrogate,
        task_name=args.task,
    )

    # For mapping acquired points back to original df rows
    design_indices = remaining_idx.clone()  # each row in `design_space` corresponds to df row index in X_all

    # --- Active sampling loop ---
    num_active_trials = args.num_evals - args.num_init_pts
    for k in range(num_active_trials):
        x_next = active_tester.get_next_point()  # [D]

        # Find which row in `design_space` this corresponds to
        pos = _find_index_in_tensor_rows(design_space, x_next)
        idx = int(design_indices[pos].item())  # original df row index

        entry = {
            "trial": len(results_data) + 1,
            "mode": "active",
            "binary_outcome": float(Y_bin[idx].item()),
            "continuous_outcome": float(Y_cont[idx].item()),
            "steps_taken": int(steps[idx]),
        }
        for d, col_name in enumerate(FACTOR_COLUMNS):
            entry[col_name] = float(x_next[d].item())
        results_data.append(entry)

        # --- Update the active tester with the new observation (CRITICAL: was missing!) ---
        y_next = torch.tensor([entry["continuous_outcome"]], **tkwargs)
        active_tester.update(x_next, y_next)

        # --- Save model at this trial (mirrors eval.py active branch) ---
        if hasattr(args, "model_name") and args.model_name is not None:
            import pickle

            if hasattr(active_tester, "model") and active_tester.model is not None:
                model_to_save = active_tester.model
                train_X_to_save = active_tester.train_X
                train_Y_to_save = active_tester.train_Y

                model_save_path = os.path.join(models_dir, f"trial_{len(results_data)}_model.pkl")
                with open(model_save_path, "wb") as f:
                    pickle.dump(
                        {
                            "model": model_to_save,
                            "model_name": args.model_name,
                            "acq_func_name": args.acq_func_name,
                            "train_X": train_X_to_save,
                            "train_Y": train_Y_to_save,
                            "bounds": BOUNDS,
                            "task_name": args.task,
                            "training_data_factors_path": args.training_data_factors_path,
                            "ood_metric": args.ood_metric,
                            "use_train_data_for_surrogate": args.use_train_data_for_surrogate,
                            "trial": len(results_data),
                            "mode": "active",
                        },
                        f,
                    )

    results_df = pd.DataFrame(results_data)
    results_df.to_csv(args.output_file, index=False)
    print("Offline ACTIVE sampling complete.")

    # --- Save final model (like eval.py active final_model.pkl) ---
    if hasattr(args, "model_name") and args.model_name is not None:
        if hasattr(active_tester, "model") and active_tester.model is not None:
            import pickle

            model_save_path = os.path.join(models_dir, "final_model.pkl")
            with open(model_save_path, "wb") as f:
                pickle.dump(
                    {
                        "model": active_tester.model,
                        "model_name": args.model_name,
                        "acq_func_name": args.acq_func_name,
                        "train_X": active_tester.train_X,
                        "train_Y": active_tester.train_Y,
                        "bounds": BOUNDS,
                        "task_name": args.task,
                        "training_data_factors_path": args.training_data_factors_path,
                        "ood_metric": args.ood_metric,
                        "use_train_data_for_surrogate": args.use_train_data_for_surrogate,
                    },
                    f,
                )
            print(f"💾 Final offline active model saved to '{model_save_path}'.")


def main():
    args = parse_args()

    if args.mode not in ["iid", "active"]:
        raise ValueError(
            "offline_eval.py only supports --mode 'iid' or 'active'. "
            "For 'loaded' or 'brute_force', use eval.py instead."
        )

    eval_dir, models_dir = _ensure_eval_id_and_paths(args)
    print(f"📁 Offline evaluation ID: {args.eval_id}")
    print(f"📁 Results directory: {eval_dir}")

    df_source = _load_source_results(args.load_path)

    if args.mode == "iid":
        run_offline_iid(args, df_source, models_dir)
    else:
        run_offline_active(args, df_source, models_dir)


if __name__ == "__main__":
    main()

