"""
Select next factor values to collect demos for, using either:

1) Certain failures method: use a saved surrogate model to pick N "certain failures"
   (low predicted mean and low variance) from the design space.

2) Observed method: use only actual observed failures from a results CSV,
   sorted by outcome (worst first). No surrogate.

You can optionally fix factors, restrict by one or more (x,y) quadrants (OR), and save outputs to CSV.
"""

import argparse
import os
import pickle

import pandas as pd
import torch

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from factors_config import (
    FACTOR_COLUMNS,
    get_design_points_robot,
    is_valid_point,
)

tkwargs = {"dtype": torch.double, "device": "cpu"}

_VALID_XY_QUADRANTS = frozenset({"bottom_left", "bottom_right", "top_left", "top_right"})


def _xy_quadrant_mask(x, y, quadrant: str, thresh: float = 0.5):
    """
    Element-wise mask for points in one (x, y) quadrant. x and y may be torch.Tensor or ndarray-like.
    Quadrants use x,y in [0,1] with split at thresh (default 0.5).
    """
    q = quadrant.lower()
    if q == "bottom_left":
        return (x <= thresh) & (y <= thresh)
    if q == "bottom_right":
        return (x >= thresh) & (y <= thresh)
    if q == "top_left":
        return (x <= thresh) & (y >= thresh)
    if q == "top_right":
        return (x >= thresh) & (y >= thresh)
    raise ValueError(
        f"Unknown quadrant '{quadrant}'. Expected one of: {', '.join(sorted(_VALID_XY_QUADRANTS))}."
    )


def _find_last_trial_model(models_dir: str):
    """Return path to the highest-numbered trial_XX_model.pkl in models_dir, or None."""
    if not os.path.isdir(models_dir):
        return None
    best_trial = None
    best_path = None
    for name in os.listdir(models_dir):
        if not name.startswith("trial_") or not name.endswith("_model.pkl"):
            continue
        middle = name[len("trial_") : -len("_model.pkl")]
        try:
            trial_idx = int(middle)
        except ValueError:
            continue
        if best_trial is None or trial_idx > best_trial:
            best_trial = trial_idx
            best_path = os.path.join(models_dir, name)
    return best_path


def _load_model_from_results(results_file: str, explicit_model_path: str | None = None):
    """
    Load a saved surrogate model given a results CSV.

    If explicit_model_path is provided, use that. Otherwise, look for:
        dirname(results_file)/models/trial_XX_model.pkl with largest XX.
    """
    model_path = explicit_model_path
    if model_path is None:
        results_dir = os.path.dirname(os.path.abspath(results_file))
        models_dir = os.path.join(results_dir, "models")
        model_path = _find_last_trial_model(models_dir)
    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Could not find a saved model for results file '{results_file}'. "
            f"Looked in '{os.path.join(os.path.dirname(results_file), 'models')}'. "
            f"Optionally pass --model_path explicitly."
        )

    print(f"Loading model from '{model_path}'...")
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    model = data["model"]
    return model


def _filter_points_by_fixed_factors(points: torch.Tensor, fixed_factors: dict[str, list[float]]) -> torch.Tensor:
    """Apply optional factor fixes to the design space (OR within each factor, AND across factors)."""
    mask = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)

    if not fixed_factors:
        return points

    for name, values in fixed_factors.items():
        if name not in FACTOR_COLUMNS:
            raise ValueError(f"FACTOR_COLUMNS does not contain '{name}'; cannot fix this factor.")
        idx = FACTOR_COLUMNS.index(name)
        factor_mask = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
        for value in values:
            target = torch.tensor(value, **tkwargs)
            factor_mask = factor_mask | torch.isclose(points[:, idx], target, atol=1e-6)
        mask = mask & factor_mask

    return points[mask]


def _filter_points_by_xy_quadrant(points: torch.Tensor, quadrants: list[str] | None) -> torch.Tensor:
    """
    Optionally restrict points to one or more (x, y) quadrants (OR).
    Quadrants are defined over x,y in [0,1]:
      - bottom_left:  x <= 0.5, y <= 0.5
      - bottom_right: x >=  0.5, y <= 0.5
      - top_left:     x <= 0.5, y >=  0.5
      - top_right:    x >=  0.5, y >=  0.5
    """
    if not quadrants:
        return points

    if "x" not in FACTOR_COLUMNS or "y" not in FACTOR_COLUMNS:
        raise ValueError("FACTOR_COLUMNS must contain 'x' and 'y' to use --fix_xy_quadrant.")

    x_idx = FACTOR_COLUMNS.index("x")
    y_idx = FACTOR_COLUMNS.index("y")
    x = points[:, x_idx]
    y = points[:, y_idx]

    thresh = 0.5
    mask = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    for q in quadrants:
        mask = mask | _xy_quadrant_mask(x, y, q, thresh)

    return points[mask]


def select_certain_failures(
    results_file: str,
    num_points: int,
    task_name: str | None = None,
    model_path: str | None = None,
    fixed_factors: dict[str, list[float]] | None = None,
    xy_quadrants: list[str] | None = None,
    var_percentile: float = 0.30,
):
    """
    Select N \"certain failures\" from the discrete robot design space.

    - \"Failure\" = low predicted continuous outcome.
    - \"Certain\" = among low-mean points, low predictive variance.

    Ranking is lexicographic on (mean, variance): first sort by mean ascending
    (worst outcomes), then by variance ascending (most certain).
    """
    print("---Selecting certain failures using surrogate---")
    model = _load_model_from_results(results_file, explicit_model_path=model_path)
    model.eval()

    # 1. Build candidate design space
    all_points = get_design_points_robot().to(**tkwargs)
    print(f"Total design points before filtering: {all_points.shape[0]}")

    # 2. Apply task-specific validity and optional fixed factors
    valid_mask = torch.tensor(
        [is_valid_point(p, task_name=task_name) for p in all_points],
        dtype=torch.bool,
        device=all_points.device,
    )
    candidates = all_points[valid_mask]
    print(f"After is_valid_point filtering: {candidates.shape[0]} valid points")

    candidates = _filter_points_by_fixed_factors(candidates, fixed_factors or {})
    candidates = _filter_points_by_xy_quadrant(candidates, xy_quadrants)
    print(f"After factor filtering: {candidates.shape[0]} candidate points")

    if candidates.shape[0] == 0:
        print("No candidate points remain after filtering; nothing to select.")
        return

    # 3. Model predictions
    with torch.no_grad():
        device = next(model.parameters()).device
        X = candidates.to(device)
        posterior = model.posterior(X)
        mean = posterior.mean
        var = posterior.variance

    # Squeeze trailing singleton dims, then reduce Bayesian sample dim if present.
    # For fully Bayesian GPs, posterior mean/var are often [S, N], where S is #samples.
    while mean.ndim > 1 and mean.shape[-1] == 1:
        mean = mean.squeeze(-1)
    while var.ndim > 1 and var.shape[-1] == 1:
        var = var.squeeze(-1)

    if mean.ndim == 2:
        mean = mean.mean(dim=0)
    if var.ndim == 2:
        var = var.mean(dim=0)

    if mean.ndim != 1 or var.ndim != 1:
        raise RuntimeError(
            f"Unexpected posterior shapes after Bayesian reduction: mean {mean.shape}, var {var.shape}"
        )

    mean_cpu = mean.cpu()
    var_cpu = var.cpu()

    # 4. Filter to lowest n% variance (most certain)
    var_threshold = torch.quantile(var_cpu, var_percentile)
    certain_mask = var_cpu <= var_threshold
    certain_indices = torch.where(certain_mask)[0].tolist()
    print(f"After filtering to lowest {var_percentile*100}% variance: {len(certain_indices)} candidate points")
    
    if len(certain_indices) == 0:
        print(f"No candidate points in lowest {var_percentile*100}% variance; nothing to select.")
        return

    # 5. Rank points by mean (within the low-variance set)
    certain_indices.sort(key=lambda i: mean_cpu[i].item())

    k = min(num_points, len(certain_indices))
    if len(certain_indices) < num_points:
        print(f"Only {len(certain_indices)} certain failure candidates available; selecting all of them.")
    selected_idx = [certain_indices[i] for i in range(k)]
    selected_points = candidates[selected_idx].cpu()

    # print(f"\nTop {k} certain failures (sorted by mean, then variance):")
    # header = " | ".join(FACTOR_COLUMNS + ["pred_mean", "pred_variance"])
    # print(header)
    # print("-" * len(header))
    # for i in range(k):
    #     p = selected_points[i]
    #     m = mean_cpu[selected_idx[i]].item()
    #     v = var_cpu[selected_idx[i]].item()
    #     factors_str = ", ".join(f"{name}={p[j].item():.3f}" for j, name in enumerate(FACTOR_COLUMNS))
    #     print(f"{factors_str} | {m:.4f} | {v:.4f}")

    return selected_points, mean_cpu[selected_idx], var_cpu[selected_idx]


def select_observed_failures(
    results_file: str,
    num_points: int,
    task_name: str | None = None,
    fixed_factors: dict[str, list[float]] | None = None,
    xy_quadrants: list[str] | None = None,
):
    """
    Select N actual observed failures from a results CSV (no surrogate).

    Loads the CSV, keeps only rows with binary_outcome == 0 (failure), applies
    optional fixed factors and xy quadrant(s) (OR), then sorts by continuous_outcome
    ascending (worst first) and returns the top num_points.
    """
    print("---Selecting observed failures---")
    df = pd.read_csv(results_file)
    for col in FACTOR_COLUMNS + ["continuous_outcome", "binary_outcome"]:
        if col not in df.columns:
            raise ValueError(f"Results CSV must contain column '{col}'.")

    # Keep only failures (binary_outcome == 0)
    failures = df[df["binary_outcome"] == 0.0].copy()
    if failures.empty:
        print("No observed failures in the results file; nothing to select.")
        return None

    # Optional: filter by fixed factors
    if fixed_factors:
        for name, values in fixed_factors.items():
            if name not in FACTOR_COLUMNS:
                raise ValueError(f"FACTOR_COLUMNS does not contain '{name}'.")
            # OR within a factor: keep rows matching any listed value for this factor
            factor_mask = False
            for value in values:
                factor_mask = factor_mask | (failures[name].sub(value).abs() < 1e-6)
            failures = failures[factor_mask]
        if failures.empty:
            print("No observed failures match the fixed factors; nothing to select.")
            return None

    # Optional: filter by xy quadrant(s) (OR)
    if xy_quadrants and "x" in FACTOR_COLUMNS and "y" in FACTOR_COLUMNS:
        x = failures["x"].values
        y = failures["y"].values
        thresh = 0.5
        mask = _xy_quadrant_mask(x, y, xy_quadrants[0], thresh)
        for q in xy_quadrants[1:]:
            mask = mask | _xy_quadrant_mask(x, y, q, thresh)
        failures = failures.loc[mask]
        if failures.empty:
            print("No observed failures in the chosen quadrant(s); nothing to select.")
            return None

    # Sort by continuous_outcome ascending (worst first)
    failures = failures.sort_values("continuous_outcome", ascending=True).reset_index(drop=True)
    print(f"After filtering: {len(failures)} candidate points")
    if len(failures) < num_points:
        print(f"Only {len(failures)} observed failures available after filtering; selecting all of them.")
    k = min(num_points, len(failures))
    selected = failures.iloc[:k]

    # print(f"\nTop {k} observed failures (sorted by outcome, worst first):")
    # extra_cols = [c for c in ["trial", "continuous_outcome", "binary_outcome", "steps_taken"] if c in selected.columns]
    # display_cols = FACTOR_COLUMNS + extra_cols
    # print(selected[display_cols].to_string(index=False))

    return selected


def _save_certain_failures_points(
    points: torch.Tensor,
    pred_mean: torch.Tensor,
    pred_var: torch.Tensor,
    filepath: str,
) -> None:
    """Save certain failures method output to CSV (factor columns + pred_mean, pred_variance)."""
    rows = []
    for i in range(points.shape[0]):
        row = {name: points[i, j].item() for j, name in enumerate(FACTOR_COLUMNS)}
        row["pred_mean"] = pred_mean[i].item()
        row["pred_variance"] = pred_var[i].item()
        rows.append(row)
    pd.DataFrame(rows).to_csv(filepath, index=False)
    print(f"Saved {len(rows)} points to {filepath}")


def _save_observed_points(selected_df: pd.DataFrame, filepath: str) -> None:
    """Save observed-failures output to CSV (all columns from selected rows)."""
    selected_df.to_csv(filepath, index=False)
    print(f"Saved {len(selected_df)} points to {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Select next demo points from one results file using either certain failures "
        "(surrogate-based) or observed failures."
    )
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to results.csv produced by eval.py or offline_eval.py. Works for active or IID runs.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["certainfail", "observed"],
        default="certainfail",
        help="Selection method: certainfail (model-based) or observed (actual failure rows).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional explicit path to a saved model .pkl (for certainfail method). "
        "If omitted, looks for trial_XX_model.pkl under dirname(results_file)/models/.",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=10,
        help="Number of points to select per method.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Optional task name to pass into is_valid_point for task-specific constraints.",
    )
    parser.add_argument(
        "--fix_factor",
        action="append",
        default=None,
        metavar="NAME=VALUE",
        help=(
            "Optional factor fix; may be given multiple times. "
            "Multiple entries with the same NAME are treated as allowed values (OR), "
            "e.g. --fix_factor table_height=1 --fix_factor table_height=3."
        ),
    )
    parser.add_argument(
        "--fix_xy_quadrant",
        action="append",
        default=None,
        metavar="QUADRANT",
        choices=["bottom_left", "bottom_right", "top_left", "top_right"],
        help=(
            "Optional (x, y) quadrant restriction; may be given multiple times (OR), "
            "e.g. --fix_xy_quadrant bottom_left --fix_xy_quadrant top_right. "
            "bottom_left: x<=0.5,y<=0.5; bottom_right: x>=0.5,y<=0.5; "
            "top_left: x<=0.5,y>=0.5; top_right: x>=0.5,y>=0.5."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="If set, save selected points to CSV here: "
        "next_demos_certain_failures.csv or next_demos_observed_failures.csv.",
    )

    args = parser.parse_args()

    # Parse fixed factors from NAME=VALUE strings into a dict[str, list[float]]
    fixed_factors: dict[str, list[float]] = {}
    if args.fix_factor:
        for spec in args.fix_factor:
            if "=" not in spec:
                raise ValueError(f"Invalid --fix_factor '{spec}'. Expected NAME=VALUE.")
            name, val_str = spec.split("=", 1)
            name = name.strip()
            try:
                value = float(val_str)
            except ValueError:
                raise ValueError(f"Could not parse value '{val_str}' in --fix_factor '{spec}' as float.")
            if name not in fixed_factors:
                fixed_factors[name] = []
            fixed_factors[name].append(value)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    xy_quadrants = args.fix_xy_quadrant
    if xy_quadrants is not None:
        xy_quadrants = list(dict.fromkeys(xy_quadrants))

    if args.method == "certainfail":
        result = select_certain_failures(
            results_file=args.results_file,
            num_points=args.num_points,
            task_name=args.task,
            model_path=args.model_path,
            fixed_factors=fixed_factors,
            xy_quadrants=xy_quadrants,
        )
        if result is not None and args.output_dir:
            points, pred_mean, pred_var = result
            out_path = os.path.join(args.output_dir, "next_demos_certain_failures.csv")
            _save_certain_failures_points(points, pred_mean, pred_var, out_path)

    if args.method == "observed":
        selected_df = select_observed_failures(
            results_file=args.results_file,
            num_points=args.num_points,
            task_name=args.task,
            fixed_factors=fixed_factors,
            xy_quadrants=xy_quadrants,
        )
        if selected_df is not None and args.output_dir:
            out_path = os.path.join(args.output_dir, "next_demos_observed_failures.csv")
            _save_observed_points(selected_df, out_path)


if __name__ == "__main__":
    main()