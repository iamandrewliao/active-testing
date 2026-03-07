"""
Live plot of RMSE and log-likelihood vs trial for an ongoing eval.py run (active or IID).

Updates only when new data is available: watches the results file and recomputes
when it changes (i.e. after each new trial). Run automatically by eval.py when
--live_plot is set, or run manually with the same paths.

On a headless server, use --save_path to write the plot to an image file each time
new data is available (e.g. results/<eval_id>/live_plot.png); refresh the file to view.

Usage (manual):
  uv run live_plot_eval.py --results_file results/<eval_id>/results.csv \\
    --gt_results_file results/<task>_bruteforce/results.csv --task <task> --model_name SingleTaskGP

Headless (save image every trial):
  uv run live_plot_eval.py ... --save_path results/<eval_id>/live_plot.png
"""

import argparse
import os
import pickle
import sys
import time

# Use non-interactive backend when saving to file (for headless servers)
if "--save_path" in sys.argv:
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import torch

from factors_config import BOUNDS, FACTOR_COLUMNS
from utils import fit_surrogate_model, calculate_rmse, calculate_log_likelihood

tkwargs = {"dtype": torch.double, "device": "cpu"}


def _load_data_safe(filepath):
    """Load CSV if it exists and is readable; return None otherwise."""
    if not filepath or not os.path.exists(filepath):
        return None
    try:
        return pd.read_csv(filepath, dtype={"mode": str})
    except Exception:
        return None


def _get_tensors_from_df(df):
    train_X = torch.tensor(df[FACTOR_COLUMNS].values, **tkwargs)
    train_Y = torch.tensor(df["continuous_outcome"].values, **tkwargs).unsqueeze(-1)
    return train_X, train_Y


def compute_metrics_so_far(results_file, models_dir, gt_X, gt_Y, model_name):
    """
    For the current results CSV, compute RMSE and LL at each trial using saved
    models (or retrain if missing). Return (trials_list, rmse_list, ll_list).
    """
    df = _load_data_safe(results_file)
    if df is None or len(df) == 0:
        return [], [], []

    trials_list = []
    rmse_list = []
    ll_list = []

    for trial_num in range(1, len(df) + 1):
        model = None
        if models_dir and os.path.isdir(models_dir):
            pkl_path = os.path.join(models_dir, f"trial_{trial_num}_model.pkl")
            if os.path.exists(pkl_path):
                try:
                    with open(pkl_path, "rb") as f:
                        model = pickle.load(f)["model"]
                except Exception:
                    pass

        if model is None:
            # Fallback: retrain on data up to this trial
            current_df = df.iloc[:trial_num]
            if len(current_df) < 2:
                continue
            train_X, train_Y = _get_tensors_from_df(current_df)
            torch.manual_seed(trial_num)
            model = fit_surrogate_model(train_X, train_Y, BOUNDS, model_name=model_name)

        try:
            rmse = calculate_rmse(model, gt_X, gt_Y)
            ll = calculate_log_likelihood(model, gt_X, gt_Y)
        except Exception:
            continue

        trials_list.append(trial_num)
        rmse_list.append(rmse)
        ll_list.append(ll)

    return trials_list, rmse_list, ll_list


def main():
    parser = argparse.ArgumentParser(
        description="Live plot RMSE and log-likelihood vs trial for an ongoing eval run."
    )
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to the current run's results.csv (e.g. results/<eval_id>/results.csv).",
    )
    parser.add_argument(
        "--gt_results_file",
        type=str,
        required=True,
        help="Path to brute-force ground truth CSV for computing RMSE/LL.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task name (for display).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="SingleTaskGP",
        help="Surrogate model name (for fallback retrain when a trial model is missing).",
    )
    parser.add_argument(
        "--check_interval",
        type=float,
        default=0.5,
        help="Seconds between checks for new data (default: 0.5). Plot only redraws when results file changes.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="If set, run headless and save the plot to this path whenever new data is available (for remote servers).",
    )
    args = parser.parse_args()

    # Load ground truth once
    gt_df = _load_data_safe(args.gt_results_file)
    if gt_df is None or len(gt_df) == 0:
        raise FileNotFoundError(
            f"Cannot load ground truth from '{args.gt_results_file}'. "
            "Ensure the file exists and is readable."
        )
    gt_X, gt_Y = _get_tensors_from_df(gt_df)
    models_dir = os.path.join(os.path.dirname(os.path.abspath(args.results_file)), "models")

    # Figure: two subplots
    fig, (ax_rmse, ax_ll) = plt.subplots(1, 2, figsize=(14, 5))
    (line_rmse,) = ax_rmse.plot([], [], "b-o", markersize=4, linewidth=1.5, label="Current run")
    (line_ll,) = ax_ll.plot([], [], "b-o", markersize=4, linewidth=1.5, label="Current run")

    ax_rmse.set_xlabel("Trial", fontsize=12)
    ax_rmse.set_ylabel("RMSE on GT", fontsize=12)
    ax_rmse.set_title("RMSE vs. Trials (live)", fontsize=14, fontweight="bold")
    ax_rmse.legend(loc="upper right")
    ax_rmse.grid(True, alpha=0.3)

    ax_ll.set_xlabel("Trial", fontsize=12)
    ax_ll.set_ylabel("Log-Likelihood on GT", fontsize=12)
    ax_ll.set_title("Log-Likelihood vs. Trials (live)", fontsize=14, fontweight="bold")
    ax_ll.legend(loc="lower right")
    ax_ll.grid(True, alpha=0.3)

    title = "Live metrics vs brute-force GT"
    if args.task:
        title += f" — {args.task}"
    fig.suptitle(title, fontsize=14, y=1.02)

    # Only recompute when results file (or model count) has changed
    last_mtime = [0.0]  # mutable so closure can update

    def init():
        ax_rmse.set_xlim(0, 1)
        ax_rmse.set_ylim(0, 1)
        ax_ll.set_xlim(0, 1)
        ax_ll.set_ylim(-5, 1)
        return line_rmse, line_ll

    def update(_frame):
        try:
            st = os.stat(args.results_file)
        except OSError:
            return line_rmse, line_ll
        if st.st_mtime <= last_mtime[0]:
            return line_rmse, line_ll
        last_mtime[0] = st.st_mtime

        trials, rmse, ll = compute_metrics_so_far(
            args.results_file,
            models_dir,
            gt_X,
            gt_Y,
            args.model_name,
        )
        if not trials:
            return line_rmse, line_ll

        line_rmse.set_data(trials, rmse)
        line_ll.set_data(trials, ll)

        ax_rmse.set_xlim(0, max(trials) + 1)
        rmin, rmax = min(rmse), max(rmse)
        margin = (rmax - rmin) * 0.1 or 0.1
        ax_rmse.set_ylim(rmin - margin, rmax + margin)
        ax_ll.set_xlim(0, max(trials) + 1)
        lmin, lmax = min(ll), max(ll)
        margin = (lmax - lmin) * 0.1 or 0.1
        ax_ll.set_ylim(lmin - margin, lmax + margin)

        return line_rmse, line_ll

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if args.save_path:
        print('SAVE PATH')
        # Headless: poll for new data and save image (no window)
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)) or ".", exist_ok=True)
        init()
        last_mtime = 0.0
        print(f"Headless mode: saving plot to {args.save_path} when new data is available (Ctrl+C to stop).")
        try:
            while True:
                time.sleep(args.check_interval)
                try:
                    st = os.stat(args.results_file)
                except OSError:
                    continue
                if st.st_mtime <= last_mtime:
                    continue
                last_mtime = st.st_mtime
                trials, rmse, ll = compute_metrics_so_far(
                    args.results_file,
                    models_dir,
                    gt_X,
                    gt_Y,
                    args.model_name,
                )
                if not trials:
                    continue
                line_rmse.set_data(trials, rmse)
                line_ll.set_data(trials, ll)
                ax_rmse.set_xlim(0, max(trials) + 1)
                rmin, rmax = min(rmse), max(rmse)
                margin = (rmax - rmin) * 0.1 or 0.1
                ax_rmse.set_ylim(rmin - margin, rmax + margin)
                ax_ll.set_xlim(0, max(trials) + 1)
                lmin, lmax = min(ll), max(ll)
                margin = (lmax - lmin) * 0.1 or 0.1
                ax_ll.set_ylim(lmin - margin, lmax + margin)
                fig.savefig(args.save_path, bbox_inches="tight", dpi=150)
                print(f"Saved live plot ({len(trials)} trials) -> {args.save_path}")
        except KeyboardInterrupt:
            print("Live plot save stopped.")
    else:
        ani = animation.FuncAnimation(
            fig,
            update,
            init_func=init,
            interval=args.check_interval * 1000,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()


if __name__ == "__main__":
    main()
