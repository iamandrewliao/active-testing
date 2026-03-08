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

from factors_config import BOUNDS, FACTOR_COLUMNS, get_design_points_robot, is_valid_point
from utils import fit_surrogate_model, calculate_rmse, calculate_log_likelihood

tkwargs = {"dtype": torch.double, "device": "cpu"}


def _filter_gt_to_design_space(gt_df, task_name=None):
    """
    Filter ground truth to only rows whose factor combination is in the current
    design space (same as eval.py: get_design_points_robot + is_valid_point).
    When factors_config is limited (e.g. VIEWPOINT_VALUES = [0.0]), metrics
    are then computed on the same subset the eval is using.
    """
    all_points = get_design_points_robot()
    valid_points = torch.stack(
        [p for p in all_points if is_valid_point(p, task_name=task_name)]
    )
    if valid_points.shape[0] == 0:
        raise ValueError(
            "No valid design points in current factors_config (e.g. after filtering by task). "
            "Cannot filter ground truth."
        )
    gt_tensor = torch.tensor(gt_df[FACTOR_COLUMNS].values, **tkwargs)
    valid_points_cpu = valid_points.to(gt_tensor.device)
    # Match each GT row to any valid point (with tolerance for floats)
    match = torch.isclose(
        gt_tensor.unsqueeze(1), valid_points_cpu.unsqueeze(0), atol=1e-5, rtol=0
    ).all(dim=2).any(dim=1)
    gt_filtered = gt_df.loc[match.cpu().numpy()]
    return gt_filtered


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


def save_video_from_results(results_file, models_dir, gt_X, gt_Y, model_name, task_name, video_path, fps=2.0):
    """
    Offline mode: given a completed results.csv and models directory, generate
    a video showing RMSE and log-likelihood evolving over trials.
    """
    trials, rmse, ll = compute_metrics_so_far(results_file, models_dir, gt_X, gt_Y, model_name)
    if not trials:
        raise ValueError(
            "No trials found when computing metrics for video. "
            "Ensure results_file and models exist and that there are at least 2 trials."
        )

    fig, (ax_rmse, ax_ll) = plt.subplots(1, 2, figsize=(14, 5))
    (line_rmse,) = ax_rmse.plot([], [], "b-o", markersize=4, linewidth=1.5, label="Current run")
    (line_ll,) = ax_ll.plot([], [], "b-o", markersize=4, linewidth=1.5, label="Current run")

    # Configure axes with fixed ranges based on full metric history
    ax_rmse.tick_params(axis='x', labelsize=14)
    ax_rmse.tick_params(axis='y', labelsize=14)
    ax_rmse.set_xlabel("Trial", fontsize=16)
    ax_rmse.set_ylabel("RMSE", fontsize=16)
    ax_rmse.set_title("RMSE vs. Trials", fontsize=18, fontweight="bold")
    ax_rmse.legend(loc="upper right")
    ax_rmse.grid(True, alpha=0.3)

    ax_ll.tick_params(axis='x', labelsize=14)
    ax_ll.tick_params(axis='y', labelsize=14)
    ax_ll.set_xlabel("Trial", fontsize=16)
    ax_ll.set_ylabel("Log-Likelihood", fontsize=16)
    ax_ll.set_title("Log-Likelihood vs. Trials", fontsize=18, fontweight="bold")
    ax_ll.legend(loc="lower right")
    ax_ll.grid(True, alpha=0.3)

    title = "Metrics over Eval Trials"
    if task_name:
        title += f" — {task_name}"
    fig.suptitle(title, fontsize=22, y=0.93)

    # Set fixed limits for the whole animation
    ax_rmse.set_xlim(0, max(trials) + 1)
    rmin, rmax = min(rmse), max(rmse)
    rm_margin = (rmax - rmin) * 0.1 or 0.1
    ax_rmse.set_ylim(rmin - rm_margin, rmax + rm_margin)

    ax_ll.set_xlim(0, max(trials) + 1)
    lmin, lmax = min(ll), max(ll)
    ll_margin = (lmax - lmin) * 0.1 or 0.1
    ax_ll.set_ylim(lmin - ll_margin, lmax + ll_margin)

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    def init():
        line_rmse.set_data([], [])
        line_ll.set_data([], [])
        return line_rmse, line_ll

    def update(frame_idx):
        # Show metrics up to this trial
        end = frame_idx + 1
        line_rmse.set_data(trials[:end], rmse[:end])
        line_ll.set_data(trials[:end], ll[:end])
        return line_rmse, line_ll

    num_frames = len(trials)
    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=num_frames,
        blit=False,
        cache_frame_data=False,
    )

    # Select writer: prefer ffmpeg, fall back to Pillow (GIF) if unavailable
    writer = None
    try:
        writer = animation.FFMpegWriter(fps=fps)
        ext = os.path.splitext(video_path)[1].lower()
        if ext not in [".mp4", ".mkv", ".avi", ".mov"]:
            # Default to .mp4 if no known video extension
            video_path = video_path + ".mp4"
    except Exception:
        from matplotlib.animation import PillowWriter
        writer = PillowWriter(fps=fps)
        ext = os.path.splitext(video_path)[1].lower()
        if ext not in [".gif"]:
            video_path = video_path + ".gif"

    os.makedirs(os.path.dirname(os.path.abspath(video_path)) or ".", exist_ok=True)
    ani.save(video_path, writer=writer, dpi=150)
    print(f"Saved metrics video ({num_frames} frames) -> {video_path}")


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
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="If set, run in offline mode and save a video (or GIF fallback) "
             "showing RMSE/log-likelihood evolving over existing trials.",
    )
    parser.add_argument(
        "--video_fps",
        type=float,
        default=2.0,
        help="Frames per second for --video_path animation (offline mode).",
    )
    args = parser.parse_args()

    # Load ground truth once and filter to current design space (same as eval.py's pool)
    gt_df = _load_data_safe(args.gt_results_file)
    if gt_df is None or len(gt_df) == 0:
        raise FileNotFoundError(
            f"Cannot load ground truth from '{args.gt_results_file}'. "
            "Ensure the file exists and is readable."
        )
    gt_df = _filter_gt_to_design_space(gt_df, task_name=args.task)
    if len(gt_df) == 0:
        raise ValueError(
            "No ground truth rows match the current design space. "
            "Ensure the GT CSV has factor columns consistent with factors_config and that "
            "at least one row lies in the valid design space (e.g. after limiting VIEWPOINT_VALUES)."
        )
    print(f"Live plot: using {len(gt_df)} ground truth points (filtered to current design space).")
    gt_X, gt_Y = _get_tensors_from_df(gt_df)
    models_dir = os.path.join(os.path.dirname(os.path.abspath(args.results_file)), "models")

    if args.video_path is not None:
        if args.save_path is not None:
            raise ValueError("Cannot use both --video_path and --save_path. Choose one output mode.")
        save_video_from_results(
            results_file=args.results_file,
            models_dir=models_dir,
            gt_X=gt_X,
            gt_Y=gt_Y,
            model_name=args.model_name,
            task_name=args.task,
            video_path=args.video_path,
            fps=args.video_fps,
        )
        return

    # Figure: two subplots
    fig, (ax_rmse, ax_ll) = plt.subplots(1, 2, figsize=(14, 5))
    (line_rmse,) = ax_rmse.plot([], [], "b-o", markersize=4, linewidth=1.5, label="Current run")
    (line_ll,) = ax_ll.plot([], [], "b-o", markersize=4, linewidth=1.5, label="Current run")

    ax_rmse.tick_params(axis='x', labelsize=14)
    ax_rmse.tick_params(axis='y', labelsize=14)
    ax_rmse.set_xlabel("Trial", fontsize=16)
    ax_rmse.set_ylabel("RMSE", fontsize=16)
    ax_rmse.set_title("RMSE vs. Trials", fontsize=18, fontweight="bold")
    ax_rmse.legend(loc="upper right")
    ax_rmse.grid(True, alpha=0.3)

    ax_ll.tick_params(axis='x', labelsize=14)
    ax_ll.tick_params(axis='y', labelsize=14)
    ax_ll.set_xlabel("Trial", fontsize=16)
    ax_ll.set_ylabel("Log-Likelihood", fontsize=16)
    ax_ll.set_title("Log-Likelihood vs. Trials", fontsize=18, fontweight="bold")
    ax_ll.legend(loc="lower right")
    ax_ll.grid(True, alpha=0.3)

    title = "Metrics over Eval Trials"
    if args.task:
        title += f" — {args.task}"
    fig.suptitle(title, fontsize=22, y=0.93)

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

    plt.tight_layout(rect=[0, 0, 1, 0.92])

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
