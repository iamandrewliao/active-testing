#!/bin/bash
# Runs offline evaluation and then visualization of the results.

# --- Editable Variables ---
TASK="pickblueblock"
MODEL="SingleTaskGP"
ACQ="PSD"
LOAD_PATH="results/pickblueblock_bruteforce/results.csv"
ACTIVE_ID="${TASK}_active_offline_${MODEL}_${ACQ}"
IID_ID="${TASK}_iid_offline_${MODEL}"

# 1. Run Active Evaluation
uv run offline_eval.py --mode active --num_evals 50 --num_init_pts 15 \
    --load_path "$LOAD_PATH" --model_name "$MODEL" --acq_func_name "$ACQ" \
    --task "$TASK" --eval_id "$ACTIVE_ID"

# 2. Run IID Evaluation
uv run offline_eval.py --mode iid --num_evals 50 \
    --load_path "$LOAD_PATH" --model_name "$MODEL" \
    --task "$TASK" --eval_id "$IID_ID"

# 3. Visualize Results
uv run viz.py plot-metrics-vs-trials \
    --gt_results_file "$LOAD_PATH" \
    --active_results_file "results/$ACTIVE_ID/results.csv" \
    --iid_results_file "results/$IID_ID/results.csv" \
    --task "$TASK" \
    --output_file "visualizations/robo_eval/${TASK}_offline_metrics_vs_trials_${MODEL}_${ACQ}.png"