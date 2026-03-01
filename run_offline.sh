#!/bin/bash
# Runs offline evaluation N times (stored under one meta-folder per method) and then visualization with mean ± std shading.

# CHANGE THESE
TASK="pickblueblock"
MODEL="MDN"
ACQ="BALD"
LOAD_PATH="results/${TASK}_bruteforce/results.csv"
NUM_RUNS=7
NUM_EVALS=100
NUM_INIT_PTS=30

# Meta-folder IDs (each run goes into results/{ID}/run_1/, run_2/, ...)
ACTIVE_SUFFIX="${MODEL}_${ACQ}" # in case you want to add a description
ACTIVE_ID="${TASK}_active_offline_${ACTIVE_SUFFIX}"
# IID_ID="${TASK}_iid_offline_${MODEL}"
IID_ID="${TASK}_iid_offline_SingleTaskGP"

# 1. Run active and IID evaluations NUM_RUNS times
for r in $(seq 1 "$NUM_RUNS"); do
    echo "--- Run $r / $NUM_RUNS ---"
    uv run offline_eval.py --mode active --num_evals $NUM_EVALS --num_init_pts $NUM_INIT_PTS \
        --load_path "$LOAD_PATH" --model_name "$MODEL" --acq_func_name "$ACQ" \
        --task "$TASK" --eval_id "$ACTIVE_ID" --run_num "$r" --sample_without_replacement \
        # --active_refit_interval 3 \
        # --active_warm_start

    # uv run offline_eval.py --mode iid --num_evals $NUM_EVALS \
    #     --load_path "$LOAD_PATH" --model_name "$MODEL" \
    #     --task "$TASK" --eval_id "$IID_ID" --run_num "$r" --sample_without_replacement
done

# 2. Visualize (plot-metrics-vs-trials discovers run_1..run_N inside each meta-folder)
uv run viz.py plot-metrics-vs-trials \
    --gt_results_file "$LOAD_PATH" \
    --active_results_dir "results/$ACTIVE_ID" \
    --iid_results_dir "results/$IID_ID" \
    --task "$TASK" \
    --output_file "visualizations/robo_eval/${TASK}_offline_metrics_vs_trials_${ACTIVE_SUFFIX}.png"
