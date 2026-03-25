# Active testing for robot policy evaluation
## Acquisition functions and surrogate models
### Available
I have included the following acquisition functions and surrogate models:
| Acq. function     | Surrogate models                                                                                           |
|-------------------|------------------------------------------------------------------------------------------------------------|
| qBALD, qNIPV, PSD, BALD, qEPIG | SingleTaskGP, I-BNN (Infinite-Width Bayesian NN), FullyBayesianSingleTaskGP, SaasFullyBayesianSingleTaskGP, MDN, DeepEnsemble |

### Compatibility
| Acq. function | Models that work                                                                    |
|---------------|-------------------------------------------------------------------------------------|
| qBALD         | Fully Bayesian models e.g. FullyBayesianSingleTaskGP, SaasFullyBayesianSingleTaskGP |
| PSD    | Any                                                                                 |
| BALD | MDN, Deep Ensemble |
| qEPIG, qNIPV | SingleTaskGP |
## Key files
### Helper files:
- [testers.py](./testers.py): Implements the logic for active testing, iid testing (uniform-random), loading points, etc.
- [utils.py](./utils.py): Helper functions
    - Note: is_valid_point() is highly setup-dependent and will most likely need to be adjusted. 
- [factors_config.py](./factors_config.py): Factor configurations for your specific evaluation. Defines factors, tasks, task outcome ranges, etc. Make sure this is set up correctly before moving on to evaluation.
If you want to set a custom order of factors for evaluation, change the following:
```
# To change the order of factors, change this code in get_design_points_robot()
    v_grid, h_grid, x_grid, y_grid = torch.meshgrid(
        VIEWPOINT_VALUES,
        TABLE_HEIGHT_VALUES,
        OBJECT_POS_X_VALUES,
        OBJECT_POS_Y_VALUES,
        indexing="ij",
    )

# To change the order of values for an individual factor, change this code
# example: viewpoint order 1 -> 2 -> 0
VIEWPOINT_VALUES = torch.tensor([1.0, 2.0, 0.0], **tkwargs)
```
### Evaluation scripts:
- [eval.py](./eval.py): Online evaluation script, run alongside robot policy deployment
Example run commands:
```
# Brute-force
uv run eval.py --mode brute_force --task uprightcup --max_steps 35 --eval_id uprightcup_bruteforce

# Active with live plotting (additionally, add --save_path if you're on headless so instead of *showing* the plot it will save a plot and update it instead)
uv run eval.py --mode active --num_evals 50 --num_init_pts 15 --model_name SingleTaskGP --acq_func_name PSD --task pickblueblock --eval_id pickblueblock_live_demo --live_plot --live_plot_gt_file results/pickblueblock_bruteforce/results.csv
```
- [run_offline.sh](./run_offline.sh): Runs [offline_eval.py](./offline_eval.py) and creates a plot from [viz.py](./viz.py) with user-specified configuration.
- [offline_eval.py](./offline_eval.py): Offline evaluation script (active or IID sampling from brute force/ground truth results)
Example run commands:
```
# Active testing
uv run offline_eval.py \
  --mode active \
  --num_evals 50 \
  --num_init_pts 15 \
  --load_path results/uprightcup_bruteforce/results.csv \
  --model_name SingleTaskGP \
  --acq_func_name PSD \
  --task uprightcup \
  --eval_id uprightcup_active_offline \
  --sample_without_replacement

# IID testing
uv run offline_eval.py \
  --mode iid \
  --num_evals 50 \
  --load_path results/uprightcup_bruteforce/results.csv \
  --model_name SingleTaskGP \
  --task uprightcup \
  --eval_id uprightcup_iid_offline
  --sample_without_replacement
```

**`--sample_without_replacement`** Use this flag in either eval.py or offline_eval.py to sample without replacement (each point in the design pool is used at most once) in IID testing or the initial random phase of active testing (the active phase already samples without replacement (see ActiveTester)).

- [test_active.py](./test_active.py): Evaluate active testing components (surrogate, acq. function) on test functions like Hartmann, visualize metrics
```
uv run test_active.py --save_path ./visualizations/test_function/PSD_SingleTaskGP.png --model_name SingleTaskGP --acq_func_name PSD
```

### Visualization:
- [viz.py](./viz.py): Visualization script for eval results, surrogate model, acquisition function, etc.
Example run commands:
```
# RMSE, log-likelihood over trials (comparison of active vs. IID vs. ground truth)
# Note that you can add multiple active_results_dir e.g. for different surrogate model + acquisition function runs
uv run viz.py plot-metrics-vs-trials
  --gt_results_file "results/pickblueblock_bruteforce/results.csv"
  --active_results_dir "results/pickblueblock_active_offline_DeepEnsemble_BALD"
  --active_results_dir "results/pickblueblock_active_offline_SingleTaskGP_PSD"
  --iid_results_dir "results/pickblueblock_iid_offline_SingleTaskGP"
  --task "pickblueblock"
  --output_file "visualizations/robo_eval/pickblueblock_offline_metrics_vs_trials_multi.png"

# table of RMSE values for all factor combinations (for the surrogate model trained on either active or IID results)
uv run viz.py create-rmse-table \
  --eval_results_file results/uprightcup_active_offline/results.csv \
  --gt_results_file results/uprightcup_bruteforce/results.csv \
  --model_name SingleTaskGP \
  --task uprightcup \
  --output_file visualizations/robo_eval/uprightcup_offline_rmse_summary_table.csv
```
- [live_plot_eval.py](./live_plot_eval.py): Functions to help dynamically plot active/random eval results against ground truth (RMSE, log-likelihood). Can be run on its own but usually automatically runs by running eval.py with the --live_plot and --live_plot_gt_file flags.
Example run commands:
```
# Produces live plot window that constantly updates based on some results.csv file(s) of an evaluation (can be currently running); can compare multiple runs by adding more results.csv files in the '--results_file' flag (see command)
uv run live_plot_eval.py \
  --results_file results/run_a/results.csv results/run_b/results.csv \
  --gt_results_file results/task_bruteforce/results.csv \
  --task my_task \
  --labels "Run A" "Run B"

# Using '--save_path' saves and constantly updates plots instead
uv run live_plot_eval.py \
  --results_file results/run_a/results.csv results/run_b/results.csv \
  --gt_results_file results/task_bruteforce/results.csv \
  --task my_task \
  --labels "Run A" "Run B"
  --save_path visualizations/robo_eval/comparison


# Using '--video_path' generates video comparing multiple runs instead (probably more appropriate to use this when you're finished evaluating)
uv run live_plot_eval.py \
  --results_file results/run_a/results.csv results/run_b/results.csv \
  --gt_results_file results/task_bruteforce/results.csv \
  --task my_task \
  --labels "Run A" "Run B" \
  --video_path visualizations/robo_eval/comparison.mp4
```
- [model_analysis.ipynb](./model_analysis.ipynb): Notebook for quick plots, hyperparameter tuning

### Data curation:
- [next_data_to_collect.py](./next_data_to_collect.py): Based on active testing results, determines what data to collect (and retrain on) next. (TO DO: add other more interesting methods)
```
# Note that you can fix quadrant or other factor values with arguments
# certain failures method
uv run next_data_to_collect.py \
  --method certainfail \
  --results_file results/uprightcup_active_offline_SingleTaskGP_PSD/run_1/results.csv \
  --task uprightcup \
  --num_points 20 \
  --fix_factor table_height=1 \
  --fix_factor table_height=3 \
  --fix_factor camera_azimuth=90 \
  --fix_xy_quadrant bottom_left \
  --fix_xy_quadrant_top_right \
  --output_dir results/next_demos/uprightcup

# observed failures method
uv run next_data_to_collect.py \
  --method observed \
  --results_file results/uprightcup_iid_offline_SingleTaskGP/run_1/results.csv \
  --task uprightcup \
  --num_points 20 \
  --fix_factor table_height=1 \
  --fix_factor table_height=3 \
  --fix_factor camera_azimuth=90 \
  --fix_xy_quadrant bottom_left \
  --fix_xy_quadrant_top_right \
  --output_dir results/next_demos/uprightcup
```
[influence_curation.ipynb](./influence_curation.ipynb): Like next_data_to_collect.py, this notebook shows a new method of data curation, using the kernel as an "influence estimator". Currently only works for SingleTaskGP and FullyBayesianSingleTaskGP surrogate models.