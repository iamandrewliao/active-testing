# Active testing for robot policy evaluation
## Acquisition functions and surrogate models
### Available
I have included the following acquisition functions and surrogate models:
| Acq. function     | Surrogate models                                                                                           |
|-------------------|------------------------------------------------------------------------------------------------------------|
| qBALD, qNIPV, PSD | SingleTaskGP, I-BNN (Infinite-Width Bayesian NN), FullyBayesianSingleTaskGP, SaasFullyBayesianSingleTaskGP, MDN, DeepEnsemble |

### Compatibility
| Acq. function | Models that work                                                                    |
|---------------|-------------------------------------------------------------------------------------|
| qBALD         | Fully Bayesian models e.g. FullyBayesianSingleTaskGP, SaasFullyBayesianSingleTaskGP |
| qNIPV, PSD    | Any                                                                                 |
| BALD | MDN, Deep Ensemble |
## Key files
- [./testers.py](./testers.py): Implements the logic for active testing, iid testing (uniform-random), loading points, etc.
- [./utils.py](./utils.py): Helper functions
    - Note: is_valid_point() is highly setup-dependent and will most likely need to be adjusted. 
- [./factors_config.py](./factors_config.py): Factor configurations for your specific evaluation. Defines factors, tasks, task outcome ranges, etc. Make sure this is set up correctly before moving on to evaluation.
If you want to set a custom order of factors for evaluation, change the following:
```
# change the order of factors in this code in get_design_points_robot()
    v_grid, h_grid, x_grid, y_grid = torch.meshgrid(
        VIEWPOINT_VALUES,
        TABLE_HEIGHT_VALUES,
        OBJECT_POS_X_VALUES,
        OBJECT_POS_Y_VALUES,
        indexing="ij",
    )
```
If you want to set a custom order of values for an individual factor, change the following:
```
# example: viewpoint order 1 -> 2 -> 0
VIEWPOINT_VALUES = torch.tensor([1.0, 2.0, 0.0], **tkwargs)
```
- [./eval.py](./eval.py): Online evaluation script, run alongside robot policy deployment
Example run commands:
```
uv run eval.py --mode brute_force --task uprightcup --max_steps 35 --eval_id uprightcup_bruteforce
```
- [./offline_eval.py](./offline_eval.py): Offline evaluation script (active or IID sampling from brute force/ground truth results)
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

- [./viz.py](./viz.py): Visualization script for eval results, surrogate model, acquisition function, etc.

Example run commands:
```
# RMSE, log-likelihood over trials (comparison of active vs. IID vs. ground truth)
uv run viz.py plot-metrics-vs-trials \
  --gt_results_file results/uprightcup_bruteforce/results.csv \
  --active_results_dir results/uprightcup_active_offline_SingleTaskGP_PSD \
  --iid_results_dir results/uprightcup_iid_offline_SingleTaskGP \
  --task uprightcup --output_file visualizations/robo_eval/plot.png

# table of RMSE values for all factor combinations (for the surrogate model trained on either active or IID results)
uv run viz.py create-rmse-table \
  --eval_results_file results/uprightcup_active_offline/results.csv \
  --gt_results_file results/uprightcup_bruteforce/results.csv \
  --model_name SingleTaskGP \
  --task uprightcup \
  --output_file visualizations/robo_eval/uprightcup_offline_rmse_summary_table.csv

# OLD
uv run viz.py plot-points --results_file results/iid_results.csv --output_file visualizations/robo_eval/iid_points.png
# For plot-active and animate-active, make sure --model_name and --acq_func_name match what you used during evaluation so the acquired points make sense
uv run viz.py plot-active --grid_resolution 11 --results_file results/active_results.csv --output_file visualizations/robo_eval/active_plots.png --model_name SingleTaskGP --acq_func_name PSD
uv run viz.py animate-active --results_file results/active_results.csv --output_file visualizations/robo_eval/active_animation.mp4 --grid_resolution 11 --interval 750 --model_name SingleTaskGP --acq_func_name PSD
# For plot-comparison it is recommended that --model_name matches what you used during evaluation for consistency (however, you might be interested in other experiments like using a different model post-eval)
uv run viz.py plot-comparison --grid_resolution 11 --gt_results_file results/bf_results.csv --add_results_file Active results/active_results.csv --add_results_file IID results/iid_results.csv --output_file visualizations/robo_eval/comparison_bf_active_iid.png --model_name SingleTaskGP --plot_mode mean
```
**Note:** The grid_resolution should be a multiple of the resolution used during eval.
- [./next_data_to_collect.py](./next_data_to_collect.py): Based on active testing results, determines what data to collect (and retrain on) next. (TO DO: add other more interesting methods)
```
uv run next_data_to_collect.py
```
- [./test_active.py](./test_active.py): Evaluate active testing components (surrogate, acq. function) on test functions like Hartmann, visualize metrics
```
uv run test_active.py --save_path ./visualizations/test_function/PSD_SingleTaskGP.png --model_name SingleTaskGP --acq_func_name PSD
```