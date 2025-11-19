# Active testing for robot policy evaluation
## Acquisition functions and surrogate models
### Available
I have included the following acquisition functions and surrogate models:
| Acq. function     | Surrogate models                                                                                           |
|-------------------|------------------------------------------------------------------------------------------------------------|
| qBALD, qNIPV, PSD | SingleTaskGP, I-BNN (Infinite-Width Bayesian NN), FullyBayesianSingleTaskGP, SaasFullyBayesianSingleTaskGP |
### Compatibility
| Acq. function | Models that work                                                                    |
|---------------|-------------------------------------------------------------------------------------|
| qBALD         | Fully Bayesian models e.g. FullyBayesianSingleTaskGP, SaasFullyBayesianSingleTaskGP |
| qNIPV, PSD    | Any                                                                                 |
## Key files
- [./testers.py](./testers.py): Implements the logic for active testing, iid testing, loading points, etc.
- [./utils.py](./utils.py): Helper functions
- [./eval.py](./eval.py): Main evaluation script, run alongside robot policy deployment (TO DO: don't train model in every other script (viz.py, etc.), just use trained model from evaluation)

Example run commands:
```
uv run eval.py --mode brute_force --num_evals 100 --output_file results/bf_results.csv --save_points points/bf_points.csv --grid_resolution 11
uv run eval.py --mode iid --num_evals 50 --output_file results/iid_results.csv --save_points points/iid_points.csv --grid_resolution 11
uv run eval.py --mode active --num_evals 50 --output_file results/active_results.csv --save_points points/active_points.csv --grid_resolution 11 --model_name SingleTaskGP --acq_func_name PSD
uv run eval.py --mode loaded --load_path points/policyA_bf_points.csv --output_file results/policyB_bf_results.csv
```
- [./viz.py](./viz.py): Visualization script for eval results, surrogate model, acquisition function, etc.

Example run commands:
```
uv run viz.py plot-points --results_file results/iid_results.csv --output_file visualizations/robo_eval/iid_points.png
# For plot-active and animate-active, make sure --model_name and --acq_func_name match what you used during evaluation so the acquired points make sense
uv run viz.py plot-active --grid_resolution 11 --results_file results/active_results.csv --output_file visualizations/robo_eval/active_plots.png --model_name SingleTaskGP --acq_func_name PSD
uv run viz.py animate-active --results_file results/active_results.csv --output_file visualizations/robo_eval/active_animation.mp4 --grid_resolution 11 --interval 750 --model_name SingleTaskGP --acq_func_name PSD
# For plot-comparison it is recommended that --model_name matches what you used during evaluation for consistency (however, you might be interested in other experiments like using a different model post-eval)
uv run viz.py plot-comparison --grid_resolution 11 --gt_results_file results/bf_results.csv --add_results_file Active results/active_results.csv --add_results_file IID results/iid_results.csv --output_file visualizations/robo_eval/comparison_bf_active_iid.png --model_name SingleTaskGP --plot_mode mean
```
**Note:** The grid_resolution should be a multiple of the resolution used during eval.
- [./next_data_to_collect.py](./next_data_to_collect.py): Based on active testing results, determines what data to collect (and retrain on) next. (TO DO: add other more interesting methods (to be developed))
```
uv run next_data_to_collect.py
```
- [./test_active.py](./test_active.py): Evaluate active testing components (surrogate, acq. function) on test functions like Hartmann, visualize metrics
```
uv run test_active.py --save_path ./visualizations/test_function/PSD_SingleTaskGP.png --model_name SingleTaskGP --acq_func_name PSD
```
**Note:** Make sure to change ActiveTester.get_next_point() in [./testers.py](./testers.py) first