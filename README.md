# Active testing for robot policy evaluation
## Key files
- [./testers.py](./testers.py): Implements the logic for active testing, iid testing, loading points, etc.
- [./utils.py](./utils.py): Helper functions
- [./eval.py](./eval.py): Main evaluation script, run alongside robot policy deployment (TO DO: don't train model in every other script (viz.py, etc.), just use trained model from evaluation)

Example run commands:
```
uv run eval.py --mode brute_force --num_evals 100 --output_file results/bf_results.csv --save_points points/bf_points.csv --grid_resolution 11
uv run eval.py --mode iid --num_evals 50 --output_file results/iid_results.csv --save_points points/iid_points.csv --grid_resolution 11
uv run eval.py --mode active --num_evals 50 --output_file results/active_results.csv --save_points points/active_points.csv --grid_resolution 11
uv run eval.py --mode loaded --load_path points/policyA_bf_points.csv --output_file results/policyB_bf_results.csv
```
- [./viz.py](./viz.py): Visualization script for eval results, surrogate model, acquisition function, etc.

Example run commands:
```
uv run viz.py plot-points --results_file results/iid_results.csv --output_file visualizations/iid_points.png
uv run viz.py plot-active --grid_resolution 11 --results_file results/active_results.csv --output_file visualizations/active_plots.png
uv run viz.py animate-active --results_file results/active_results.csv --output_file visualizations/active_animation.mp4 --grid_resolution 11 --interval 750
uv run viz.py plot-comparison --grid_resolution 11 --gt results/bf_results.csv --model Active results/active_results.csv --model IID results/iid_results.csv --output_file visualizations/comparison_bf_active_iid.png
```
**Note:** The grid_resolution should be a multiple of the resolution used during eval.
- [./next_data_to_collect.py](./next_data_to_collect.py): Based on active testing results, determines what data to collect (and retrain on) next. (TO DO: add other more interesting methods (to be developed))
```
uv run next_data_to_collect.py
```
- [./test_active.py](./test_active.py): Evaluate active testing components (surrogate, acq. function) on test functions like Hartmann, visualize metrics
```
uv run test_active.py --save_path ./visualizations/test/sampler_comparison.png
```
**Note:** Make sure to change ActiveTester.get_next_point() in [./testers.py](./testers.py) first