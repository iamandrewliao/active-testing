# Active testing for robot policy evaluation
## Key files
- [./eval.py](./eval.py): Main evaluation script, run alongside robot policy deployment

Example run commands:
```
uv run eval.py --mode brute_force --output_file results/bf_results.csv --save_points points/bf_points.csv --grid_resolution 11
uv run eval.py --mode iid --num_evals 50 --output_file results/iid_results.csv --save_points points/iid_points.csv --grid_resolution 11
uv run eval.py --mode active --num_evals 50 --output_file results/active_results.csv --save_points points/active_points.csv --grid_resolution 11
```
- [./viz.py](./viz.py): Visualization script for eval results, surrogate model, acquisition function, etc.

Example run commands:
```
uv run viz.py plot-points --results_file results/iid_results.csv --output_file visualizations/iid_points.png
uv run viz.py plot-active --grid_resolution 20 --results_file results/active_results.csv --output_file visualizations/active_plots.png
uv run viz.py plot-comparison --grid_resolution 20 --gt results/bf_results.csv --model Active results/active_results.csv --model IID results/iid_results.csv --output_file visualizations/comparison_bf_active_iid.png
```
- [./demo_eval.ipynb](./demo_eval.ipynb): Play around with active testing (BoTorch), visualize results (TODO)

**Running Jupyter Notebook in VSCode within this uv project:**  
Locally (not on a server):  
1. See https://docs.astral.sh/uv/guides/integration/jupyter/#using-jupyter-from-vs-code  

On a server:
1. Make sure ipykernel is installed.
```
uv pip install ipykernel
```
2. Run this code to create a dedicated kernel spec that points to your uv environment.
```
uv run python -m ipykernel install --user --name={choose a name}
```
3. Reload the VSCode window (in the VSCode Command Palette choose 'Developer: Reload Window'). This will not disrupt ongoing training.
4. Open your Jupyter Notebook and choose the new kernel you created. 