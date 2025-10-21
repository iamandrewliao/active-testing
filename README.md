# Active testing for robot policy evaluation
## Key files
- [./eval.py](./eval.py): Main evaluation script, run alongside robot policy deployment

Example run command:
```
uv run eval.py --mode iid --num_evals 50 --output_file results/iid_50_results.csv --save_points points/iid_50_points.csv
```
- [./viz.py](./viz.py): Visualization script for eval results, surrogate model, acquisition function, etc.

Example run commands:
```
uv run viz.py plot-points --results_file results/iid_50_results.csv --out visualizations/iid_points.png
uv run viz.py plot-active --results_file results/active_50_results.csv --out visualizations/active_plots.png
uv run viz.py plot-comparison --gt results/brute_force_100_results.csv --model Active results/active_50_results.csv --model IID results/iid_50_results.csv
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