# Active testing for robot policy evaluation
## Key files
- [./eval.py](./eval.py): Main evaluation script, run alongside robot policy deployment
```
uv run eval.py
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