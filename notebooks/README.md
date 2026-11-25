# notebooks/

| File | What it does |
|---|---|
| `run_local.py` ↔ `run_local.ipynb` | runs one or more training queues on this machine: datasets, trainings and tuning studies (idempotent: finished runs are skipped, interrupted ones resume), then the test errors and figures of every finished run. Settings in the first cell |

The two files are **coupled with jupytext**: the `.py` is edited, then
```
python -m jupytext --sync notebooks/run_local.py
```
updates the `.ipynb`, which is the one to open in Jupyter.

The notebook contains no logic: it calls the package's commands (`pinn_feynman.data.build`, `pinn_feynman.training.queue`,
`pinn_feynman.evaluation.evaluate`).
