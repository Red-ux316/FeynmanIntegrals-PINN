# Tutorial — using the code, step by step

For those who did not write the code. Each step says **what it does**, **the command**, and **what to expect**.
Commands are given from the root of the repository, in a terminal where the environment is active
(`conda activate pinn-feynman`).

What is where:
- [code_map.md](code_map.md): every file and every function, with the points where they are used;
- `README.md` of each folder in `src/pinn_feynman/`: the explanation of the modules;
- `README.md` of each folder in `configs/`: what each configuration changes and what each queue runs.

---

## 0. The overview in one page

The example of this tutorial is the one-loop massless box. Its three master integrals $M_1, M_2, M_3$ satisfy
$dM/dx = A(x,\varepsilon)\,M$ with $x=t/s$; the network learns the **Laurent coefficients** of the box $M_3$, orders
$\varepsilon^{-2}\ldots\varepsilon^{2}$, as functions of $x$. The two bubbles $M_1, M_2$ are known in closed form and
enter the equation of the box as a forcing term (**bottom-up** formulation). The network receives $10x$, as in Boni's
thesis.

The code chain, one package per ring:

```
systems/     the matrix A(x, eps) of the equation dM/dx = A M              (thesis, eq. A.2)
reference/   the true values, from closed-form formulas                   (Bern-Dixon-Kosower)
providers/   the known sub-topologies of the bottom-up formulation        (the forcing term)
data/        the points (boundary, validation, test) with the true values, cached on disk
models/      the network (MLP 1 -> 32 -> 32 -> 32 -> 5 for the massless box)
physics/     network derivatives, data loss (6.11) and physics loss (6.12)
balancing/   the weights of the two losses (ReLoBRaLo)
collocation/ the points on which the equation is enforced
training/    the training loop, checkpoints, queues, the command to launch them
evaluation/  errors on the test set and figures
```

All choices live in **one configuration file** per experiment: `configs/<problem>/<name>.yaml`. The bottom-up
baseline of the massless box is `configs/box_massless/bu_baseline.yaml` (Boni's Tab. 6.1 with the bottom-up system).

Datasets and runs go to `$PINN_FEYNMAN_STORAGE` if the variable is set, otherwise to `storage/` at the root of the
repository: `storage/data/` and `storage/runs/`.

## 1. Generate the data (once only)

```
python -m pinn_feynman.data.build configs/box_massless/bu_baseline.yaml
```
Creates in `storage/data/` the datasets of the configuration:
- the 2 boundary points;
- the 1000 validation points;
- the 128 Chebyshev nodes and the 100 000 test points interpolated from them.

A few minutes. If the files already exist, nothing is recomputed.

Optional check, i.e. how accurate the interpolated test set is:
```
python -m pinn_feynman.data.check_interpolation configs/box_massless/bu_baseline.yaml
```
Expected: maximum error about $10^{-13}$.

To look at the data in Python:
```python
from pinn_feynman.data.build import dataset_specs
from pinn_feynman.data.cache import get_dataset
from pinn_feynman.config import load_config
specs = dataset_specs(load_config("configs/box_massless/bu_baseline.yaml"))
val = get_dataset(specs["validation"])
val.x.shape, val.values.shape          # (1000,), (1000, 3, 5) complex: all three masters
```

## 2. Train the network

Two separate training runs, real part and imaginary part (thesis §6.2.1.1):
```
python -m pinn_feynman.training.run configs/box_massless/bu_baseline.yaml --part real
python -m pinn_feynman.training.run configs/box_massless/bu_baseline.yaml --part imag
```
About 6 s per epoch on one CPU thread, so about 40 minutes for the 400 epochs. They can be launched **together**, in
two terminals: each uses one thread.

A progress bar shows the epochs with the last losses; every `log_every` epochs a line stays above it: `train` is the
weighted loss, `data` and `DE` the two losses, `val` the validation loss, `lr` the learning rate.

Useful options:

| Option | Effect |
|---|---|
| `--epochs 20` | short training, to try things out |
| `--seed 1` | another random initialization |
| `--resume storage/runs/<folder>` | **resumes** an interrupted training run from the last checkpoint |

The outputs live in `storage/runs/<problem>-<config>_<part>_<date-time>/`:

| File | Content |
|---|---|
| `config.yaml` | the configuration used |
| `history.csv` | the losses epoch by epoch |
| `checkpoint.pt` | the complete state (needed by `--resume`) |
| `best_model.pt`, `final_model.pt` | the best network (minimum $L_{val}$) and the final one |
| `summary.json` | the summary |

## 3. Evaluate

```
python -m pinn_feynman.evaluation.evaluate storage/runs/<real run> storage/runs/<imaginary run>
```
For each run it writes into its folder:

| File | Content |
|---|---|
| `evaluation.json` | absolute and relative errors: means, medians, 95th percentile, maxima; by order, by master, by component, plus Boni's numbers for comparison |
| `loss.png` | loss curves over training (like Fig. 6.2) |
| `hist_abs.png`, `hist_rel.png` | error histograms by order (Fig. 6.3); the dashed line is Boni's mean |
| `curves.png` | the network over the exact solutions (Fig. 6.5–6.7) |
| `error_vs_x.png` | error along the domain (not in the thesis) |

With two runs, the parent folder also gets `real_vs_imag.png` (Fig. 6.4).

## 4. Training queues

```
python -m pinn_feynman.training.queue configs/box_massless/queues/bottom_up.yaml --status
python -m pinn_feynman.training.queue configs/box_massless/queues/bottom_up.yaml --max-minutes 60
```
A queue is a list of runs (configuration, part, seed). Each run gets a fixed folder,
`storage/runs/<problem>-<config>_<part>[_seed<k>]`, so rerunning the queue skips the completed runs and resumes the
interrupted ones; `--max-minutes` pauses cleanly at the first checkpoint after that time. The queues of each problem
are listed in `configs/<problem>/README.md`. The notebook `notebooks/run_local.ipynb` runs one or more queues, builds
their datasets first and evaluates the finished runs.

## 5. Running a different experiment

The code is not touched. Write a new configuration that **inherits** from an existing one and changes only what is
needed, for example `configs/box_massless/bu_tanh.yaml`:
```yaml
base: bu_baseline.yaml

model:
  activation: tanh
```
Then:
```
python -m pinn_feynman.training.run configs/box_massless/bu_tanh.yaml --part real
```
What can be changed is listed in `configs/README.md`. The options available for each field (networks, activations,
balancers, samplers, point weightings) are in the registries, i.e. the dictionaries `MODELS`, `ACTIVATIONS`,
`BALANCERS`, `COLLOCATION_SAMPLERS` and `POINT_WEIGHTING` described in the folder READMEs. For example, R3 sampling on
the continuous domain is `collocation: {name: r3_continuous, ...}` and the distance weighting near the singular points
is `point_weighting: {name: distance}` (`src/pinn_feynman/collocation/README.md`, `src/pinn_feynman/weighting/README.md`).
`python -m pinn_feynman.devtools.list_configs box_massless` prints the inheritance tree of a folder.

## 6. CPU or GPU?

```
python -m pinn_feynman.devtools.benchmark configs/box_massless/bu_baseline.yaml --device cpu
```
prints the milliseconds per training step and the minutes for a full training. With Boni's batch (64 points) the CPU
was faster than a small laptop GPU; a GPU pays off only with much larger batches. For the GPU you need an environment
with a CUDA build of PyTorch and `device: cuda` (or `auto`) in the `training` section.

## 7. Hyperparameter tuning

A tuning study starts from a training configuration and draws the values of some of its fields with Optuna. Its file
(format in `src/pinn_feynman/tuning/README.md`) names the base configuration, the budget of the short trials and the
search space, for example `configs/box_massless/tune.yaml`:
```
python -m pinn_feynman.tuning.tune configs/box_massless/tune.yaml
```
Trials that do poorly are stopped early. In `storage/runs/tuning/<study>/` it writes the trial table, the Optuna plots
and **`best_config.yaml`**, the best configuration ready for a full training run. Rerunning the same command
**resumes** the study up to its number of trials. In a queue, an entry `{tune: <problem>/<name>}` runs the study and
writes its best configuration as `configs/<problem>/tuned/<study>.yaml`, which the next entries can train.

## 8. Adding a new diagram

A problem is declared in **one file** of `src/pinn_feynman/problems/`, which points to its other pieces; no other file
lists the problems.

| Piece | Where | What it must contain |
|---|---|---|
| declaration | `problems/<problem>.py` | `PROBLEM = Problem(name="<problem>", system=..., reference=..., boni=...)` (dotted names; the name is the file name); `parts=("real",)` when the imaginary part vanishes in the domain |
| system: matrices $A=\sum_j\varepsilon^jA^{(j)}$ in sympy | `systems/<problem>.py` | `VARIABLES`, `N_MASTERS`, `connection_matrix_orders(variable)` |
| true values | `reference/<problem>.py` | `reference_values(points) -> (N, n_MI, n_orders)`, `ORDERS`, `CACHE_PARAMS` (what identifies the output: it enters the cache key), `COMPUTE_OPTIONS` (how the points are split) |
| bottom-up version | `problems/<problem>_bu.py`, `providers/<problem>.py` | the masters the network learns (`unknowns`) and the provider of the known sub-topologies (the forcing term) |

`physics.SYSTEMS`, `data.generation.GENERATORS` and `evaluation.boni_reference.BONI` are read-only views of these
files. A problem without a closed formula can integrate its own system from values computed once elsewhere: the
equal-mass sunrise for s > 3 starts from three AMFlow values (`reference/starting_values.csv`). Problems with several
variables (the one-mass box has two) are supported by the data, physics and training code.

Then create `configs/<problem>/bu_baseline.yaml`.

## 9. Where to look when changing something

1. Open [code_map.md](code_map.md), section 2, and look up the function.
2. The **Used in** column lists the file and line of **every** point where it appears: these are the places to update.
3. After the change, regenerate the map:
   ```
   python -m pinn_feynman.devtools.codemap
   ```
