# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Run a training queue locally
#
# Runs a **queue** of the project (trainings and tuning studies) on this machine, then evaluates the finished runs.
# The notebook contains no logic: it calls the package's commands (`data.build`, `training.queue`,
# `evaluation.evaluate`).
#
# - **Code:** the repository this notebook is in, installed with `pip install -e .` (see the README at the root).
# - **Data and runs:** in `$PINN_FEYNMAN_STORAGE` if it is set, otherwise in `storage/` at the root of the repository
#   (`data/` = cached datasets, `runs/` = one folder per training). Interrupted runs resume from their checkpoint.
#
# Coupled file: `run_local.ipynb` ↔ `run_local.py` (jupytext). After every change to the `.py`:
# `python -m jupytext --sync notebooks/run_local.py`.

# %% [markdown]
# ## 0. Settings (the only cell to edit)

# %%
# Queue(s) to run, in this order: a path, or a list of paths. Every queue of the repository is listed, one
# problem at a time; uncomment the ones to run.
QUEUE = [
    # massless box
    "configs/box_massless/queues/bottom_up.yaml",
    # "configs/box_massless/queues/bu_explore.yaml",
    # "configs/box_massless/queues/bu_explore2.yaml",
    # "configs/box_massless/queues/bu_fixed_w_grid.yaml",
    # "configs/box_massless/queues/bu_gradrec.yaml",
    # "configs/box_massless/queues/bu_imag.yaml",
    # "configs/box_massless/queues/bu_long2000.yaml",
    # "configs/box_massless/queues/bu_m1_far.yaml",
    # "configs/box_massless/queues/bu_margin_long.yaml",
    # "configs/box_massless/queues/bu_near_m1_long.yaml",
    # "configs/box_massless/queues/bu_near_more.yaml",
    # "configs/box_massless/queues/bu_near_singularity.yaml",
    # "configs/box_massless/queues/bu_p3.yaml",
    # "configs/box_massless/queues/bu_r3_all.yaml",
    # "configs/box_massless/queues/bu_r3_near.yaml",
    # "configs/box_massless/queues/bu_seeds.yaml",
    # "configs/box_massless/queues/bu_seeds2.yaml",
    # "configs/box_massless/queues/bu_w64d4_exp_1000ep_seeds.yaml",
    # one-mass box
    # "configs/box_one_mass/queues/bottom_up.yaml",
    # "configs/box_one_mass/queues/bu_explore.yaml",
    # "configs/box_one_mass/queues/bu_explore2.yaml",
    # "configs/box_one_mass/queues/bu_fixed_w_grid.yaml",
    # "configs/box_one_mass/queues/bu_gradrec.yaml",
    # "configs/box_one_mass/queues/bu_imag.yaml",
    # "configs/box_one_mass/queues/bu_near.yaml",
    # "configs/box_one_mass/queues/bu_p3.yaml",
    # "configs/box_one_mass/queues/bu_r3_seeds.yaml",
    # "configs/box_one_mass/queues/bu_seeds.yaml",
    # "configs/box_one_mass/queues/bu_w128d3_points4x_seeds.yaml",
    # equal-mass sunrise, s > 3
    # "configs/sunrise_equal_mass/queues/bu_near_singularity.yaml",
    # "configs/sunrise_equal_mass/queues/bu_p3.yaml",
    # "configs/sunrise_equal_mass/queues/bu_phys_3_27_w128d4_exp_1000ep.yaml",
    # "configs/sunrise_equal_mass/queues/bu_phys_near.yaml",
    # "configs/sunrise_equal_mass/queues/bu_physical.yaml",
    # "configs/sunrise_equal_mass/queues/bu_physical2.yaml",
    # "configs/sunrise_equal_mass/queues/bu_physical_seeds.yaml",
    # "configs/sunrise_equal_mass/queues/bu_r3_all.yaml",
    # "configs/sunrise_equal_mass/queues/bu_r3_exp.yaml",
    # "configs/sunrise_equal_mass/queues/bu_threshold_w128d4_exp_1000ep.yaml",
    # non-planar three-point function
    # "configs/nonplanar_three_point/queues/bu_thr.yaml",
    # several problems
    # "configs/queues/bu_exp_replace.yaml",
    # "configs/queues/bu_seeds_3x3.yaml",
]
STORAGE = None             # folder for data/ and runs/; None: $PINN_FEYNMAN_STORAGE, else <repository>/storage
DATA_ONLY = False          # True: only build the datasets of QUEUE (cell 2), no training
MAX_MINUTES = None         # clean pause at the first checkpoint after this many minutes (None: no limit)
RUN_EVALUATION = True      # cell 4: test errors and figures of every finished run of the queue

# %% [markdown]
# ## 1. Environment

# %%
import os
import sys
from pathlib import Path

CODE_DIR = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
os.chdir(CODE_DIR)
if STORAGE is not None:
    os.environ["PINN_FEYNMAN_STORAGE"] = str(Path(STORAGE).resolve())

import pinn_feynman  # noqa: E402,F401  (first: sets the MKL threading layer)
from pinn_feynman import runtime  # noqa: E402
from pinn_feynman.progress import stream_command  # noqa: E402


def run(cmd, cwd=None):
    """Run a command and show its output as it arrives: progress bars are redrawn in place."""
    print(f"$ {cmd}", flush=True)
    code = stream_command(cmd, cwd=cwd)
    if code != 0:
        raise RuntimeError(f"command failed ({code}): {cmd}")


for key, value in runtime.summary().items():
    print(f"{key:12s} {value}")

# %% [markdown]
# ## 2. Data (computed once only)
# For each configuration in the queue (for a tuning study: its base configuration): boundary points, validation, test.
# Datasets already in `data/` are not recomputed. One line per dataset (`cached` or `computing`), and a progress bar for
# each one computed: the first time it takes minutes (about 1100 exact points per domain).

# %%
import yaml  # noqa: E402

from pinn_feynman.training.queue import TuneSpec, load_queue, queue_status  # noqa: E402

QUEUES = [QUEUE] if isinstance(QUEUE, str) else list(QUEUE)
specs = [spec for path in QUEUES for spec in load_queue(path)]
config_paths = set()
for spec in specs:
    if isinstance(spec, TuneSpec):
        tuning = yaml.safe_load(spec.tuning_path.read_text(encoding="utf-8"))
        config_paths.add(str(spec.tuning_path.parent / tuning["base_config"]))
    elif spec.config_path.exists():
        config_paths.add(str(spec.config_path))
configs = " ".join(f'"{path}"' for path in sorted(config_paths))
if configs:
    run(f"{sys.executable} -m pinn_feynman.data.build {configs} --workers {os.cpu_count()}")

# %% [markdown]
# ## 3. Queue: tuning studies and trainings
# Idempotent: skips completed runs and studies, resumes interrupted ones. A `tune` entry runs its study up to the total
# of trials; once finished, the best configuration is written as `configs/<problem>/tuned/<study>.yaml` and the entries
# that use it, until then `waiting`, are trained next. With `MAX_MINUTES` the current run pauses after a checkpoint:
# running the cell again restarts from there. The queues are chained: the next one starts when this one is finished.

# %%
from pinn_feynman.progress import header  # noqa: E402
from pinn_feynman.training.queue import print_table, run_queue  # noqa: E402

if DATA_ONLY:
    print("DATA_ONLY: datasets built (cell 2), no training; set DATA_ONLY = False to train")
for path in [] if DATA_ONLY else QUEUES:
    header(f"queue {path}")
    run_queue(load_queue(path), max_minutes=MAX_MINUTES)
    status = queue_status(load_queue(path))
    if any(state != "done" for state in status.values()):
        print_table(status)
        print(f"queue {path} is not finished: the next queues wait (run the cell again)")
        break

# %% [markdown]
# ## 4. Evaluation of the completed runs
# Only with `RUN_EVALUATION = True`: test errors and figures (`evaluation.json`, `*.png`) in the folder of each run.

# %%
from pinn_feynman.paths import runs_dir  # noqa: E402

for spec in specs if RUN_EVALUATION else []:
    if isinstance(spec, TuneSpec):
        continue
    directory = spec.directory()
    if queue_status([spec])[spec.run_name] == "done" and not (directory / "evaluation.json").exists():
        run(f"{sys.executable} -m pinn_feynman.evaluation.evaluate {directory}")
print(f"results in: {runs_dir()}")
