# pinn_feynman

Project package: PINN for the differential equations of the Master Integrals. Installed in *editable* mode in the `pinn-feynman` environment (`pip install -e .` from the repository root): code changes take effect immediately, without reinstalling.

| Subfolder | Responsibility |
|---|---|
| [problems/](problems/README.md) | one file per diagram that declares it (system, true values, Boni's numbers); the registry only finds it by name |
| [systems/](systems/README.md) | systems of differential equations: connection matrices, variables, domains |
| [reference/](reference/README.md) | true values of the Master Integrals from closed formulas (training and test data) |
| [data/](data/README.md) | boundary, validation and test datasets, with on-disk cache |
| [models/](models/README.md) | neural networks (MLP from Tab. 6.1) and their construction from configuration |
| [balancing/](balancing/README.md) | weights of the loss terms: ReLoBRaLo (§4.3), fixed weights |
| [collocation/](collocation/README.md) | collocation points: fixed base points with uniform (or Dirichlet) probabilities and per-epoch partitions (§6.1.1), fresh points, R3 on the continuum |
| [training/](training/README.md) | training loop, optimizer and scheduler, experiment construction, entry point |
| [evaluation/](evaluation/README.md) | test errors of a trained run (6.18–6.19), figures as in the thesis, comparison with the numbers of Boni |
| [devtools/](devtools/README.md) | developer tools: code map with the index of usages, timing of a training step, tree of the configurations |
| [weighting/](weighting/README.md) | per-point weights in the physics loss: none, or distance weighting (DW) near the singular points |
| [visualization/](visualization/README.md) | every figure: pure functions, data in, PNG out, one style |
| [tuning/](tuning/README.md) | hyperparameter tuning with Optuna, per-problem YAML search space |
| [providers/](providers/README.md) | known master integrals (sub-topologies) for the bottom-up formulation: closed forms, constants, composite |
| [physics/](physics/README.md) | connection matrix in torch, network derivatives, data loss (6.11) and physics loss (6.12) |

Modules at the root of the package:

| Module | What it does |
|---|---|
| `runtime.py` | `storage_root()` (root of `data/` and `runs/`: `$PINN_FEYNMAN_STORAGE`, otherwise the folder `storage/` at the repository root, not versioned), `repository_root()`, `summary()` (versions, device and paths) |
| `paths.py` | `project_root()` (repository root), `configs_dir()`, `data_dir()` (dataset cache: `$PINN_FEYNMAN_DATA_DIR`, otherwise `storage_root()/data`), `runs_dir()` (`storage_root()/runs`) |
| `config.py` | `load_config(path)`: YAML file → `dict`; `torch_dtype(name)` |
| `progress.py` | console output of all the commands: `bar(total, desc, unit, initial, leave, disable)` (tqdm on stdout, refreshed at most once per second outside a terminal), `note(text)` (a line above the active bar), `header(text)` (`== phase ==`), `table(rows)` (aligned columns), `duration(seconds)`, `stream_command(cmd, cwd)` (runs a command and copies its output as it arrives, carriage returns included: used by the notebook). `PINN_FEYNMAN_PROGRESS=0` turns the bars off |

Usage guide: `docs/tutorial.md`.
