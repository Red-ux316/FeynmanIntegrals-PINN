# Code map

> **Generated file** by `python -m pinn_feynman.devtools.codemap`: do not edit by hand. `python -m pinn_feynman.devtools.codemap --check` exits with 1 if it is out of date.

How to use it: to know **what to touch when you change a function**, look it up in section 2 and read the *used in* column (files and lines). Usages are found by name: for generic names, marked with ⚠️, different objects with the same name may also appear.

## 1. Repository files

| File | Description |
|---|---|
| `configs/` | YAML configurations of the trainings and the queues that use them, one folder per problem (see its README) |
| `docs/README.md` | docs/ — technical documentation |
| `docs/tutorial.md` | Tutorial — using the code, step by step |
| `environment.yml` | Conda environment of the project. |
| `frobenius/mbox_x0.py` | Frobenius expansion of the massless box around x = 0, in Boni's basis. |
| `frobenius/plot_parts.py` | Massless box near x = 0: the true M_3^(k) (k = -1, 0, 1) and dM_3^(k)/dx split into the analytic part (the Frobenius |
| `frobenius/plot_series_error.py` | Massless box near x = 0: difference between the Frobenius form and the exact box. |
| `frobenius/README.md` | frobenius/ |
| `notebooks/README.md` | notebooks/ |
| `notebooks/run_local.py` |  |
| `pyproject.toml` | Runtime dependencies are provided by the conda environment (environment.yml). |
| `README.md` | FeynmanIntegrals-PINN |
| `src/pinn_feynman/__init__.py` | pinn_feynman: PINNs for the differential equations of Feynman master integrals. |
| `src/pinn_feynman/balancing/__init__.py` | Weights of the loss terms (data, physics) during training. |
| `src/pinn_feynman/balancing/factory.py` | Loss balancer from the `balancing` section of a configuration. |
| `src/pinn_feynman/balancing/fixed.py` | Constant weights for the loss terms (reference for comparison with ReLoBRaLo). |
| `src/pinn_feynman/balancing/README.md` | pinn_feynman.balancing |
| `src/pinn_feynman/balancing/relobralo.py` | ReLoBRaLo: Relative Loss Balancing with Random Lookback. |
| `src/pinn_feynman/collocation/__init__.py` | Collocation points for the physics loss: base points (uniform or Dirichlet, Boni sec. 6.1.1), fresh points, R3 on the continuum. |
| `src/pinn_feynman/collocation/factory.py` | Collocation sampler from the `collocation` section of a configuration. |
| `src/pinn_feynman/collocation/r3.py` | R3 sampling on the continuous domain (Daw et al., arXiv:2207.02338, Algorithm 1). |
| `src/pinn_feynman/collocation/README.md` | pinn_feynman.collocation |
| `src/pinn_feynman/collocation/sampler.py` | Collocation points on fixed base points (Boni, thesis sec. 6.1.1, eqs. 6.7-6.9) or fresh each epoch. |
| `src/pinn_feynman/config.py` | YAML configuration files, with inheritance. |
| `src/pinn_feynman/data/__init__.py` | Reference datasets (boundary, validation, test points) with an on-disk cache. |
| `src/pinn_feynman/data/build.py` | Generate (or find in the cache) the datasets of one or more configurations. |
| `src/pinn_feynman/data/cache.py` | On-disk cache of reference datasets (.npz files in paths.data_dir()). |
| `src/pinn_feynman/data/check_interpolation.py` | Check a Chebyshev-interpolated dataset against an exactly computed one. |
| `src/pinn_feynman/data/dataset.py` | What to generate (DatasetSpec) and what is stored (ReferenceDataset). |
| `src/pinn_feynman/data/domains.py` | Domains of the kinematic variables: an interval (one variable) or the one-mass triangle (two). |
| `src/pinn_feynman/data/generation.py` | Exact reference values on a grid of points, computed in parallel. |
| `src/pinn_feynman/data/interpolation.py` | Chebyshev interpolation on an interval (for the large test set). |
| `src/pinn_feynman/data/README.md` | pinn_feynman.data |
| `src/pinn_feynman/data/sampling.py` | Point samplers on a domain (`data.domains`: interval or triangle), selected by name. |
| `src/pinn_feynman/devtools/__init__.py` | Developer tools: code map with usage index (docs/code_map.md), step timing, configuration tree. |
| `src/pinn_feynman/devtools/benchmark.py` | Time training steps of a configuration on a device (CPU/GPU comparison). |
| `src/pinn_feynman/devtools/codemap.py` | Generate docs/code_map.md: files, public symbols, where each symbol is used, imports. |
| `src/pinn_feynman/devtools/list_configs.py` | Inheritance tree of the configuration files, with what each one changes with respect to its base. |
| `src/pinn_feynman/devtools/README.md` | pinn_feynman.devtools |
| `src/pinn_feynman/evaluation/__init__.py` | Evaluation of trained networks: errors (Boni eqs. 6.18-6.19), figures, comparison with the thesis. |
| `src/pinn_feynman/evaluation/boni_reference.py` | Results reported in Boni's thesis: massless box (sec. 6.2.1.1, printed pp. 102-103), one-mass box |
| `src/pinn_feynman/evaluation/evaluate.py` | Evaluate trained runs on the test set and draw the figures of the thesis. |
| `src/pinn_feynman/evaluation/metrics.py` | Error metrics of Boni, thesis eqs. (6.18)-(6.19). |
| `src/pinn_feynman/evaluation/plots.py` | Re-export of the figures of `pinn_feynman.visualization` used by `evaluate.py`. |
| `src/pinn_feynman/evaluation/predict.py` | Reload a training run and evaluate its network. |
| `src/pinn_feynman/evaluation/README.md` | pinn_feynman.evaluation |
| `src/pinn_feynman/models/__init__.py` | Neural networks for the master integrals. |
| `src/pinn_feynman/models/factory.py` | Model construction from the `model` section of a configuration. |
| `src/pinn_feynman/models/mlp.py` | Fully connected network: x -> Laurent coefficients of the master integrals. |
| `src/pinn_feynman/models/order_mask.py` | Output mask on the orders in eps: the network computes only the active orders, the others are exact zeros. |
| `src/pinn_feynman/models/README.md` | pinn_feynman.models |
| `src/pinn_feynman/paths.py` | Project locations, derived from the package position and the storage root (no hard-coded paths). |
| `src/pinn_feynman/physics/__init__.py` | Differential-equation side of the PINN: connection matrix, derivatives, losses. |
| `src/pinn_feynman/physics/bottom_up.py` | Bottom-up system: the network learns the unknowns U, the knowns K come from a provider. |
| `src/pinn_feynman/physics/coefficients.py` | The coefficients of the cascade residual at a set of points: A_0..A_p and an optional forcing term. |
| `src/pinn_feynman/physics/derivatives.py` | Derivatives of the network output with respect to its inputs, one per direction. |
| `src/pinn_feynman/physics/losses.py` | Training losses of Boni's PINN (thesis sec. 6.1.2). |
| `src/pinn_feynman/physics/propagation.py` | Propagation-failure statistics of the physics residual (Daw et al., arXiv:2207.02338, sec. 3, App. H). |
| `src/pinn_feynman/physics/README.md` | pinn_feynman.physics |
| `src/pinn_feynman/physics/system.py` | Connection matrices A = sum_j eps^j A_j as torch tensors, in the network's input variables. |
| `src/pinn_feynman/problems/__init__.py` | The diagrams the pipeline can train on: one file per problem, found by name (`get_problem`). |
| `src/pinn_feynman/problems/_registry.py` | The registry of problems: finds a problem by name, declares none. |
| `src/pinn_feynman/problems/box_massless.py` | One-loop massless box (Boni sec. 6.2.1): one variable x = t/s, true values in closed form. |
| `src/pinn_feynman/problems/box_massless_bu.py` | Massless box, bottom-up: the network learns the box M3, the bubbles M1 and M2 come in closed form. |
| `src/pinn_feynman/problems/box_one_mass.py` | One-loop one-mass box (Boni sec. 6.2.2): two variables (s1, s2) at s0 = 1, true values by series in Nielsen polylogs. |
| `src/pinn_feynman/problems/box_one_mass_bu.py` | One-mass box, bottom-up: the network learns the box M1, the three bubbles come in closed form. |
| `src/pinn_feynman/problems/nonplanar_three_point.py` | Non-planar two-loop three-point function (Boni sec. 6.3.3), the two masters of the top sector (m10, m11), x = -s/m^2. |
| `src/pinn_feynman/problems/nonplanar_three_point_bu.py` | Non-planar three-point, bottom-up name: the same problem as `nonplanar_three_point` (network on (m10, m11), |
| `src/pinn_feynman/problems/README.md` | pinn_feynman.problems |
| `src/pinn_feynman/problems/sunrise_equal_mass_positive.py` | Equal-mass sunrise for s > 0, m^2 = 3: domains between the singular points 0, 3, 27 of (A.10), true values by |
| `src/pinn_feynman/problems/sunrise_equal_mass_positive_bu.py` | Equal-mass sunrise for s > 0, bottom-up: the network learns the coupled block (I_11100, I_21100), the |
| `src/pinn_feynman/progress.py` | Console output of the commands: progress bars, notes, phase headers, tables. |
| `src/pinn_feynman/providers/__init__.py` | Providers of the known master integrals (sub-topologies) for the bottom-up formulation, see `base`. |
| `src/pinn_feynman/providers/base.py` | Providers of the known master integrals M_K (and of the forcing term) of the bottom-up formulation. |
| `src/pinn_feynman/providers/box_massless.py` | Known master integrals of the massless box: the two bubbles, in closed form (s = 1, x = t/s < 0). |
| `src/pinn_feynman/providers/box_one_mass.py` | Known master integrals of the one-mass box: the three bubbles, in closed form (s0 = 1, points (s1, s2)). |
| `src/pinn_feynman/providers/nonplanar_three_point.py` | Forcing term of the non-planar three-point masters (m10, m11): the subtopologies enter only through N = (0, N11). |
| `src/pinn_feynman/providers/README.md` | pinn_feynman.providers |
| `src/pinn_feynman/providers/sunrise_equal_mass.py` | Known master integral of the equal-mass sunrise: the product of two tadpoles I_11000, constant in s. |
| `src/pinn_feynman/README.md` | pinn_feynman |
| `src/pinn_feynman/reference/__init__.py` | Exact reference values of master integrals (ground truth for training and testing). |
| `src/pinn_feynman/reference/box_massless.py` | Exact reference values of the one-loop massless box master integrals. |
| `src/pinn_feynman/reference/box_one_mass.py` | Exact reference values of the one-loop one-mass box master integrals (physical region). |
| `src/pinn_feynman/reference/nonplanar_three_point.py` | Reference values of the non-planar three-point masters (m10, m11) (Boni sec. 6.3.3), x = -s/m^2, s -> s + i0. |
| `src/pinn_feynman/reference/README.md` | pinn_feynman.reference |
| `src/pinn_feynman/reference/starting_values.csv` |  |
| `src/pinn_feynman/reference/sunrise_equal_mass.py` | Reference values of the equal-mass sunrise master integrals (m^2 = 3, s < 0), by integration of Boni's (A.10). |
| `src/pinn_feynman/reference/sunrise_equal_mass_positive.py` | Reference values of the equal-mass sunrise master integrals (m^2 = 3) for s > 0, between the singular points of |
| `src/pinn_feynman/runtime.py` | Where data and runs are stored. |
| `src/pinn_feynman/systems/__init__.py` | Differential-equation systems of master integrals (connection matrices). |
| `src/pinn_feynman/systems/analysis.py` | Local analysis of a system dM/dx = A(x, eps) M at its singular points (Frobenius theory). |
| `src/pinn_feynman/systems/blocks.py` | Blocks of a system dM/dz_k = A_k M: dependency graph, bottom-up order, split into unknowns U and knowns K. |
| `src/pinn_feynman/systems/box_massless.py` | One-loop massless box: differential-equation system in Boni's basis. |
| `src/pinn_feynman/systems/box_massless_canonical.py` | One-loop massless box in a canonical (epsilon-factorised, dlog) basis. |
| `src/pinn_feynman/systems/box_one_mass.py` | One-loop one-mass box: differential-equation system in Boni's basis, physical region. |
| `src/pinn_feynman/systems/nonplanar_three_point.py` | Non-planar two-loop three-point function with a massive loop (Boni sec. 6.3.3): the two masters of the top sector. |
| `src/pinn_feynman/systems/README.md` | pinn_feynman.systems |
| `src/pinn_feynman/systems/singular_kinds.py` | Kind of every singular point (one variable) and singular line (two variables) of the systems: one place for it. |
| `src/pinn_feynman/systems/sunrise_equal_mass.py` | Two-loop equal-mass sunrise: differential-equation system in Boni's basis, m^2 = 3. |
| `src/pinn_feynman/training/__init__.py` | Training loop, optimizer/scheduler, experiment assembly and entry point. |
| `src/pinn_feynman/training/environment.py` | Where a run was made: library versions, CPU model, threads. |
| `src/pinn_feynman/training/history.py` | Per-epoch training record. |
| `src/pinn_feynman/training/optim.py` | Optimizer and learning-rate scheduler from the `training` section of a configuration. |
| `src/pinn_feynman/training/queue.py` | Run a queue of trainings, idempotently. |
| `src/pinn_feynman/training/README.md` | pinn_feynman.training |
| `src/pinn_feynman/training/recorder.py` | Training recorder: what happens during the training, observed without changing it. |
| `src/pinn_feynman/training/run.py` | Train one PINN (real or imaginary part) from a configuration, with checkpoints. |
| `src/pinn_feynman/training/setup.py` | Assemble all the pieces of an experiment from a configuration. |
| `src/pinn_feynman/training/trainer.py` | Training loop of Boni's PINN (thesis sec. 4.4, 6.1.2, Table 6.1). |
| `src/pinn_feynman/tuning/__init__.py` | Hyperparameter tuning with Optuna (per-problem YAML search space). |
| `src/pinn_feynman/tuning/figures.py` | Re-export of `pinn_feynman.visualization.tuning.tuning_history`. |
| `src/pinn_feynman/tuning/objective.py` | Objective of a tuning trial: short training, best validation loss, pruning. |
| `src/pinn_feynman/tuning/publish.py` | Write the best configuration of a finished study as a NEW configuration file. |
| `src/pinn_feynman/tuning/README.md` | pinn_feynman.tuning |
| `src/pinn_feynman/tuning/space.py` | Search space of a tuning study: YAML description -> Optuna suggestions -> configuration. |
| `src/pinn_feynman/tuning/tune.py` | Run (or resume) a tuning study, idempotently. |
| `src/pinn_feynman/visualization/__init__.py` | Every figure of the project: pure functions, data in, PNG out, one style (`style`). |
| `src/pinn_feynman/visualization/accuracy.py` | Accuracy of the network against the exact solution (Boni's Fig. 6.3-6.7): curves, errors along x and by |
| `src/pinn_feynman/visualization/components.py` | Figures per component M_i^(k) of the network output: the masters and the orders in epsilon separately. |
| `src/pinn_feynman/visualization/data_io.py` | The data behind the figures, saved next to a report (`report/figure_data/`) so that a figure can be redrawn without |
| `src/pinn_feynman/visualization/methods.py` | Colours and styles of the training methods, shared by the figures. |
| `src/pinn_feynman/visualization/README.md` | visualization/ |
| `src/pinn_feynman/visualization/singularities.py` | Dashed vertical lines at the singular points of a problem, on every figure with the domain on the x axis. |
| `src/pinn_feynman/visualization/style.py` | Style shared by every figure of the project: colours, matplotlib settings, saving, warnings. |
| `src/pinn_feynman/visualization/training.py` | Figures along the epochs: losses, training dynamics, adaptive-method columns. |
| `src/pinn_feynman/visualization/triangle.py` | Maps on a two-variable domain with a grid of panels: rows = groups (e.g. the real and imaginary parts), columns = |
| `src/pinn_feynman/visualization/tuning.py` | Figures of a tuning study (log scale: values span many decades). |
| `src/pinn_feynman/weighting/__init__.py` | Per-point weights of the physics loss (distance from the singular points); default: none. |
| `src/pinn_feynman/weighting/distance.py` | Distance weighting (DW): physics residual multiplied by the distance from the nearest pole of the block A_UU. |
| `src/pinn_feynman/weighting/factory.py` | Point weighting from the optional `point_weighting` section of a configuration. |
| `src/pinn_feynman/weighting/point_weights.py` | Per-point weights of the physics loss: the interface, and `NoWeighting` (Boni's plain mean). |
| `src/pinn_feynman/weighting/README.md` | pinn_feynman.weighting |

## 2. Package modules and public symbols

### `pinn_feynman`

[`src/pinn_feynman/__init__.py`](../src/pinn_feynman/__init__.py) — pinn_feynman: PINNs for the differential equations of Feynman master integrals.

### `pinn_feynman.balancing`

[`src/pinn_feynman/balancing/__init__.py`](../src/pinn_feynman/balancing/__init__.py) — Weights of the loss terms (data, physics) during training.

### `pinn_feynman.balancing.factory`

[`src/pinn_feynman/balancing/factory.py`](../src/pinn_feynman/balancing/factory.py) — Loss balancer from the `balancing` section of a configuration.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `BALANCERS` | constant |  | 6 | `src/pinn_feynman/balancing/factory.py` (14, 16) |
| `build_balancer` | function | `(config, n_losses, seed=None)` | 9 | `src/pinn_feynman/balancing/__init__.py` (3)<br>`src/pinn_feynman/training/setup.py` (7, 187) |

### `pinn_feynman.balancing.fixed`

[`src/pinn_feynman/balancing/fixed.py`](../src/pinn_feynman/balancing/fixed.py) — Constant weights for the loss terms (reference for comparison with ReLoBRaLo).

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `FixedWeights` | class |  | 6 | `src/pinn_feynman/balancing/__init__.py` (4)<br>`src/pinn_feynman/balancing/factory.py` (3, 6) |
| `FixedWeights.update` | method | `(self, losses)` | 15 | `src/pinn_feynman/balancing/relobralo.py` (97)<br>`src/pinn_feynman/collocation/factory.py` (13)<br>`src/pinn_feynman/data/generation.py` (61, 66)<br>`src/pinn_feynman/devtools/codemap.py` (132)<br>`src/pinn_feynman/devtools/list_configs.py` (30)<br>`src/pinn_feynman/evaluation/evaluate.py` (153)<br>`src/pinn_feynman/runtime.py` (62)<br>`src/pinn_feynman/training/queue.py` (63)<br>`src/pinn_feynman/training/recorder.py` (333, 355, 358)<br>`src/pinn_feynman/training/trainer.py` (236, 276, 277, 279, 281, 289)<br>`src/pinn_feynman/tuning/tune.py` (84)<br>`src/pinn_feynman/visualization/methods.py` (22, 23)<br>`src/pinn_feynman/weighting/factory.py` (11) ⚠️ generic name |
| `FixedWeights.state_dict` | method | `(self)` | 18 | `src/pinn_feynman/training/recorder.py` (351)<br>`src/pinn_feynman/training/run.py` (78, 102)<br>`src/pinn_feynman/training/trainer.py` (267, 298, 299, 300, 301, 302, 303, 304, …) ⚠️ generic name |
| `FixedWeights.load_state_dict` | method | `(self, state)` | 21 | `src/pinn_feynman/evaluation/predict.py` (92)<br>`src/pinn_feynman/training/run.py` (164)<br>`src/pinn_feynman/training/trainer.py` (315, 316, 317, 318, 319, 322, 323, 326) ⚠️ generic name |
| `FixedWeights.__call__` | method | `(self, losses)` | 24 | — |

### `pinn_feynman.balancing.relobralo`

[`src/pinn_feynman/balancing/relobralo.py`](../src/pinn_feynman/balancing/relobralo.py) — ReLoBRaLo: Relative Loss Balancing with Random Lookback.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `HISTORIES` | constant |  | 15 | `src/pinn_feynman/balancing/relobralo.py` (23, 24) |
| `ReLoBRaLo` | class |  | 18 | `src/pinn_feynman/balancing/__init__.py` (5)<br>`src/pinn_feynman/balancing/factory.py` (4, 6, 17) |
| `ReLoBRaLo.set_epoch` | method | `(self, epoch)` | 47 | `src/pinn_feynman/training/trainer.py` (250) |
| `ReLoBRaLo.update` | method | `(self, losses)` | 63 | `src/pinn_feynman/balancing/relobralo.py` (97)<br>`src/pinn_feynman/collocation/factory.py` (13)<br>`src/pinn_feynman/data/generation.py` (61, 66)<br>`src/pinn_feynman/devtools/codemap.py` (132)<br>`src/pinn_feynman/devtools/list_configs.py` (30)<br>`src/pinn_feynman/evaluation/evaluate.py` (153)<br>`src/pinn_feynman/runtime.py` (62)<br>`src/pinn_feynman/training/queue.py` (63)<br>`src/pinn_feynman/training/recorder.py` (333, 355, 358)<br>`src/pinn_feynman/training/trainer.py` (236, 276, 277, 279, 281, 289)<br>`src/pinn_feynman/tuning/tune.py` (84)<br>`src/pinn_feynman/visualization/methods.py` (22, 23)<br>`src/pinn_feynman/weighting/factory.py` (11) ⚠️ generic name |
| `ReLoBRaLo.state_dict` | method | `(self)` | 81 | `src/pinn_feynman/training/recorder.py` (351)<br>`src/pinn_feynman/training/run.py` (78, 102)<br>`src/pinn_feynman/training/trainer.py` (267, 298, 299, 300, 301, 302, 303, 304, …) ⚠️ generic name |
| `ReLoBRaLo.load_state_dict` | method | `(self, state)` | 86 | `src/pinn_feynman/evaluation/predict.py` (92)<br>`src/pinn_feynman/training/run.py` (164)<br>`src/pinn_feynman/training/trainer.py` (315, 316, 317, 318, 319, 322, 323, 326) ⚠️ generic name |
| `ReLoBRaLo.coefficients` | method | `(self)` | 91 | `src/pinn_feynman/physics/losses.py` (45)<br>`src/pinn_feynman/systems/blocks.py` (37, 38, 39)<br>`src/pinn_feynman/training/recorder.py` (171, 202)<br>`src/pinn_feynman/training/trainer.py` (79, 86, 91, 104) |
| `ReLoBRaLo.__call__` | method | `(self, losses)` | 95 | — |

### `pinn_feynman.collocation`

[`src/pinn_feynman/collocation/__init__.py`](../src/pinn_feynman/collocation/__init__.py) — Collocation points for the physics loss: base points (uniform or Dirichlet, Boni sec. 6.1.1), fresh points, R3 on the continuum.

### `pinn_feynman.collocation.factory`

[`src/pinn_feynman/collocation/factory.py`](../src/pinn_feynman/collocation/factory.py) — Collocation sampler from the `collocation` section of a configuration.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `COLLOCATION_SAMPLERS` | constant |  | 8 | `src/pinn_feynman/collocation/__init__.py` (3)<br>`src/pinn_feynman/collocation/factory.py` (13, 20, 22) |
| `sampler_class` | function | `(config)` | 16 | `src/pinn_feynman/collocation/__init__.py` (3)<br>`src/pinn_feynman/collocation/factory.py` (31)<br>`src/pinn_feynman/training/setup.py` (8, 64, 135, 147) |
| `build_collocation_sampler` | function | `(config, domain, seed=None, dtype=None, gate=None)` | 25 | `src/pinn_feynman/collocation/__init__.py` (3)<br>`src/pinn_feynman/training/setup.py` (8, 185) |

### `pinn_feynman.collocation.r3`

[`src/pinn_feynman/collocation/r3.py`](../src/pinn_feynman/collocation/r3.py) — R3 sampling on the continuous domain (Daw et al., arXiv:2207.02338, Algorithm 1).

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `R3ContinuousSampler` | class |  | 14 | `src/pinn_feynman/collocation/__init__.py` (4)<br>`src/pinn_feynman/collocation/factory.py` (5, 10) |
| `R3ContinuousSampler.base_points` | method | `(self)` | 58 | `src/pinn_feynman/collocation/r3.py` (82)<br>`src/pinn_feynman/collocation/sampler.py` (50, 62, 90, 124, 129, 148)<br>`src/pinn_feynman/training/recorder.py` (168)<br>`src/pinn_feynman/training/trainer.py` (76, 81, 104, 135, 155) |
| `R3ContinuousSampler.batches_per_epoch` | method | `(self)` | 63 | — |
| `R3ContinuousSampler.blocks_per_epoch` | method | `(self)` | 67 | — |
| `R3ContinuousSampler.to` | method | `(self, device)` | 71 | `src/pinn_feynman/balancing/fixed.py` (25)<br>`src/pinn_feynman/balancing/relobralo.py` (93, 100)<br>`src/pinn_feynman/collocation/r3.py` (55, 73, 112, 124)<br>`src/pinn_feynman/collocation/sampler.py` (62, 124, 129)<br>`src/pinn_feynman/data/domains.py` (75, 147)<br>`src/pinn_feynman/models/factory.py` (47)<br>`src/pinn_feynman/physics/bottom_up.py` (102)<br>`src/pinn_feynman/training/recorder.py` (207, 225, 265, 274, 279, 280, 292, 380, …)<br>`src/pinn_feynman/training/trainer.py` (49, 57, 58, 59, 60, 67, 142, 152)<br>`src/pinn_feynman/weighting/distance.py` (168, 169) ⚠️ generic name |
| `R3ContinuousSampler.epoch_indices` | method | `(self)` | 76 | `src/pinn_feynman/collocation/r3.py` (82)<br>`src/pinn_feynman/collocation/sampler.py` (90, 148)<br>`src/pinn_feynman/devtools/benchmark.py` (23)<br>`src/pinn_feynman/training/trainer.py` (191) |
| `R3ContinuousSampler.epoch` | method | `(self)` | 81 | `src/pinn_feynman/balancing/relobralo.py` (49)<br>`src/pinn_feynman/training/recorder.py` (84, 215, 240, 334, 339, 340, 350, 408, …)<br>`src/pinn_feynman/training/run.py` (88, 93, 165)<br>`src/pinn_feynman/training/trainer.py` (95, 231, 232, 237, 238, 247, 250, 266, …)<br>`src/pinn_feynman/tuning/objective.py` (17, 19) ⚠️ generic name |
| `R3ContinuousSampler.population_points` | method | `(self)` | 84 | `src/pinn_feynman/training/recorder.py` (238, 361)<br>`src/pinn_feynman/training/trainer.py` (254) |
| `R3ContinuousSampler.refreshes` | method | `(self)` | 89 | — |
| `R3ContinuousSampler.refresh` | method | `(self, residual_fn)` | 107 | `src/pinn_feynman/training/trainer.py` (201) |
| `R3ContinuousSampler.state_dict` | method | `(self)` | 118 | `src/pinn_feynman/training/recorder.py` (351)<br>`src/pinn_feynman/training/run.py` (78, 102)<br>`src/pinn_feynman/training/trainer.py` (267, 298, 299, 300, 301, 302, 303, 304, …) ⚠️ generic name |
| `R3ContinuousSampler.load_state_dict` | method | `(self, state)` | 122 | `src/pinn_feynman/evaluation/predict.py` (92)<br>`src/pinn_feynman/training/run.py` (164)<br>`src/pinn_feynman/training/trainer.py` (315, 316, 317, 318, 319, 322, 323, 326) ⚠️ generic name |

### `pinn_feynman.collocation.sampler`

[`src/pinn_feynman/collocation/sampler.py`](../src/pinn_feynman/collocation/sampler.py) — Collocation points on fixed base points (Boni, thesis sec. 6.1.1, eqs. 6.7-6.9) or fresh each epoch.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `PROBABILITIES` | constant |  | 19 | `src/pinn_feynman/collocation/sampler.py` (42, 43) |
| `BasePointsSampler` | class |  | 29 | `src/pinn_feynman/collocation/__init__.py` (5)<br>`src/pinn_feynman/collocation/factory.py` (6, 8, 9)<br>`src/pinn_feynman/collocation/sampler.py` (96) |
| `BasePointsSampler.to` | method | `(self, device)` | 60 | `src/pinn_feynman/balancing/fixed.py` (25)<br>`src/pinn_feynman/balancing/relobralo.py` (93, 100)<br>`src/pinn_feynman/collocation/r3.py` (55, 73, 112, 124)<br>`src/pinn_feynman/collocation/sampler.py` (62, 124, 129)<br>`src/pinn_feynman/data/domains.py` (75, 147)<br>`src/pinn_feynman/models/factory.py` (47)<br>`src/pinn_feynman/physics/bottom_up.py` (102)<br>`src/pinn_feynman/training/recorder.py` (207, 225, 265, 274, 279, 280, 292, 380, …)<br>`src/pinn_feynman/training/trainer.py` (49, 57, 58, 59, 60, 67, 142, 152)<br>`src/pinn_feynman/weighting/distance.py` (168, 169) ⚠️ generic name |
| `BasePointsSampler.state_dict` | method | `(self)` | 65 | `src/pinn_feynman/training/recorder.py` (351)<br>`src/pinn_feynman/training/run.py` (78, 102)<br>`src/pinn_feynman/training/trainer.py` (267, 298, 299, 300, 301, 302, 303, 304, …) ⚠️ generic name |
| `BasePointsSampler.load_state_dict` | method | `(self, state)` | 69 | `src/pinn_feynman/evaluation/predict.py` (92)<br>`src/pinn_feynman/training/run.py` (164)<br>`src/pinn_feynman/training/trainer.py` (315, 316, 317, 318, 319, 322, 323, 326) ⚠️ generic name |
| `BasePointsSampler.batches_per_epoch` | method | `(self)` | 73 | — |
| `BasePointsSampler.epoch_indices` | method | `(self)` | 76 | `src/pinn_feynman/collocation/r3.py` (82)<br>`src/pinn_feynman/collocation/sampler.py` (90, 148)<br>`src/pinn_feynman/devtools/benchmark.py` (23)<br>`src/pinn_feynman/training/trainer.py` (191) |
| `BasePointsSampler.epoch` | method | `(self)` | 88 | `src/pinn_feynman/balancing/relobralo.py` (49)<br>`src/pinn_feynman/training/recorder.py` (84, 215, 240, 334, 339, 340, 350, 408, …)<br>`src/pinn_feynman/training/run.py` (88, 93, 165)<br>`src/pinn_feynman/training/trainer.py` (95, 231, 232, 237, 238, 247, 250, 266, …)<br>`src/pinn_feynman/tuning/objective.py` (17, 19) ⚠️ generic name |
| `DirichletCollocationSampler` | function | `(domain, n_base, n_per_epoch, batch_size, concentration=1.0, seed=None, dtype=torch.float64)` | 93 | `src/pinn_feynman/collocation/__init__.py` (5) |
| `ResampleSampler` | class |  | 100 | `src/pinn_feynman/collocation/__init__.py` (5)<br>`src/pinn_feynman/collocation/factory.py` (6, 8) |
| `ResampleSampler.to` | method | `(self, device)` | 127 | `src/pinn_feynman/balancing/fixed.py` (25)<br>`src/pinn_feynman/balancing/relobralo.py` (93, 100)<br>`src/pinn_feynman/collocation/r3.py` (55, 73, 112, 124)<br>`src/pinn_feynman/collocation/sampler.py` (62, 124, 129)<br>`src/pinn_feynman/data/domains.py` (75, 147)<br>`src/pinn_feynman/models/factory.py` (47)<br>`src/pinn_feynman/physics/bottom_up.py` (102)<br>`src/pinn_feynman/training/recorder.py` (207, 225, 265, 274, 279, 280, 292, 380, …)<br>`src/pinn_feynman/training/trainer.py` (49, 57, 58, 59, 60, 67, 142, 152)<br>`src/pinn_feynman/weighting/distance.py` (168, 169) ⚠️ generic name |
| `ResampleSampler.state_dict` | method | `(self)` | 132 | `src/pinn_feynman/training/recorder.py` (351)<br>`src/pinn_feynman/training/run.py` (78, 102)<br>`src/pinn_feynman/training/trainer.py` (267, 298, 299, 300, 301, 302, 303, 304, …) ⚠️ generic name |
| `ResampleSampler.load_state_dict` | method | `(self, state)` | 135 | `src/pinn_feynman/evaluation/predict.py` (92)<br>`src/pinn_feynman/training/run.py` (164)<br>`src/pinn_feynman/training/trainer.py` (315, 316, 317, 318, 319, 322, 323, 326) ⚠️ generic name |
| `ResampleSampler.batches_per_epoch` | method | `(self)` | 139 | — |
| `ResampleSampler.epoch_indices` | method | `(self)` | 142 | `src/pinn_feynman/collocation/r3.py` (82)<br>`src/pinn_feynman/collocation/sampler.py` (90, 148)<br>`src/pinn_feynman/devtools/benchmark.py` (23)<br>`src/pinn_feynman/training/trainer.py` (191) |
| `ResampleSampler.epoch` | method | `(self)` | 147 | `src/pinn_feynman/balancing/relobralo.py` (49)<br>`src/pinn_feynman/training/recorder.py` (84, 215, 240, 334, 339, 340, 350, 408, …)<br>`src/pinn_feynman/training/run.py` (88, 93, 165)<br>`src/pinn_feynman/training/trainer.py` (95, 231, 232, 237, 238, 247, 250, 266, …)<br>`src/pinn_feynman/tuning/objective.py` (17, 19) ⚠️ generic name |

### `pinn_feynman.config`

[`src/pinn_feynman/config.py`](../src/pinn_feynman/config.py) — YAML configuration files, with inheritance.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `DTYPES` | constant |  | 15 | `src/pinn_feynman/config.py` (48, 50) |
| `merge` | function | `(base, override)` | 18 | `src/pinn_feynman/config.py` (24, 41)<br>`src/pinn_feynman/tuning/space.py` (14, 59) |
| `_YAML_LOADER` | constant |  | 31 | `src/pinn_feynman/config.py` (38) |
| `load_config` | function | `(path)` | 34 | `src/pinn_feynman/config.py` (41)<br>`src/pinn_feynman/data/build.py` (14, 30)<br>`src/pinn_feynman/data/check_interpolation.py` (13, 34)<br>`src/pinn_feynman/devtools/benchmark.py` (17, 45)<br>`src/pinn_feynman/evaluation/predict.py` (10, 89)<br>`src/pinn_feynman/training/queue.py` (49, 75)<br>`src/pinn_feynman/training/run.py` (29, 123, 129)<br>`src/pinn_feynman/tuning/tune.py` (31, 49) |
| `torch_dtype` | function | `(name)` | 45 | `src/pinn_feynman/evaluation/predict.py` (10, 91)<br>`src/pinn_feynman/training/setup.py` (9, 170) |

### `pinn_feynman.data`

[`src/pinn_feynman/data/__init__.py`](../src/pinn_feynman/data/__init__.py) — Reference datasets (boundary, validation, test points) with an on-disk cache.

### `pinn_feynman.data.build`

[`src/pinn_feynman/data/build.py`](../src/pinn_feynman/data/build.py) — Generate (or find in the cache) the datasets of one or more configurations.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `dataset_specs` | function | `(config)` | 19 | `src/pinn_feynman/data/build.py` (30)<br>`src/pinn_feynman/data/check_interpolation.py` (14, 34)<br>`src/pinn_feynman/evaluation/evaluate.py` (18, 151)<br>`src/pinn_feynman/training/setup.py` (10, 57) |
| `unique_specs` | function | `(config_paths, only=None)` | 26 | `src/pinn_feynman/data/build.py` (51) |
| `describe` | function | `(spec)` | 38 | `src/pinn_feynman/data/build.py` (57, 59)<br>`src/pinn_feynman/devtools/codemap.py` (221) |
| `build_all` | function | `(config_paths, workers=1, only=None)` | 49 | `src/pinn_feynman/data/build.py` (72) |
| `main` | function | `(argv=None)` | 66 | `src/pinn_feynman/data/build.py` (76)<br>`src/pinn_feynman/data/check_interpolation.py` (46)<br>`src/pinn_feynman/devtools/benchmark.py` (68)<br>`src/pinn_feynman/devtools/codemap.py` (318)<br>`src/pinn_feynman/devtools/list_configs.py` (97)<br>`src/pinn_feynman/evaluation/evaluate.py` (167)<br>`src/pinn_feynman/systems/analysis.py` (171)<br>`src/pinn_feynman/training/queue.py` (379)<br>`src/pinn_feynman/training/run.py` (212)<br>`src/pinn_feynman/tuning/tune.py` (253) |

### `pinn_feynman.data.cache`

[`src/pinn_feynman/data/cache.py`](../src/pinn_feynman/data/cache.py) — On-disk cache of reference datasets (.npz files in paths.data_dir()).

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `cache_key` | function | `(spec)` | 22 | `src/pinn_feynman/data/cache.py` (30) |
| `cache_path` | function | `(spec, directory=None)` | 27 | `src/pinn_feynman/data/build.py` (15, 52)<br>`src/pinn_feynman/data/cache.py` (78) |
| `save_dataset` | function | `(dataset, path)` | 33 | `src/pinn_feynman/data/cache.py` (82) |
| `load_dataset` | function | `(path)` | 43 | `src/pinn_feynman/data/cache.py` (80) |
| `nodes_spec` | function | `(spec)` | 50 | `src/pinn_feynman/data/cache.py` (69)<br>`src/pinn_feynman/data/check_interpolation.py` (15, 38) |
| `interpolate_reference` | function | `(nodes_dataset, xs)` | 55 | `src/pinn_feynman/data/cache.py` (70)<br>`src/pinn_feynman/data/check_interpolation.py` (15, 20) |
| `build_dataset` | function | `(spec, workers=1, directory=None, progress=False)` | 60 | `src/pinn_feynman/data/cache.py` (81) |
| `get_dataset` | function | `(spec, workers=1, directory=None, progress=False)` | 76 | `src/pinn_feynman/data/__init__.py` (3)<br>`src/pinn_feynman/data/build.py` (15, 61)<br>`src/pinn_feynman/data/cache.py` (69)<br>`src/pinn_feynman/data/check_interpolation.py` (15, 38, 39)<br>`src/pinn_feynman/evaluation/evaluate.py` (19, 151)<br>`src/pinn_feynman/training/setup.py` (11, 58) |

### `pinn_feynman.data.check_interpolation`

[`src/pinn_feynman/data/check_interpolation.py`](../src/pinn_feynman/data/check_interpolation.py) — Check a Chebyshev-interpolated dataset against an exactly computed one.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `interpolation_errors` | function | `(nodes_dataset, exact_dataset)` | 18 | `src/pinn_feynman/data/check_interpolation.py` (41) |
| `main` | function | `(argv=None)` | 28 | `src/pinn_feynman/data/build.py` (76)<br>`src/pinn_feynman/data/check_interpolation.py` (46)<br>`src/pinn_feynman/devtools/benchmark.py` (68)<br>`src/pinn_feynman/devtools/codemap.py` (318)<br>`src/pinn_feynman/devtools/list_configs.py` (97)<br>`src/pinn_feynman/evaluation/evaluate.py` (167)<br>`src/pinn_feynman/systems/analysis.py` (171)<br>`src/pinn_feynman/training/queue.py` (379)<br>`src/pinn_feynman/training/run.py` (212)<br>`src/pinn_feynman/tuning/tune.py` (253) |

### `pinn_feynman.data.dataset`

[`src/pinn_feynman/data/dataset.py`](../src/pinn_feynman/data/dataset.py) — What to generate (DatasetSpec) and what is stored (ReferenceDataset).

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `PARTS` | constant |  | 11 | `src/pinn_feynman/data/dataset.py` (64)<br>`src/pinn_feynman/physics/bottom_up.py` (18, 37, 38)<br>`src/pinn_feynman/training/run.py` (30, 193) |
| `METHODS` | constant |  | 14 | `src/pinn_feynman/data/dataset.py` (36, 37) |
| `DatasetSpec` | class |  | 18 | `src/pinn_feynman/data/__init__.py` (4)<br>`src/pinn_feynman/data/build.py` (16, 22)<br>`src/pinn_feynman/data/cache.py` (11, 46, 52)<br>`src/pinn_feynman/data/dataset.py` (75) |
| `DatasetSpec.from_config` | method | `(cls, problem, domain, entry)` | 45 | `src/pinn_feynman/data/build.py` (22) ⚠️ generic name |
| `DatasetSpec.as_dict` | method | `(self)` | 51 | `src/pinn_feynman/data/cache.py` (19) ⚠️ generic name |
| `select_part` | function | `(values, part)` | 58 | `src/pinn_feynman/data/dataset.py` (91) |
| `ReferenceDataset` | class |  | 68 | `src/pinn_feynman/data/__init__.py` (4)<br>`src/pinn_feynman/data/cache.py` (11, 47, 73) |
| `ReferenceDataset.part` | method | `(self, part)` | 89 | `src/pinn_feynman/data/check_interpolation.py` (22, 24, 41, 42)<br>`src/pinn_feynman/data/dataset.py` (60, 62, 64, 91, 96)<br>`src/pinn_feynman/evaluation/boni_reference.py` (65)<br>`src/pinn_feynman/evaluation/evaluate.py` (34, 38, 45, 47, 53, 54, 64, 66, …)<br>`src/pinn_feynman/evaluation/predict.py` (49, 74)<br>`src/pinn_feynman/physics/bottom_up.py` (22, 37, 38, 39, 90, 106)<br>`src/pinn_feynman/physics/system.py` (144, 158)<br>`src/pinn_feynman/providers/base.py` (81, 82, 84)<br>`src/pinn_feynman/training/queue.py` (76, 147, 173, 269, 271, 272, 273)<br>`src/pinn_feynman/training/run.py` (46, 105, 117, 124, 125, 141, 142, 143, …)<br>`src/pinn_feynman/training/setup.py` (28, 50, 101, 102, 166, 174, 175, 180, …)<br>`src/pinn_feynman/training/trainer.py` (143)<br>`src/pinn_feynman/tuning/objective.py` (32)<br>`src/pinn_feynman/visualization/components.py` (165, 167, 171, 172, 174, 176) ⚠️ generic name |
| `ReferenceDataset.tensors` | method | `(self, part, dtype=torch.float64)` | 93 | `src/pinn_feynman/training/setup.py` (50) ⚠️ generic name |

### `pinn_feynman.data.domains`

[`src/pinn_feynman/data/domains.py`](../src/pinn_feynman/data/domains.py) — Domains of the kinematic variables: an interval (one variable) or the one-mass triangle (two).

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `Interval` | class |  | 36 | `src/pinn_feynman/data/dataset.py` (8, 28)<br>`src/pinn_feynman/data/domains.py` (89, 203, 215) |
| `Interval.lo` | method | `(self)` | 51 | `src/pinn_feynman/data/domains.py` (44, 45, 46, 47, 48, 59, 64, 69, …)<br>`src/pinn_feynman/data/interpolation.py` (13, 15)<br>`src/pinn_feynman/evaluation/evaluate.py` (87, 88)<br>`src/pinn_feynman/reference/nonplanar_three_point.py` (209, 210, 212)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (153, 154, 156)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (72, 76, 92, 93)<br>`src/pinn_feynman/training/recorder.py` (89, 90, 92)<br>`src/pinn_feynman/training/trainer.py` (82, 84)<br>`src/pinn_feynman/visualization/accuracy.py` (15, 16, 120, 124, 133)<br>`src/pinn_feynman/visualization/singularities.py` (106, 107, 111)<br>`src/pinn_feynman/visualization/style.py` (38)<br>`src/pinn_feynman/visualization/tuning.py` (90, 91)<br>`src/pinn_feynman/weighting/distance.py` (139, 140) |
| `Interval.hi` | method | `(self)` | 55 | `src/pinn_feynman/data/domains.py` (43, 44, 45, 46, 47, 48, 59, 64, …)<br>`src/pinn_feynman/data/interpolation.py` (13, 15)<br>`src/pinn_feynman/evaluation/evaluate.py` (87, 88)<br>`src/pinn_feynman/reference/nonplanar_three_point.py` (209, 210, 212)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (153, 154, 156)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (72, 76, 92, 93)<br>`src/pinn_feynman/training/recorder.py` (89, 90, 92)<br>`src/pinn_feynman/training/trainer.py` (82, 84)<br>`src/pinn_feynman/visualization/accuracy.py` (15, 16, 120, 124, 133)<br>`src/pinn_feynman/visualization/singularities.py` (106, 107, 111)<br>`src/pinn_feynman/visualization/style.py` (38)<br>`src/pinn_feynman/visualization/tuning.py` (90, 91)<br>`src/pinn_feynman/weighting/distance.py` (139, 140) |
| `Interval.sample` | method | `(self, n, seed=None)` | 61 | `src/pinn_feynman/data/cache.py` (14, 66)<br>`src/pinn_feynman/data/domains.py` (139)<br>`src/pinn_feynman/data/sampling.py` (22) |
| `Interval.sample_columns` | method | `(self, n, seed=None)` | 66 | `src/pinn_feynman/collocation/sampler.py` (50) |
| `Interval.sample_torch` | method | `(self, n, generator, dtype=torch.float64)` | 71 | `src/pinn_feynman/collocation/r3.py` (55)<br>`src/pinn_feynman/collocation/sampler.py` (123) |
| `Interval.boundary` | method | `(self)` | 77 | `src/pinn_feynman/data/sampling.py` (13)<br>`src/pinn_feynman/training/recorder.py` (235)<br>`src/pinn_feynman/training/setup.py` (174, 176, 190)<br>`src/pinn_feynman/training/trainer.py` (58, 153) |
| `Interval.contains` | method | `(self, points)` | 81 | — |
| `Interval.scaled` | method | `(self, c)` | 86 | `src/pinn_feynman/balancing/relobralo.py` (40, 53)<br>`src/pinn_feynman/training/setup.py` (181) |
| `Interval.as_config` | method | `(self)` | 91 | `src/pinn_feynman/data/dataset.py` (54) |
| `Interval.label` | method | `(self)` | 95 | `src/pinn_feynman/data/build.py` (41)<br>`src/pinn_feynman/data/cache.py` (68, 69)<br>`src/pinn_feynman/data/dataset.py` (42)<br>`src/pinn_feynman/data/generation.py` (56, 57)<br>`src/pinn_feynman/data/sampling.py` (29)<br>`src/pinn_feynman/training/setup.py` (85)<br>`src/pinn_feynman/visualization/accuracy.py` (50, 54, 151, 159, 219, 237, 265, 269, …)<br>`src/pinn_feynman/visualization/components.py` (27, 33, 43, 50, 94, 95, 129, 130, …)<br>`src/pinn_feynman/visualization/methods.py` (40, 46, 48, 68, 69, 70, 72, 73, …)<br>`src/pinn_feynman/visualization/singularities.py` (66, 67, 110, 114, 127, 140)<br>`src/pinn_feynman/visualization/training.py` (62, 64, 80, 86, 89, 134, 135, 142, …)<br>`src/pinn_feynman/visualization/triangle.py` (37, 63, 115, 138, 148, 166, 220, 221, …)<br>`src/pinn_feynman/visualization/tuning.py` (23, 24) |
| `Triangle` | class |  | 100 | `src/pinn_feynman/data/dataset.py` (8, 28)<br>`src/pinn_feynman/data/domains.py` (175, 197, 203) |
| `Triangle.vertices` | method | `(self)` | 119 | `src/pinn_feynman/data/domains.py` (117, 121, 130, 146, 161, 166)<br>`src/pinn_feynman/evaluation/evaluate.py` (114) |
| `Triangle.from_unit` | method | `(self, u)` | 123 | `src/pinn_feynman/data/domains.py` (135) |
| `Triangle.sample` | method | `(self, n, seed=None)` | 133 | `src/pinn_feynman/data/cache.py` (14, 66)<br>`src/pinn_feynman/data/domains.py` (139)<br>`src/pinn_feynman/data/sampling.py` (22) |
| `Triangle.sample_columns` | method | `(self, n, seed=None)` | 137 | `src/pinn_feynman/collocation/sampler.py` (50) |
| `Triangle.sample_torch` | method | `(self, n, generator, dtype=torch.float64)` | 141 | `src/pinn_feynman/collocation/r3.py` (55)<br>`src/pinn_feynman/collocation/sampler.py` (123) |
| `Triangle.grid` | method | `(self, n)` | 149 | `src/pinn_feynman/training/trainer.py` (77)<br>`src/pinn_feynman/weighting/distance.py` (138) |
| `Triangle.boundary` | method | `(self)` | 164 | `src/pinn_feynman/data/sampling.py` (13)<br>`src/pinn_feynman/training/recorder.py` (235)<br>`src/pinn_feynman/training/setup.py` (174, 176, 190)<br>`src/pinn_feynman/training/trainer.py` (58, 153) |
| `Triangle.contains` | method | `(self, points)` | 168 | — |
| `Triangle.scaled` | method | `(self, c)` | 173 | `src/pinn_feynman/balancing/relobralo.py` (40, 53)<br>`src/pinn_feynman/training/setup.py` (181) |
| `Triangle.as_config` | method | `(self)` | 177 | `src/pinn_feynman/data/dataset.py` (54) |
| `Triangle.label` | method | `(self)` | 184 | `src/pinn_feynman/data/build.py` (41)<br>`src/pinn_feynman/data/cache.py` (68, 69)<br>`src/pinn_feynman/data/dataset.py` (42)<br>`src/pinn_feynman/data/generation.py` (56, 57)<br>`src/pinn_feynman/data/sampling.py` (29)<br>`src/pinn_feynman/training/setup.py` (85)<br>`src/pinn_feynman/visualization/accuracy.py` (50, 54, 151, 159, 219, 237, 265, 269, …)<br>`src/pinn_feynman/visualization/components.py` (27, 33, 43, 50, 94, 95, 129, 130, …)<br>`src/pinn_feynman/visualization/methods.py` (40, 46, 48, 68, 69, 70, 72, 73, …)<br>`src/pinn_feynman/visualization/singularities.py` (66, 67, 110, 114, 127, 140)<br>`src/pinn_feynman/visualization/training.py` (62, 64, 80, 86, 89, 134, 135, 142, …)<br>`src/pinn_feynman/visualization/triangle.py` (37, 63, 115, 138, 148, 166, 220, 221, …)<br>`src/pinn_feynman/visualization/tuning.py` (23, 24) |
| `DOMAINS` | constant |  | 197 | `src/pinn_feynman/data/domains.py` (209, 211) |
| `as_domain` | function | `(value)` | 200 | `src/pinn_feynman/collocation/r3.py` (11, 43)<br>`src/pinn_feynman/collocation/sampler.py` (17, 46, 113)<br>`src/pinn_feynman/data/dataset.py` (8, 35, 47)<br>`src/pinn_feynman/data/sampling.py` (6, 13, 22, 27)<br>`src/pinn_feynman/training/setup.py` (12, 83, 181) |

### `pinn_feynman.data.generation`

[`src/pinn_feynman/data/generation.py`](../src/pinn_feynman/data/generation.py) — Exact reference values on a grid of points, computed in parallel.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `GENERATORS` | constant |  | 19 | `src/pinn_feynman/data/generation.py` (31) |
| `COMPUTE_OPTIONS` | constant |  | 24 | `src/pinn_feynman/data/generation.py` (48)<br>`src/pinn_feynman/problems/_registry.py` (71)<br>`src/pinn_feynman/reference/box_massless.py` (37)<br>`src/pinn_feynman/reference/box_one_mass.py` (61)<br>`src/pinn_feynman/reference/nonplanar_three_point.py` (44)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (79)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (45) |
| `DEFAULT_COMPUTE_OPTIONS` | constant |  | 25 | `src/pinn_feynman/data/generation.py` (48) |
| `generator_info` | function | `(problem)` | 28 | `src/pinn_feynman/data/cache.py` (12, 18)<br>`src/pinn_feynman/data/generation.py` (37)<br>`src/pinn_feynman/evaluation/predict.py` (11, 27)<br>`src/pinn_feynman/physics/system.py` (146, 156) |
| `compute_reference` | function | `(problem, xs, workers=1, chunk_size=None, progress=False)` | 41 | `src/pinn_feynman/data/cache.py` (12, 72) |

### `pinn_feynman.data.interpolation`

[`src/pinn_feynman/data/interpolation.py`](../src/pinn_feynman/data/interpolation.py) — Chebyshev interpolation on an interval (for the large test set).

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `chebyshev_nodes` | function | `(domain, n)` | 11 | `src/pinn_feynman/data/sampling.py` (7, 30) |
| `barycentric_interpolate` | function | `(nodes, values, xs)` | 25 | `src/pinn_feynman/data/cache.py` (13, 57) |

### `pinn_feynman.data.sampling`

[`src/pinn_feynman/data/sampling.py`](../src/pinn_feynman/data/sampling.py) — Point samplers on a domain (`data.domains`: interval or triangle), selected by name.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `boundary_points` | function | `(domain, n=2, seed=None)` | 10 | `src/pinn_feynman/data/sampling.py` (34) |
| `uniform_points` | function | `(domain, n, seed)` | 19 | `src/pinn_feynman/data/sampling.py` (35) |
| `chebyshev_points` | function | `(domain, n, seed=None)` | 25 | `src/pinn_feynman/data/sampling.py` (36) |
| `SAMPLERS` | constant |  | 33 | `src/pinn_feynman/data/sampling.py` (43, 45)<br>`src/pinn_feynman/tuning/tune.py` (40, 54) |
| `sample` | function | `(name, domain, n, seed)` | 40 | `src/pinn_feynman/data/cache.py` (14, 66)<br>`src/pinn_feynman/data/domains.py` (139)<br>`src/pinn_feynman/data/sampling.py` (22) |

### `pinn_feynman.devtools`

[`src/pinn_feynman/devtools/__init__.py`](../src/pinn_feynman/devtools/__init__.py) — Developer tools: code map with usage index (docs/code_map.md), step timing, configuration tree.

### `pinn_feynman.devtools.benchmark`

[`src/pinn_feynman/devtools/benchmark.py`](../src/pinn_feynman/devtools/benchmark.py) — Time training steps of a configuration on a device (CPU/GPU comparison).

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `time_steps` | function | `(trainer, steps, warmup=20)` | 21 | `src/pinn_feynman/devtools/benchmark.py` (56) |
| `main` | function | `(argv=None)` | 35 | `src/pinn_feynman/data/build.py` (76)<br>`src/pinn_feynman/data/check_interpolation.py` (46)<br>`src/pinn_feynman/devtools/benchmark.py` (68)<br>`src/pinn_feynman/devtools/codemap.py` (318)<br>`src/pinn_feynman/devtools/list_configs.py` (97)<br>`src/pinn_feynman/evaluation/evaluate.py` (167)<br>`src/pinn_feynman/systems/analysis.py` (171)<br>`src/pinn_feynman/training/queue.py` (379)<br>`src/pinn_feynman/training/run.py` (212)<br>`src/pinn_feynman/tuning/tune.py` (253) |

### `pinn_feynman.devtools.codemap`

[`src/pinn_feynman/devtools/codemap.py`](../src/pinn_feynman/devtools/codemap.py) — Generate docs/code_map.md: files, public symbols, where each symbol is used, imports.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `PACKAGE` | constant |  | 20 | `src/pinn_feynman/devtools/codemap.py` (133, 137, 157) |
| `OUTPUT` | constant |  | 21 | `src/pinn_feynman/devtools/codemap.py` (213, 303, 307, 309, 313) |
| `SKIP_DIRS` | constant |  | 23 | `src/pinn_feynman/devtools/codemap.py` (62, 66) |
| `SKIP_TOP` | constant |  | 25 | `src/pinn_feynman/devtools/codemap.py` (66) |
| `SKIP_FILES` | constant |  | 26 | `src/pinn_feynman/devtools/codemap.py` (211) |
| `LOCAL_SKIP` | constant |  | 28 | `src/pinn_feynman/devtools/codemap.py` (200) |
| `FOLDED_TOP` | constant |  | 30 | `src/pinn_feynman/devtools/codemap.py` (216, 219) |
| `LISTED_SUFFIXES` | constant |  | 32 | `src/pinn_feynman/devtools/codemap.py` (213) |
| `GENERIC_NAMES` | constant |  | 34 | `src/pinn_feynman/devtools/codemap.py` (238) |
| `Symbol` | class |  | 39 | `src/pinn_feynman/devtools/codemap.py` (82, 85, 89, 94)<br>`src/pinn_feynman/physics/system.py` (71, 152)<br>`src/pinn_feynman/systems/analysis.py` (80)<br>`src/pinn_feynman/systems/blocks.py` (22)<br>`src/pinn_feynman/systems/box_massless_canonical.py` (23)<br>`src/pinn_feynman/systems/nonplanar_three_point.py` (26, 27)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (17, 18)<br>`src/pinn_feynman/visualization/singularities.py` (81)<br>`src/pinn_feynman/weighting/distance.py` (50) |
| `first_line` | function | `(text)` | 50 | `src/pinn_feynman/devtools/codemap.py` (83, 86, 90, 161, 186) |
| `module_name` | function | `(path, src_root)` | 54 | `src/pinn_feynman/devtools/codemap.py` (158)<br>`src/pinn_feynman/problems/_registry.py` (101, 103, 105, 110, 112) |
| `python_files` | function | `(root)` | 61 | `src/pinn_feynman/devtools/codemap.py` (157, 166) |
| `parse` | function | `(path)` | 69 | `src/pinn_feynman/devtools/codemap.py` (70, 159, 167, 186) |
| `public_symbols` | function | `(module, path, tree)` | 77 | `src/pinn_feynman/devtools/codemap.py` (162) |
| `identifier_occurrences` | function | `(path, tree)` | 98 | `src/pinn_feynman/devtools/codemap.py` (168) |
| `resolve_import` | function | `(module, node)` | 117 | `src/pinn_feynman/devtools/codemap.py` (130) |
| `import_edges` | function | `(module, path, tree)` | 125 | `src/pinn_feynman/devtools/codemap.py` (163) |
| `CodeMap` | class |  | 143 | `src/pinn_feynman/devtools/codemap.py` (175) |
| `build` | function | `(root=None)` | 152 | `src/pinn_feynman/devtools/codemap.py` (302)<br>`src/pinn_feynman/weighting/factory.py` (32) |
| `describe` | function | `(path)` | 178 | `src/pinn_feynman/data/build.py` (57, 59)<br>`src/pinn_feynman/devtools/codemap.py` (221) |
| `local_skips` | function | `(root)` | 198 | `src/pinn_feynman/devtools/codemap.py` (207) |
| `list_files` | function | `(root)` | 206 | `src/pinn_feynman/devtools/codemap.py` (175) |
| `render` | function | `(cmap)` | 242 | `src/pinn_feynman/devtools/codemap.py` (302) |
| `untested_symbols` | function | `(cmap)` | 290 | `src/pinn_feynman/devtools/codemap.py` (285) |
| `main` | function | `(argv=None)` | 297 | `src/pinn_feynman/data/build.py` (76)<br>`src/pinn_feynman/data/check_interpolation.py` (46)<br>`src/pinn_feynman/devtools/benchmark.py` (68)<br>`src/pinn_feynman/devtools/codemap.py` (318)<br>`src/pinn_feynman/devtools/list_configs.py` (97)<br>`src/pinn_feynman/evaluation/evaluate.py` (167)<br>`src/pinn_feynman/systems/analysis.py` (171)<br>`src/pinn_feynman/training/queue.py` (379)<br>`src/pinn_feynman/training/run.py` (212)<br>`src/pinn_feynman/tuning/tune.py` (253) |

### `pinn_feynman.devtools.list_configs`

[`src/pinn_feynman/devtools/list_configs.py`](../src/pinn_feynman/devtools/list_configs.py) — Inheritance tree of the configuration files, with what each one changes with respect to its base.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `read_raw` | function | `(path)` | 18 | `src/pinn_feynman/devtools/list_configs.py` (41, 46) |
| `flatten` | function | `(node, prefix='')` | 24 | `src/pinn_feynman/devtools/list_configs.py` (30, 48)<br>`src/pinn_feynman/tuning/tune.py` (110) |
| `config_files` | function | `(folder)` | 34 | `src/pinn_feynman/devtools/list_configs.py` (54, 63) |
| `base_of` | function | `(path)` | 39 | `src/pinn_feynman/devtools/list_configs.py` (55) |
| `overrides` | function | `(path)` | 44 | `src/pinn_feynman/devtools/list_configs.py` (72) |
| `children_map` | function | `(folder)` | 51 | `src/pinn_feynman/devtools/list_configs.py` (62) |
| `tree_lines` | function | `(folder, with_diff=True)` | 59 | `src/pinn_feynman/devtools/list_configs.py` (92) |
| `main` | function | `(argv=None)` | 83 | `src/pinn_feynman/data/build.py` (76)<br>`src/pinn_feynman/data/check_interpolation.py` (46)<br>`src/pinn_feynman/devtools/benchmark.py` (68)<br>`src/pinn_feynman/devtools/codemap.py` (318)<br>`src/pinn_feynman/devtools/list_configs.py` (97)<br>`src/pinn_feynman/evaluation/evaluate.py` (167)<br>`src/pinn_feynman/systems/analysis.py` (171)<br>`src/pinn_feynman/training/queue.py` (379)<br>`src/pinn_feynman/training/run.py` (212)<br>`src/pinn_feynman/tuning/tune.py` (253) |

### `pinn_feynman.evaluation`

[`src/pinn_feynman/evaluation/__init__.py`](../src/pinn_feynman/evaluation/__init__.py) — Evaluation of trained networks: errors (Boni eqs. 6.18-6.19), figures, comparison with the thesis.

### `pinn_feynman.evaluation.boni_reference`

[`src/pinn_feynman/evaluation/boni_reference.py`](../src/pinn_feynman/evaluation/boni_reference.py) — Results reported in Boni's thesis: massless box (sec. 6.2.1.1, printed pp. 102-103), one-mass box

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `BONI_MASSLESS_BOX` | constant |  | 10 | — |
| `BONI_ONE_MASS_BOX` | constant |  | 30 | — |
| `BONI_NONPLANAR` | constant |  | 46 | — |
| `BONI` | constant |  | 60 | `src/pinn_feynman/evaluation/boni_reference.py` (65) |
| `boni_part` | function | `(problem, part)` | 63 | `src/pinn_feynman/evaluation/evaluate.py` (22, 38, 47) |

### `pinn_feynman.evaluation.evaluate`

[`src/pinn_feynman/evaluation/evaluate.py`](../src/pinn_feynman/evaluation/evaluate.py) — Evaluate trained runs on the test set and draw the figures of the thesis.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `evaluate_run` | function | `(run, dataset)` | 27 | `src/pinn_feynman/evaluation/evaluate.py` (152) |
| `draw_figures` | function | `(run, dataset, prediction, target, abs_err, rel_err, out_dir=None)` | 43 | `src/pinn_feynman/evaluation/evaluate.py` (155) |
| `read_history` | function | `(path)` | 132 | `src/pinn_feynman/evaluation/evaluate.py` (48) |
| `main` | function | `(argv=None)` | 141 | `src/pinn_feynman/data/build.py` (76)<br>`src/pinn_feynman/data/check_interpolation.py` (46)<br>`src/pinn_feynman/devtools/benchmark.py` (68)<br>`src/pinn_feynman/devtools/codemap.py` (318)<br>`src/pinn_feynman/devtools/list_configs.py` (97)<br>`src/pinn_feynman/evaluation/evaluate.py` (167)<br>`src/pinn_feynman/systems/analysis.py` (171)<br>`src/pinn_feynman/training/queue.py` (379)<br>`src/pinn_feynman/training/run.py` (212)<br>`src/pinn_feynman/tuning/tune.py` (253) |

### `pinn_feynman.evaluation.metrics`

[`src/pinn_feynman/evaluation/metrics.py`](../src/pinn_feynman/evaluation/metrics.py) — Error metrics of Boni, thesis eqs. (6.18)-(6.19).

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `ZERO_THRESHOLD` | constant |  | 11 | `src/pinn_feynman/evaluation/metrics.py` (27) |
| `absolute_error` | function | `(prediction, target)` | 21 | `src/pinn_feynman/evaluation/__init__.py` (3)<br>`src/pinn_feynman/evaluation/evaluate.py` (23, 31) |
| `relative_error` | function | `(prediction, target, threshold=ZERO_THRESHOLD)` | 27 | `src/pinn_feynman/evaluation/__init__.py` (3)<br>`src/pinn_feynman/evaluation/evaluate.py` (23, 32) |
| `summarize` | function | `(errors, orders)` | 45 | `src/pinn_feynman/evaluation/__init__.py` (3)<br>`src/pinn_feynman/evaluation/evaluate.py` (23, 35, 36) |

### `pinn_feynman.evaluation.plots`

[`src/pinn_feynman/evaluation/plots.py`](../src/pinn_feynman/evaluation/plots.py) — Re-export of the figures of `pinn_feynman.visualization` used by `evaluate.py`.

### `pinn_feynman.evaluation.predict`

[`src/pinn_feynman/evaluation/predict.py`](../src/pinn_feynman/evaluation/predict.py) — Reload a training run and evaluate its network.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `problem_shape` | function | `(problem)` | 17 | `src/pinn_feynman/evaluation/predict.py` (39, 79, 83, 90)<br>`src/pinn_feynman/training/setup.py` (113, 117) |
| `require_one_variable` | function | `(config, what)` | 36 | — |
| `Run` | class |  | 46 | `src/pinn_feynman/evaluation/predict.py` (96) |
| `Run.input_scale` | method | `(self)` | 54 | `src/pinn_feynman/evaluation/evaluate.py` (30, 70, 88)<br>`src/pinn_feynman/evaluation/predict.py` (106)<br>`src/pinn_feynman/physics/bottom_up.py` (42, 52, 87, 90)<br>`src/pinn_feynman/physics/system.py` (53, 63, 71, 84, 94, 95, 99, 102, …)<br>`src/pinn_feynman/training/recorder.py` (129, 462)<br>`src/pinn_feynman/training/setup.py` (53)<br>`src/pinn_feynman/weighting/distance.py` (155, 159, 161, 168)<br>`src/pinn_feynman/weighting/factory.py` (32) |
| `Run.problem` | method | `(self)` | 58 | `src/pinn_feynman/data/cache.py` (18, 30, 52, 72)<br>`src/pinn_feynman/data/dataset.py` (26, 47)<br>`src/pinn_feynman/data/generation.py` (19, 20, 24, 31, 33, 37, 48, 54)<br>`src/pinn_feynman/devtools/list_configs.py` (89)<br>`src/pinn_feynman/evaluation/boni_reference.py` (60, 65)<br>`src/pinn_feynman/evaluation/evaluate.py` (38, 46, 47, 88, 129)<br>`src/pinn_feynman/evaluation/predict.py` (23, 24, 25, 26, 30, 33, 38, 39, …)<br>`src/pinn_feynman/paths.py` (44, 49)<br>`src/pinn_feynman/physics/system.py` (24, 141, 142, 150, 153, 156, 157)<br>`src/pinn_feynman/problems/_registry.py` (108, 109, 111, 112, 113, 114, 118, 119, …)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (49, 58, 62)<br>`src/pinn_feynman/systems/singular_kinds.py` (52, 54, 60, 62)<br>`src/pinn_feynman/training/queue.py` (173, 240)<br>`src/pinn_feynman/training/setup.py` (131, 132, 137, 140)<br>`src/pinn_feynman/tuning/publish.py` (30, 71)<br>`src/pinn_feynman/visualization/singularities.py` (35, 40, 51, 54, 57, 59, 76, 80, …)<br>`src/pinn_feynman/weighting/distance.py` (40, 41) |
| `Run.system_problem` | method | `(self)` | 63 | `src/pinn_feynman/evaluation/predict.py` (70, 79, 83) |
| `Run.columns` | method | `(self)` | 68 | `src/pinn_feynman/evaluation/evaluate.py` (96, 97)<br>`src/pinn_feynman/evaluation/predict.py` (70, 75)<br>`src/pinn_feynman/physics/bottom_up.py` (47, 48)<br>`src/pinn_feynman/providers/base.py` (28, 38, 39, 53, 64, 65, 66, 67, …)<br>`src/pinn_feynman/training/setup.py` (51, 52, 173, 174, 175)<br>`src/pinn_feynman/visualization/data_io.py` (85, 86)<br>`src/pinn_feynman/visualization/triangle.py` (102, 137, 147, 153, 166)<br>`src/pinn_feynman/visualization/tuning.py` (81, 91, 98)<br>`src/pinn_feynman/weighting/distance.py` (41, 58, 72, 75, 87, 90) |
| `Run.target` | method | `(self, dataset)` | 72 | `src/pinn_feynman/devtools/codemap.py` (92, 93, 94, 130, 132, 133, 134, 303, …)<br>`src/pinn_feynman/evaluation/evaluate.py` (29, 31, 32, 40, 73, 152, 155)<br>`src/pinn_feynman/evaluation/metrics.py` (16, 17, 23, 24, 29, 30, 31, 32, …)<br>`src/pinn_feynman/physics/losses.py` (17)<br>`src/pinn_feynman/training/recorder.py` (406, 407, 409, 411, 413, 415)<br>`src/pinn_feynman/tuning/publish.py` (57, 58, 59, 60, 61, 62, 63, 64)<br>`src/pinn_feynman/tuning/tune.py` (76, 77, 131, 171, 173, 187, 209, 222, …)<br>`src/pinn_feynman/visualization/accuracy.py` (67, 75) |
| `Run.orders` | method | `(self)` | 78 | `src/pinn_feynman/evaluation/evaluate.py` (35, 36, 63, 66, 68, 73, 77, 79, …)<br>`src/pinn_feynman/evaluation/metrics.py` (55, 58)<br>`src/pinn_feynman/evaluation/predict.py` (90, 91)<br>`src/pinn_feynman/models/factory.py` (22, 24, 25, 27, 30, 51, 52, 53)<br>`src/pinn_feynman/physics/bottom_up.py` (40, 59, 60, 71, 72, 76, 95, 107, …)<br>`src/pinn_feynman/physics/coefficients.py` (19, 23, 27, 30)<br>`src/pinn_feynman/physics/losses.py` (46)<br>`src/pinn_feynman/physics/system.py` (59, 60, 62, 70, 71, 76, 77, 78, …)<br>`src/pinn_feynman/providers/base.py` (29, 33, 34, 38, 40, 53, 69, 70, …)<br>`src/pinn_feynman/reference/box_massless.py` (93, 101, 103, 113)<br>`src/pinn_feynman/reference/box_one_mass.py` (178, 180, 305)<br>`src/pinn_feynman/reference/nonplanar_three_point.py` (238, 239)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (164, 165)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (102, 103)<br>`src/pinn_feynman/systems/blocks.py` (40, 42, 44, 114, 115, 232, 233, 239, …)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (63, 64, 67)<br>`src/pinn_feynman/training/recorder.py` (300)<br>`src/pinn_feynman/training/setup.py` (117, 118, 120, 125, 168, 179)<br>`src/pinn_feynman/training/trainer.py` (110, 114, 118, 122, 126, 130, 137, 209, …)<br>`src/pinn_feynman/visualization/accuracy.py` (22, 24, 66, 97, 125, 152, 178, 179, …)<br>`src/pinn_feynman/visualization/components.py` (293, 298, 305)<br>`src/pinn_feynman/visualization/triangle.py` (33, 42) |
| `Run.variables` | method | `(self)` | 82 | `src/pinn_feynman/evaluation/predict.py` (39, 40, 41)<br>`src/pinn_feynman/physics/bottom_up.py` (42, 52)<br>`src/pinn_feynman/physics/system.py` (62, 69, 70, 71, 82, 83, 97, 112, …)<br>`src/pinn_feynman/training/setup.py` (81, 82, 85, 89)<br>`src/pinn_feynman/visualization/singularities.py` (77, 78, 81) |
| `load_run` | function | `(directory, which='best')` | 86 | `src/pinn_feynman/evaluation/__init__.py` (4)<br>`src/pinn_feynman/evaluation/evaluate.py` (24, 150) |
| `predict` | function | `(model, x, input_scale, batch=20000)` | 100 | `src/pinn_feynman/evaluation/__init__.py` (4)<br>`src/pinn_feynman/evaluation/evaluate.py` (24, 30) |

### `pinn_feynman.models`

[`src/pinn_feynman/models/__init__.py`](../src/pinn_feynman/models/__init__.py) — Neural networks for the master integrals.

### `pinn_feynman.models.factory`

[`src/pinn_feynman/models/factory.py`](../src/pinn_feynman/models/factory.py) — Model construction from the `model` section of a configuration.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `MODELS` | constant |  | 9 | `src/pinn_feynman/models/factory.py` (43) |
| `ACTIVATIONS` | constant |  | 10 | `src/pinn_feynman/models/factory.py` (44) |
| `active_positions` | function | `(active_orders, orders)` | 20 | `src/pinn_feynman/models/factory.py` (53)<br>`src/pinn_feynman/training/setup.py` (114, 118) |
| `build_model` | function | `(model_config, n_masters, n_orders, dtype=torch.float64, seed=None, orders=None)` | 33 | `src/pinn_feynman/evaluation/predict.py` (12, 91)<br>`src/pinn_feynman/models/__init__.py` (3)<br>`src/pinn_feynman/training/setup.py` (13, 178) |
| `count_parameters` | function | `(model)` | 59 | `src/pinn_feynman/models/__init__.py` (3) |

### `pinn_feynman.models.mlp`

[`src/pinn_feynman/models/mlp.py`](../src/pinn_feynman/models/mlp.py) — Fully connected network: x -> Laurent coefficients of the master integrals.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `ACTIVATION_DERIVATIVES` | constant |  | 26 | `src/pinn_feynman/models/mlp.py` (67, 77) |
| `MLP` | class |  | 29 | `src/pinn_feynman/models/__init__.py` (4)<br>`src/pinn_feynman/models/factory.py` (6, 9) |
| `MLP.forward` | method | `(self, z)` | 49 | — |
| `MLP.forward_with_derivative` | method | `(self, z)` | 53 | `src/pinn_feynman/models/order_mask.py` (40)<br>`src/pinn_feynman/physics/derivatives.py` (18) |

### `pinn_feynman.models.order_mask`

[`src/pinn_feynman/models/order_mask.py`](../src/pinn_feynman/models/order_mask.py) — Output mask on the orders in eps: the network computes only the active orders, the others are exact zeros.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `ActiveOrders` | class |  | 9 | `src/pinn_feynman/models/__init__.py` (5)<br>`src/pinn_feynman/models/order_mask.py` (60) |
| `ActiveOrders.forward` | method | `(self, z)` | 34 | — |
| `ActiveOrders.forward_with_derivative` | method | `(self, z)` | 38 | `src/pinn_feynman/models/order_mask.py` (40)<br>`src/pinn_feynman/physics/derivatives.py` (18) |
| `mask_orders` | function | `(mlp, active)` | 44 | `src/pinn_feynman/models/factory.py` (7, 56) |

### `pinn_feynman.paths`

[`src/pinn_feynman/paths.py`](../src/pinn_feynman/paths.py) — Project locations, derived from the package position and the storage root (no hard-coded paths).

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `DATA_DIR_ENV` | constant |  | 14 | `src/pinn_feynman/paths.py` (24) |
| `project_root` | function | `()` | 17 | `src/pinn_feynman/devtools/codemap.py` (18, 153, 301)<br>`src/pinn_feynman/paths.py` (35)<br>`src/pinn_feynman/tuning/publish.py` (17, 57) |
| `data_dir` | function | `()` | 22 | `src/pinn_feynman/data/cache.py` (10, 29)<br>`src/pinn_feynman/runtime.py` (58, 69) |
| `runs_dir` | function | `()` | 28 | `src/pinn_feynman/runtime.py` (58, 70)<br>`src/pinn_feynman/training/queue.py` (50, 176)<br>`src/pinn_feynman/training/run.py` (31, 44)<br>`src/pinn_feynman/tuning/tune.py` (32, 149) |
| `configs_dir` | function | `()` | 33 | `src/pinn_feynman/devtools/list_configs.py` (15, 88)<br>`src/pinn_feynman/paths.py` (44)<br>`src/pinn_feynman/training/queue.py` (50, 163, 197, 240) |
| `CONFIG_FOLDERS` | constant |  | 39 | `src/pinn_feynman/paths.py` (44) |
| `config_folder` | function | `(problem)` | 42 | `src/pinn_feynman/paths.py` (49) |
| `config_file` | function | `(problem, name='baseline')` | 47 | — |

### `pinn_feynman.physics`

[`src/pinn_feynman/physics/__init__.py`](../src/pinn_feynman/physics/__init__.py) — Differential-equation side of the PINN: connection matrix, derivatives, losses.

### `pinn_feynman.physics.bottom_up`

[`src/pinn_feynman/physics/bottom_up.py`](../src/pinn_feynman/physics/bottom_up.py) — Bottom-up system: the network learns the unknowns U, the knowns K come from a provider.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `PARTS` | constant |  | 18 | `src/pinn_feynman/data/dataset.py` (11, 64)<br>`src/pinn_feynman/physics/bottom_up.py` (37, 38)<br>`src/pinn_feynman/training/run.py` (30, 193) |
| `BottomUpSystem` | class |  | 25 | `src/pinn_feynman/physics/system.py` (148, 158) |
| `BottomUpSystem.matrices` | method | `(self, z)` | 79 | `src/pinn_feynman/physics/system.py` (107) ⚠️ generic name |
| `BottomUpSystem.coefficients` | method | `(self, z)` | 82 | `src/pinn_feynman/physics/losses.py` (45)<br>`src/pinn_feynman/systems/blocks.py` (37, 38, 39)<br>`src/pinn_feynman/training/recorder.py` (171, 202)<br>`src/pinn_feynman/training/trainer.py` (79, 86, 91, 104) |
| `BottomUpSystem.forcing` | method | `(self, z)` | 85 | `src/pinn_feynman/physics/bottom_up.py` (83)<br>`src/pinn_feynman/physics/coefficients.py` (20, 24, 30)<br>`src/pinn_feynman/physics/losses.py` (34, 35, 46)<br>`src/pinn_feynman/providers/nonplanar_three_point.py` (15, 18)<br>`src/pinn_feynman/reference/nonplanar_three_point.py` (163)<br>`src/pinn_feynman/training/recorder.py` (301)<br>`src/pinn_feynman/training/trainer.py` (137, 209, 222) |

### `pinn_feynman.physics.coefficients`

[`src/pinn_feynman/physics/coefficients.py`](../src/pinn_feynman/physics/coefficients.py) — The coefficients of the cascade residual at a set of points: A_0..A_p and an optional forcing term.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `ResidualCoefficients` | class |  | 11 | `src/pinn_feynman/physics/__init__.py` (4)<br>`src/pinn_feynman/physics/bottom_up.py` (16, 83)<br>`src/pinn_feynman/physics/coefficients.py` (23)<br>`src/pinn_feynman/physics/system.py` (21, 120) |
| `ResidualCoefficients.residual` | method | `(self, y, dy)` | 29 | `src/pinn_feynman/physics/losses.py` (30, 33, 35, 36, 51, 57, 58)<br>`src/pinn_feynman/training/trainer.py` (143, 154, 156, 168) |

### `pinn_feynman.physics.derivatives`

[`src/pinn_feynman/physics/derivatives.py`](../src/pinn_feynman/physics/derivatives.py) — Derivatives of the network output with respect to its inputs, one per direction.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `value_and_derivative` | function | `(model, z, fast=True)` | 6 | `src/pinn_feynman/physics/__init__.py` (3)<br>`src/pinn_feynman/physics/losses.py` (12, 44)<br>`src/pinn_feynman/training/recorder.py` (69, 299)<br>`src/pinn_feynman/training/trainer.py` (34, 135, 207, 220) |

### `pinn_feynman.physics.losses`

[`src/pinn_feynman/physics/losses.py`](../src/pinn_feynman/physics/losses.py) — Training losses of Boni's PINN (thesis sec. 6.1.2).

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `data_loss` | function | `(prediction, target)` | 15 | `src/pinn_feynman/physics/__init__.py` (5)<br>`src/pinn_feynman/training/recorder.py` (70, 235)<br>`src/pinn_feynman/training/trainer.py` (35, 153, 215) |
| `cascade_residual` | function | `(y, dy, a0, a1, *higher, forcing=None)` | 20 | `src/pinn_feynman/physics/__init__.py` (5)<br>`src/pinn_feynman/physics/coefficients.py` (7, 30)<br>`src/pinn_feynman/physics/losses.py` (46)<br>`src/pinn_feynman/training/recorder.py` (70, 300)<br>`src/pinn_feynman/training/trainer.py` (35, 137, 209, 222) |
| `physics_loss` | function | `(model, system, z)` | 39 | `src/pinn_feynman/physics/__init__.py` (5)<br>`src/pinn_feynman/training/recorder.py` (70, 234, 239) |
| `residual_norm` | function | `(residual)` | 49 | `src/pinn_feynman/physics/__init__.py` (5)<br>`src/pinn_feynman/training/recorder.py` (70, 300)<br>`src/pinn_feynman/training/trainer.py` (35, 143, 156, 209) |
| `weighted_physics_loss` | function | `(residual, weights=None)` | 54 | `src/pinn_feynman/physics/__init__.py` (5)<br>`src/pinn_feynman/training/trainer.py` (35, 168) |

### `pinn_feynman.physics.propagation`

[`src/pinn_feynman/physics/propagation.py`](../src/pinn_feynman/physics/propagation.py) — Propagation-failure statistics of the physics residual (Daw et al., arXiv:2207.02338, sec. 3, App. H).

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `propagation_stats` | function | `(norms)` | 12 | `src/pinn_feynman/physics/__init__.py` (6)<br>`src/pinn_feynman/training/trainer.py` (36, 210) |

### `pinn_feynman.physics.system`

[`src/pinn_feynman/physics/system.py`](../src/pinn_feynman/physics/system.py) — Connection matrices A = sum_j eps^j A_j as torch tensors, in the network's input variables.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `SYSTEMS` | constant |  | 24 | `src/pinn_feynman/evaluation/evaluate.py` (20, 46)<br>`src/pinn_feynman/evaluation/predict.py` (13, 23, 26)<br>`src/pinn_feynman/physics/__init__.py` (7)<br>`src/pinn_feynman/physics/system.py` (125)<br>`src/pinn_feynman/training/setup.py` (115, 119) |
| `TorchSystem` | class |  | 45 | `src/pinn_feynman/physics/__init__.py` (7)<br>`src/pinn_feynman/physics/system.py` (143, 154, 155) |
| `TorchSystem.from_orders` | method | `(cls, orders, symbols, input_scale=1.0, variables=None)` | 56 | `src/pinn_feynman/physics/system.py` (71, 154, 155) |
| `TorchSystem.from_module` | method | `(cls, module, input_scale=1.0)` | 67 | `src/pinn_feynman/physics/system.py` (143) |
| `TorchSystem.matrices` | method | `(self, z)` | 88 | `src/pinn_feynman/physics/system.py` (107) ⚠️ generic name |
| `TorchSystem.coefficient_orders` | method | `(self, z)` | 104 | `src/pinn_feynman/physics/bottom_up.py` (83, 94)<br>`src/pinn_feynman/physics/system.py` (120) |
| `TorchSystem.coefficients` | method | `(self, z)` | 118 | `src/pinn_feynman/physics/losses.py` (45)<br>`src/pinn_feynman/systems/blocks.py` (37, 38, 39)<br>`src/pinn_feynman/training/recorder.py` (171, 202)<br>`src/pinn_feynman/training/trainer.py` (79, 86, 91, 104) |
| `system_variables` | function | `(name)` | 130 | `src/pinn_feynman/evaluation/predict.py` (13, 33)<br>`src/pinn_feynman/physics/__init__.py` (7)<br>`src/pinn_feynman/training/setup.py` (14, 81) |
| `build_system` | function | `(name, input_scale=1.0, part=None)` | 135 | `src/pinn_feynman/physics/__init__.py` (7)<br>`src/pinn_feynman/training/setup.py` (14, 180) |

### `pinn_feynman.problems`

[`src/pinn_feynman/problems/__init__.py`](../src/pinn_feynman/problems/__init__.py) — The diagrams the pipeline can train on: one file per problem, found by name (`get_problem`).

### `pinn_feynman.problems._registry`

[`src/pinn_feynman/problems/_registry.py`](../src/pinn_feynman/problems/_registry.py) — The registry of problems: finds a problem by name, declares none.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `UnknownProblem` | class |  | 25 | `src/pinn_feynman/problems/__init__.py` (7)<br>`src/pinn_feynman/problems/_registry.py` (98, 156) |
| `Problem` | class |  | 30 | `src/pinn_feynman/problems/__init__.py` (7)<br>`src/pinn_feynman/problems/_registry.py` (109)<br>`src/pinn_feynman/problems/box_massless.py` (3, 5)<br>`src/pinn_feynman/problems/box_massless_bu.py` (3, 5)<br>`src/pinn_feynman/problems/box_one_mass.py` (3, 5)<br>`src/pinn_feynman/problems/box_one_mass_bu.py` (3, 5)<br>`src/pinn_feynman/problems/nonplanar_three_point.py` (9, 11)<br>`src/pinn_feynman/problems/nonplanar_three_point_bu.py` (5, 7)<br>`src/pinn_feynman/problems/sunrise_equal_mass_positive.py` (6, 8)<br>`src/pinn_feynman/problems/sunrise_equal_mass_positive_bu.py` (4, 6) |
| `Problem.system_module` | method | `(self)` | 58 | `src/pinn_feynman/physics/system.py` (24, 150)<br>`src/pinn_feynman/weighting/distance.py` (41) |
| `Problem.reference_module` | method | `(self)` | 61 | `src/pinn_feynman/evaluation/predict.py` (28)<br>`src/pinn_feynman/problems/_registry.py` (65, 68, 71) |
| `Problem.generator` | method | `(self)` | 64 | `src/pinn_feynman/balancing/relobralo.py` (30, 32, 61, 84, 89)<br>`src/pinn_feynman/collocation/r3.py` (46, 48, 55, 78, 119, 123)<br>`src/pinn_feynman/collocation/sampler.py` (56, 58, 67, 70, 83, 85, 116, 118, …)<br>`src/pinn_feynman/data/domains.py` (74, 144)<br>`src/pinn_feynman/data/generation.py` (19, 37, 38) |
| `Problem.cache_params` | method | `(self)` | 67 | `src/pinn_feynman/data/generation.py` (19) |
| `Problem.compute_options` | method | `(self)` | 70 | `src/pinn_feynman/data/generation.py` (24) |
| `Problem.data_problem` | method | `(self)` | 73 | `src/pinn_feynman/evaluation/predict.py` (27)<br>`src/pinn_feynman/physics/system.py` (156)<br>`src/pinn_feynman/training/setup.py` (77, 78, 80) |
| `Problem.columns` | method | `(self)` | 77 | `src/pinn_feynman/evaluation/evaluate.py` (96, 97)<br>`src/pinn_feynman/evaluation/predict.py` (70, 75)<br>`src/pinn_feynman/physics/bottom_up.py` (47, 48)<br>`src/pinn_feynman/providers/base.py` (28, 38, 39, 53, 64, 65, 66, 67, …)<br>`src/pinn_feynman/training/setup.py` (51, 52, 173, 174, 175)<br>`src/pinn_feynman/visualization/data_io.py` (85, 86)<br>`src/pinn_feynman/visualization/triangle.py` (102, 137, 147, 153, 166)<br>`src/pinn_feynman/visualization/tuning.py` (81, 91, 98)<br>`src/pinn_feynman/weighting/distance.py` (41, 58, 72, 75, 87, 90) |
| `Problem.build_provider` | method | `(self)` | 81 | `src/pinn_feynman/physics/system.py` (157)<br>`src/pinn_feynman/training/setup.py` (132) |
| `Problem.boni_results` | method | `(self)` | 85 | `src/pinn_feynman/evaluation/boni_reference.py` (60) |
| `available_problems` | function | `()` | 89 | `src/pinn_feynman/data/generation.py` (16, 33)<br>`src/pinn_feynman/evaluation/predict.py` (14, 24)<br>`src/pinn_feynman/physics/system.py` (20, 127, 141)<br>`src/pinn_feynman/problems/__init__.py` (7)<br>`src/pinn_feynman/problems/_registry.py` (98, 148)<br>`src/pinn_feynman/training/setup.py` (15, 77, 98) |
| `get_problem` | function | `(name)` | 96 | `src/pinn_feynman/evaluation/predict.py` (14, 25, 70)<br>`src/pinn_feynman/physics/system.py` (20, 141)<br>`src/pinn_feynman/problems/__init__.py` (7)<br>`src/pinn_feynman/problems/_registry.py` (128, 148, 155)<br>`src/pinn_feynman/training/setup.py` (15, 77, 100, 131, 173)<br>`src/pinn_feynman/weighting/distance.py` (38, 40) |
| `ProblemMapping` | class |  | 137 | `src/pinn_feynman/data/generation.py` (16, 19, 24)<br>`src/pinn_feynman/evaluation/boni_reference.py` (8, 60)<br>`src/pinn_feynman/physics/system.py` (20, 24)<br>`src/pinn_feynman/problems/__init__.py` (7) |

### `pinn_feynman.problems.box_massless`

[`src/pinn_feynman/problems/box_massless.py`](../src/pinn_feynman/problems/box_massless.py) — One-loop massless box (Boni sec. 6.2.1): one variable x = t/s, true values in closed form.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `PROBLEM` | constant |  | 5 | `src/pinn_feynman/problems/box_massless_bu.py` (5)<br>`src/pinn_feynman/problems/box_one_mass.py` (5)<br>`src/pinn_feynman/problems/box_one_mass_bu.py` (5)<br>`src/pinn_feynman/problems/nonplanar_three_point.py` (11)<br>`src/pinn_feynman/problems/nonplanar_three_point_bu.py` (7)<br>`src/pinn_feynman/problems/sunrise_equal_mass_positive.py` (8)<br>`src/pinn_feynman/problems/sunrise_equal_mass_positive_bu.py` (6) |

### `pinn_feynman.problems.box_massless_bu`

[`src/pinn_feynman/problems/box_massless_bu.py`](../src/pinn_feynman/problems/box_massless_bu.py) — Massless box, bottom-up: the network learns the box M3, the bubbles M1 and M2 come in closed form.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `PROBLEM` | constant |  | 5 | `src/pinn_feynman/problems/box_massless.py` (5)<br>`src/pinn_feynman/problems/box_one_mass.py` (5)<br>`src/pinn_feynman/problems/box_one_mass_bu.py` (5)<br>`src/pinn_feynman/problems/nonplanar_three_point.py` (11)<br>`src/pinn_feynman/problems/nonplanar_three_point_bu.py` (7)<br>`src/pinn_feynman/problems/sunrise_equal_mass_positive.py` (8)<br>`src/pinn_feynman/problems/sunrise_equal_mass_positive_bu.py` (6) |

### `pinn_feynman.problems.box_one_mass`

[`src/pinn_feynman/problems/box_one_mass.py`](../src/pinn_feynman/problems/box_one_mass.py) — One-loop one-mass box (Boni sec. 6.2.2): two variables (s1, s2) at s0 = 1, true values by series in Nielsen polylogs.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `PROBLEM` | constant |  | 5 | `src/pinn_feynman/problems/box_massless.py` (5)<br>`src/pinn_feynman/problems/box_massless_bu.py` (5)<br>`src/pinn_feynman/problems/box_one_mass_bu.py` (5)<br>`src/pinn_feynman/problems/nonplanar_three_point.py` (11)<br>`src/pinn_feynman/problems/nonplanar_three_point_bu.py` (7)<br>`src/pinn_feynman/problems/sunrise_equal_mass_positive.py` (8)<br>`src/pinn_feynman/problems/sunrise_equal_mass_positive_bu.py` (6) |

### `pinn_feynman.problems.box_one_mass_bu`

[`src/pinn_feynman/problems/box_one_mass_bu.py`](../src/pinn_feynman/problems/box_one_mass_bu.py) — One-mass box, bottom-up: the network learns the box M1, the three bubbles come in closed form.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `PROBLEM` | constant |  | 5 | `src/pinn_feynman/problems/box_massless.py` (5)<br>`src/pinn_feynman/problems/box_massless_bu.py` (5)<br>`src/pinn_feynman/problems/box_one_mass.py` (5)<br>`src/pinn_feynman/problems/nonplanar_three_point.py` (11)<br>`src/pinn_feynman/problems/nonplanar_three_point_bu.py` (7)<br>`src/pinn_feynman/problems/sunrise_equal_mass_positive.py` (8)<br>`src/pinn_feynman/problems/sunrise_equal_mass_positive_bu.py` (6) |

### `pinn_feynman.problems.nonplanar_three_point`

[`src/pinn_feynman/problems/nonplanar_three_point.py`](../src/pinn_feynman/problems/nonplanar_three_point.py) — Non-planar two-loop three-point function (Boni sec. 6.3.3), the two masters of the top sector (m10, m11), x = -s/m^2.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `PROBLEM` | constant |  | 11 | `src/pinn_feynman/problems/box_massless.py` (5)<br>`src/pinn_feynman/problems/box_massless_bu.py` (5)<br>`src/pinn_feynman/problems/box_one_mass.py` (5)<br>`src/pinn_feynman/problems/box_one_mass_bu.py` (5)<br>`src/pinn_feynman/problems/nonplanar_three_point_bu.py` (7)<br>`src/pinn_feynman/problems/sunrise_equal_mass_positive.py` (8)<br>`src/pinn_feynman/problems/sunrise_equal_mass_positive_bu.py` (6) |

### `pinn_feynman.problems.nonplanar_three_point_bu`

[`src/pinn_feynman/problems/nonplanar_three_point_bu.py`](../src/pinn_feynman/problems/nonplanar_three_point_bu.py) — Non-planar three-point, bottom-up name: the same problem as `nonplanar_three_point` (network on (m10, m11),

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `PROBLEM` | constant |  | 7 | `src/pinn_feynman/problems/box_massless.py` (5)<br>`src/pinn_feynman/problems/box_massless_bu.py` (5)<br>`src/pinn_feynman/problems/box_one_mass.py` (5)<br>`src/pinn_feynman/problems/box_one_mass_bu.py` (5)<br>`src/pinn_feynman/problems/nonplanar_three_point.py` (11)<br>`src/pinn_feynman/problems/sunrise_equal_mass_positive.py` (8)<br>`src/pinn_feynman/problems/sunrise_equal_mass_positive_bu.py` (6) |

### `pinn_feynman.problems.sunrise_equal_mass_positive`

[`src/pinn_feynman/problems/sunrise_equal_mass_positive.py`](../src/pinn_feynman/problems/sunrise_equal_mass_positive.py) — Equal-mass sunrise for s > 0, m^2 = 3: domains between the singular points 0, 3, 27 of (A.10), true values by

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `PROBLEM` | constant |  | 8 | `src/pinn_feynman/problems/box_massless.py` (5)<br>`src/pinn_feynman/problems/box_massless_bu.py` (5)<br>`src/pinn_feynman/problems/box_one_mass.py` (5)<br>`src/pinn_feynman/problems/box_one_mass_bu.py` (5)<br>`src/pinn_feynman/problems/nonplanar_three_point.py` (11)<br>`src/pinn_feynman/problems/nonplanar_three_point_bu.py` (7)<br>`src/pinn_feynman/problems/sunrise_equal_mass_positive_bu.py` (6) |

### `pinn_feynman.problems.sunrise_equal_mass_positive_bu`

[`src/pinn_feynman/problems/sunrise_equal_mass_positive_bu.py`](../src/pinn_feynman/problems/sunrise_equal_mass_positive_bu.py) — Equal-mass sunrise for s > 0, bottom-up: the network learns the coupled block (I_11100, I_21100), the

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `PROBLEM` | constant |  | 6 | `src/pinn_feynman/problems/box_massless.py` (5)<br>`src/pinn_feynman/problems/box_massless_bu.py` (5)<br>`src/pinn_feynman/problems/box_one_mass.py` (5)<br>`src/pinn_feynman/problems/box_one_mass_bu.py` (5)<br>`src/pinn_feynman/problems/nonplanar_three_point.py` (11)<br>`src/pinn_feynman/problems/nonplanar_three_point_bu.py` (7)<br>`src/pinn_feynman/problems/sunrise_equal_mass_positive.py` (8) |

### `pinn_feynman.progress`

[`src/pinn_feynman/progress.py`](../src/pinn_feynman/progress.py) — Console output of the commands: progress bars, notes, phase headers, tables.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `BAR_FORMAT` | constant |  | 23 | `src/pinn_feynman/progress.py` (39) |
| `enabled` | function | `()` | 26 | `src/pinn_feynman/progress.py` (39) |
| `bar` | function | `(total, desc, unit='it', initial=0, leave=True, disable=False)` | 31 | `src/pinn_feynman/data/generation.py` (57, 61, 66)<br>`src/pinn_feynman/training/trainer.py` (231, 234, 236) |
| `note` | function | `(text)` | 43 | `src/pinn_feynman/data/build.py` (57, 59, 62)<br>`src/pinn_feynman/devtools/codemap.py` (238, 239)<br>`src/pinn_feynman/progress.py` (50, 60)<br>`src/pinn_feynman/training/queue.py` (242, 256, 259, 304, 313, 315, 322)<br>`src/pinn_feynman/training/run.py` (165, 167, 175, 184, 186, 203)<br>`src/pinn_feynman/training/trainer.py` (238)<br>`src/pinn_feynman/tuning/publish.py` (77)<br>`src/pinn_feynman/tuning/tune.py` (116, 138, 142, 189, 193, 209, 214, 219, …)<br>`src/pinn_feynman/visualization/accuracy.py` (39, 40)<br>`src/pinn_feynman/visualization/singularities.py` (60, 66) |
| `header` | function | `(text)` | 49 | `src/pinn_feynman/data/build.py` (53)<br>`src/pinn_feynman/training/queue.py` (318, 328)<br>`src/pinn_feynman/training/run.py` (206)<br>`src/pinn_feynman/tuning/publish.py` (45, 52)<br>`src/pinn_feynman/tuning/tune.py` (247) |
| `table` | function | `(rows, indent='  ')` | 53 | `src/pinn_feynman/training/queue.py` (335)<br>`src/pinn_feynman/visualization/accuracy.py` (117, 118, 119, 123, 126, 127, 128, 129, …) |
| `duration` | function | `(seconds)` | 63 | `src/pinn_feynman/data/build.py` (62)<br>`src/pinn_feynman/training/queue.py` (304, 315, 328)<br>`src/pinn_feynman/training/run.py` (184) |
| `stream_command` | function | `(cmd, cwd=None, out=None)` | 74 | — |

### `pinn_feynman.providers`

[`src/pinn_feynman/providers/__init__.py`](../src/pinn_feynman/providers/__init__.py) — Providers of the known master integrals (sub-topologies) for the bottom-up formulation, see `base`.

### `pinn_feynman.providers.base`

[`src/pinn_feynman/providers/base.py`](../src/pinn_feynman/providers/base.py) — Providers of the known master integrals M_K (and of the forcing term) of the bottom-up formulation.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `ClosedFormProvider` | class |  | 22 | `src/pinn_feynman/providers/__init__.py` (3)<br>`src/pinn_feynman/providers/base.py` (44)<br>`src/pinn_feynman/providers/box_massless.py` (11, 28)<br>`src/pinn_feynman/providers/box_one_mass.py` (10, 30)<br>`src/pinn_feynman/providers/sunrise_equal_mass.py` (9, 23) |
| `ClosedFormProvider.values` | method | `(self, v)` | 36 | `src/pinn_feynman/data/cache.py` (39, 57, 70, 72, 73)<br>`src/pinn_feynman/data/check_interpolation.py` (23, 24)<br>`src/pinn_feynman/data/dataset.py` (61, 63, 77, 80, 91)<br>`src/pinn_feynman/data/generation.py` (64, 65)<br>`src/pinn_feynman/data/interpolation.py` (37, 38)<br>`src/pinn_feynman/evaluation/metrics.py` (38)<br>`src/pinn_feynman/evaluation/predict.py` (74, 75)<br>`src/pinn_feynman/physics/bottom_up.py` (22, 90, 106, 108, 111)<br>`src/pinn_feynman/providers/base.py` (81, 126)<br>`src/pinn_feynman/reference/box_massless.py` (93, 100, 103)<br>`src/pinn_feynman/reference/box_one_mass.py` (177, 181)<br>`src/pinn_feynman/training/recorder.py` (330, 331, 426, 427, 432, 434, 435, 442)<br>`src/pinn_feynman/tuning/objective.py` (28, 29)<br>`src/pinn_feynman/tuning/space.py` (21, 25, 27, 29, 32, 47, 59)<br>`src/pinn_feynman/visualization/accuracy.py` (14, 50, 51, 331, 332, 372, 374)<br>`src/pinn_feynman/visualization/components.py` (55, 81, 120, 157, 232, 288, 297, 299, …)<br>`src/pinn_feynman/visualization/data_io.py` (85)<br>`src/pinn_feynman/visualization/training.py` (116, 117, 118, 119, 157, 181, 214, 218, …)<br>`src/pinn_feynman/visualization/triangle.py` (37, 38, 39, 44, 138, 144, 145, 147, …)<br>`src/pinn_feynman/visualization/tuning.py` (44, 65, 66, 118, 119, 138, 139)<br>`src/pinn_feynman/weighting/distance.py` (117, 128) |
| `ConstantProvider` | class |  | 44 | `src/pinn_feynman/providers/__init__.py` (3) |
| `CompositeProvider` | class |  | 56 | `src/pinn_feynman/providers/__init__.py` (3) |
| `CompositeProvider.values` | method | `(self, v)` | 78 | `src/pinn_feynman/data/cache.py` (39, 57, 70, 72, 73)<br>`src/pinn_feynman/data/check_interpolation.py` (23, 24)<br>`src/pinn_feynman/data/dataset.py` (61, 63, 77, 80, 91)<br>`src/pinn_feynman/data/generation.py` (64, 65)<br>`src/pinn_feynman/data/interpolation.py` (37, 38)<br>`src/pinn_feynman/evaluation/metrics.py` (38)<br>`src/pinn_feynman/evaluation/predict.py` (74, 75)<br>`src/pinn_feynman/physics/bottom_up.py` (22, 90, 106, 108, 111)<br>`src/pinn_feynman/providers/base.py` (81, 126)<br>`src/pinn_feynman/reference/box_massless.py` (93, 100, 103)<br>`src/pinn_feynman/reference/box_one_mass.py` (177, 181)<br>`src/pinn_feynman/training/recorder.py` (330, 331, 426, 427, 432, 434, 435, 442)<br>`src/pinn_feynman/tuning/objective.py` (28, 29)<br>`src/pinn_feynman/tuning/space.py` (21, 25, 27, 29, 32, 47, 59)<br>`src/pinn_feynman/visualization/accuracy.py` (14, 50, 51, 331, 332, 372, 374)<br>`src/pinn_feynman/visualization/components.py` (55, 81, 120, 157, 232, 288, 297, 299, …)<br>`src/pinn_feynman/visualization/data_io.py` (85)<br>`src/pinn_feynman/visualization/training.py` (116, 117, 118, 119, 157, 181, 214, 218, …)<br>`src/pinn_feynman/visualization/triangle.py` (37, 38, 39, 44, 138, 144, 145, 147, …)<br>`src/pinn_feynman/visualization/tuning.py` (44, 65, 66, 118, 119, 138, 139)<br>`src/pinn_feynman/weighting/distance.py` (117, 128) |
| `ClosedFormForcing` | class |  | 89 | `src/pinn_feynman/providers/__init__.py` (3)<br>`src/pinn_feynman/providers/nonplanar_three_point.py` (9, 24) |
| `ClosedFormForcing.values` | method | `(self, v)` | 103 | `src/pinn_feynman/data/cache.py` (39, 57, 70, 72, 73)<br>`src/pinn_feynman/data/check_interpolation.py` (23, 24)<br>`src/pinn_feynman/data/dataset.py` (61, 63, 77, 80, 91)<br>`src/pinn_feynman/data/generation.py` (64, 65)<br>`src/pinn_feynman/data/interpolation.py` (37, 38)<br>`src/pinn_feynman/evaluation/metrics.py` (38)<br>`src/pinn_feynman/evaluation/predict.py` (74, 75)<br>`src/pinn_feynman/physics/bottom_up.py` (22, 90, 106, 108, 111)<br>`src/pinn_feynman/providers/base.py` (81, 126)<br>`src/pinn_feynman/reference/box_massless.py` (93, 100, 103)<br>`src/pinn_feynman/reference/box_one_mass.py` (177, 181)<br>`src/pinn_feynman/training/recorder.py` (330, 331, 426, 427, 432, 434, 435, 442)<br>`src/pinn_feynman/tuning/objective.py` (28, 29)<br>`src/pinn_feynman/tuning/space.py` (21, 25, 27, 29, 32, 47, 59)<br>`src/pinn_feynman/visualization/accuracy.py` (14, 50, 51, 331, 332, 372, 374)<br>`src/pinn_feynman/visualization/components.py` (55, 81, 120, 157, 232, 288, 297, 299, …)<br>`src/pinn_feynman/visualization/data_io.py` (85)<br>`src/pinn_feynman/visualization/training.py` (116, 117, 118, 119, 157, 181, 214, 218, …)<br>`src/pinn_feynman/visualization/triangle.py` (37, 38, 39, 44, 138, 144, 145, 147, …)<br>`src/pinn_feynman/visualization/tuning.py` (44, 65, 66, 118, 119, 138, 139)<br>`src/pinn_feynman/weighting/distance.py` (117, 128) |
| `ForcingFromMasters` | class |  | 111 | `src/pinn_feynman/providers/__init__.py` (3) |
| `ForcingFromMasters.values` | method | `(self, v)` | 125 | `src/pinn_feynman/data/cache.py` (39, 57, 70, 72, 73)<br>`src/pinn_feynman/data/check_interpolation.py` (23, 24)<br>`src/pinn_feynman/data/dataset.py` (61, 63, 77, 80, 91)<br>`src/pinn_feynman/data/generation.py` (64, 65)<br>`src/pinn_feynman/data/interpolation.py` (37, 38)<br>`src/pinn_feynman/evaluation/metrics.py` (38)<br>`src/pinn_feynman/evaluation/predict.py` (74, 75)<br>`src/pinn_feynman/physics/bottom_up.py` (22, 90, 106, 108, 111)<br>`src/pinn_feynman/providers/base.py` (81, 126)<br>`src/pinn_feynman/reference/box_massless.py` (93, 100, 103)<br>`src/pinn_feynman/reference/box_one_mass.py` (177, 181)<br>`src/pinn_feynman/training/recorder.py` (330, 331, 426, 427, 432, 434, 435, 442)<br>`src/pinn_feynman/tuning/objective.py` (28, 29)<br>`src/pinn_feynman/tuning/space.py` (21, 25, 27, 29, 32, 47, 59)<br>`src/pinn_feynman/visualization/accuracy.py` (14, 50, 51, 331, 332, 372, 374)<br>`src/pinn_feynman/visualization/components.py` (55, 81, 120, 157, 232, 288, 297, 299, …)<br>`src/pinn_feynman/visualization/data_io.py` (85)<br>`src/pinn_feynman/visualization/training.py` (116, 117, 118, 119, 157, 181, 214, 218, …)<br>`src/pinn_feynman/visualization/triangle.py` (37, 38, 39, 44, 138, 144, 145, 147, …)<br>`src/pinn_feynman/visualization/tuning.py` (44, 65, 66, 118, 119, 138, 139)<br>`src/pinn_feynman/weighting/distance.py` (117, 128) |

### `pinn_feynman.providers.box_massless`

[`src/pinn_feynman/providers/box_massless.py`](../src/pinn_feynman/providers/box_massless.py) — Known master integrals of the massless box: the two bubbles, in closed form (s = 1, x = t/s < 0).

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `ORDERS` | constant |  | 13 | `src/pinn_feynman/providers/box_massless.py` (20, 28)<br>`src/pinn_feynman/providers/box_one_mass.py` (12, 20, 30)<br>`src/pinn_feynman/providers/nonplanar_three_point.py` (11, 17, 18, 24)<br>`src/pinn_feynman/providers/sunrise_equal_mass.py` (11, 18, 23)<br>`src/pinn_feynman/reference/box_massless.py` (27, 33, 87, 96, 107)<br>`src/pinn_feynman/reference/box_one_mass.py` (47, 58, 162, 287, 297, 305, 307)<br>`src/pinn_feynman/reference/nonplanar_three_point.py` (29, 40, 235, 238, 239, 241, 242)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (34, 75, 90, 161, 164, 165, 167)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (30, 41, 99, 102, 103, 105) |
| `bubbles` | function | `()` | 26 | — |

### `pinn_feynman.providers.box_one_mass`

[`src/pinn_feynman/providers/box_one_mass.py`](../src/pinn_feynman/providers/box_one_mass.py) — Known master integrals of the one-mass box: the three bubbles, in closed form (s0 = 1, points (s1, s2)).

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `ORDERS` | constant |  | 12 | `src/pinn_feynman/providers/box_massless.py` (13, 20, 28)<br>`src/pinn_feynman/providers/box_one_mass.py` (20, 30)<br>`src/pinn_feynman/providers/nonplanar_three_point.py` (11, 17, 18, 24)<br>`src/pinn_feynman/providers/sunrise_equal_mass.py` (11, 18, 23)<br>`src/pinn_feynman/reference/box_massless.py` (27, 33, 87, 96, 107)<br>`src/pinn_feynman/reference/box_one_mass.py` (47, 58, 162, 287, 297, 305, 307)<br>`src/pinn_feynman/reference/nonplanar_three_point.py` (29, 40, 235, 238, 239, 241, 242)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (34, 75, 90, 161, 164, 165, 167)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (30, 41, 99, 102, 103, 105) |
| `bubbles` | function | `()` | 28 | — |

### `pinn_feynman.providers.nonplanar_three_point`

[`src/pinn_feynman/providers/nonplanar_three_point.py`](../src/pinn_feynman/providers/nonplanar_three_point.py) — Forcing term of the non-planar three-point masters (m10, m11): the subtopologies enter only through N = (0, N11).

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `ORDERS` | constant |  | 11 | `src/pinn_feynman/providers/box_massless.py` (13, 20, 28)<br>`src/pinn_feynman/providers/box_one_mass.py` (12, 20, 30)<br>`src/pinn_feynman/providers/nonplanar_three_point.py` (17, 18, 24)<br>`src/pinn_feynman/providers/sunrise_equal_mass.py` (11, 18, 23)<br>`src/pinn_feynman/reference/box_massless.py` (27, 33, 87, 96, 107)<br>`src/pinn_feynman/reference/box_one_mass.py` (47, 58, 162, 287, 297, 305, 307)<br>`src/pinn_feynman/reference/nonplanar_three_point.py` (29, 40, 235, 238, 239, 241, 242)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (34, 75, 90, 161, 164, 165, 167)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (30, 41, 99, 102, 103, 105) |
| `source` | function | `()` | 22 | `src/pinn_feynman/tuning/tune.py` (172) |

### `pinn_feynman.providers.sunrise_equal_mass`

[`src/pinn_feynman/providers/sunrise_equal_mass.py`](../src/pinn_feynman/providers/sunrise_equal_mass.py) — Known master integral of the equal-mass sunrise: the product of two tadpoles I_11000, constant in s.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `ORDERS` | constant |  | 11 | `src/pinn_feynman/providers/box_massless.py` (13, 20, 28)<br>`src/pinn_feynman/providers/box_one_mass.py` (12, 20, 30)<br>`src/pinn_feynman/providers/nonplanar_three_point.py` (11, 17, 18, 24)<br>`src/pinn_feynman/providers/sunrise_equal_mass.py` (18, 23)<br>`src/pinn_feynman/reference/box_massless.py` (27, 33, 87, 96, 107)<br>`src/pinn_feynman/reference/box_one_mass.py` (47, 58, 162, 287, 297, 305, 307)<br>`src/pinn_feynman/reference/nonplanar_three_point.py` (29, 40, 235, 238, 239, 241, 242)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (34, 75, 90, 161, 164, 165, 167)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (30, 41, 99, 102, 103, 105) |
| `tadpole` | function | `()` | 21 | `src/pinn_feynman/reference/sunrise_equal_mass.py` (117, 122)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (51, 56) |

### `pinn_feynman.reference`

[`src/pinn_feynman/reference/__init__.py`](../src/pinn_feynman/reference/__init__.py) — Exact reference values of master integrals (ground truth for training and testing).

### `pinn_feynman.reference.box_massless`

[`src/pinn_feynman/reference/box_massless.py`](../src/pinn_feynman/reference/box_massless.py) — Exact reference values of the one-loop massless box master integrals.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `ORDERS` | constant |  | 27 | `src/pinn_feynman/providers/box_massless.py` (13, 20, 28)<br>`src/pinn_feynman/providers/box_one_mass.py` (12, 20, 30)<br>`src/pinn_feynman/providers/nonplanar_three_point.py` (11, 17, 18, 24)<br>`src/pinn_feynman/providers/sunrise_equal_mass.py` (11, 18, 23)<br>`src/pinn_feynman/reference/box_massless.py` (33, 87, 96, 107)<br>`src/pinn_feynman/reference/box_one_mass.py` (47, 58, 162, 287, 297, 305, 307)<br>`src/pinn_feynman/reference/nonplanar_three_point.py` (29, 40, 235, 238, 239, 241, 242)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (34, 75, 90, 161, 164, 165, 167)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (30, 41, 99, 102, 103, 105) |
| `CAUCHY_POINTS` | constant |  | 29 | `src/pinn_feynman/reference/box_massless.py` (33, 84)<br>`src/pinn_feynman/reference/box_one_mass.py` (63, 176, 181) |
| `CAUCHY_RADIUS` | constant |  | 30 | `src/pinn_feynman/reference/box_massless.py` (33, 84)<br>`src/pinn_feynman/reference/box_one_mass.py` (64, 175) |
| `WORKING_DPS` | constant |  | 31 | `src/pinn_feynman/reference/box_massless.py` (34, 98)<br>`src/pinn_feynman/reference/box_one_mass.py` (65, 173) |
| `CACHE_PARAMS` | constant |  | 33 | `src/pinn_feynman/problems/_registry.py` (68)<br>`src/pinn_feynman/reference/box_one_mass.py` (58)<br>`src/pinn_feynman/reference/nonplanar_three_point.py` (40)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (75)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (41) |
| `COMPUTE_OPTIONS` | constant |  | 37 | `src/pinn_feynman/data/generation.py` (24, 48)<br>`src/pinn_feynman/problems/_registry.py` (71)<br>`src/pinn_feynman/reference/box_one_mass.py` (61)<br>`src/pinn_feynman/reference/nonplanar_three_point.py` (44)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (79)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (45) |
| `log_minus` | function | `(v)` | 40 | `src/pinn_feynman/reference/box_massless.py` (58, 65) |
| `r_gamma` | function | `(eps)` | 52 | `src/pinn_feynman/reference/box_massless.py` (64) |
| `bubble` | function | `(p2, eps)` | 56 | `src/pinn_feynman/providers/box_massless.py` (17, 19)<br>`src/pinn_feynman/providers/box_one_mass.py` (16, 19)<br>`src/pinn_feynman/reference/box_massless.py` (76, 77) |
| `box` | function | `(s, t, eps)` | 61 | `src/pinn_feynman/reference/box_massless.py` (78)<br>`src/pinn_feynman/reference/box_one_mass.py` (155) |
| `boni_basis` | function | `(x, eps, s=1)` | 69 | `src/pinn_feynman/reference/box_massless.py` (100)<br>`src/pinn_feynman/reference/box_one_mass.py` (177) |
| `cauchy_nodes` | function | `()` | 82 | `src/pinn_feynman/reference/box_massless.py` (99) |
| `cauchy_coefficients` | function | `(values, nodes, orders=ORDERS)` | 87 | `src/pinn_feynman/reference/box_massless.py` (103) |
| `laurent_coefficients` | function | `(x, orders=ORDERS)` | 96 | `src/pinn_feynman/reference/box_massless.py` (113) |
| `reference_values` | function | `(xs, orders=ORDERS)` | 107 | `src/pinn_feynman/problems/_registry.py` (65) |

### `pinn_feynman.reference.box_one_mass`

[`src/pinn_feynman/reference/box_one_mass.py`](../src/pinn_feynman/reference/box_one_mass.py) — Exact reference values of the one-loop one-mass box master integrals (physical region).

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `ORDERS` | constant |  | 47 | `src/pinn_feynman/providers/box_massless.py` (13, 20, 28)<br>`src/pinn_feynman/providers/box_one_mass.py` (12, 20, 30)<br>`src/pinn_feynman/providers/nonplanar_three_point.py` (11, 17, 18, 24)<br>`src/pinn_feynman/providers/sunrise_equal_mass.py` (11, 18, 23)<br>`src/pinn_feynman/reference/box_massless.py` (27, 33, 87, 96, 107)<br>`src/pinn_feynman/reference/box_one_mass.py` (58, 162, 287, 297, 305, 307)<br>`src/pinn_feynman/reference/nonplanar_three_point.py` (29, 40, 235, 238, 239, 241, 242)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (34, 75, 90, 161, 164, 165, 167)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (30, 41, 99, 102, 103, 105) |
| `N_MASTERS` | constant |  | 48 | `src/pinn_feynman/evaluation/predict.py` (26)<br>`src/pinn_feynman/reference/box_one_mass.py` (178, 179, 287, 307)<br>`src/pinn_feynman/reference/nonplanar_three_point.py` (30, 220, 241)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (35, 167)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (31, 105)<br>`src/pinn_feynman/systems/box_massless.py` (26, 51)<br>`src/pinn_feynman/systems/box_massless_canonical.py` (22)<br>`src/pinn_feynman/systems/box_one_mass.py` (32, 90)<br>`src/pinn_feynman/systems/nonplanar_three_point.py` (34)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (25, 65) |
| `METHOD` | constant |  | 50 | `src/pinn_feynman/reference/box_one_mass.py` (58) |
| `TS_STEP` | constant |  | 53 | `src/pinn_feynman/reference/box_one_mass.py` (58, 237) |
| `TS_TMAX` | constant |  | 54 | `src/pinn_feynman/reference/box_one_mass.py` (58, 237) |
| `BLOCK_SIZE` | constant |  | 56 | `src/pinn_feynman/reference/box_one_mass.py` (61, 308, 309) |
| `CACHE_PARAMS` | constant |  | 58 | `src/pinn_feynman/problems/_registry.py` (68)<br>`src/pinn_feynman/reference/box_massless.py` (33)<br>`src/pinn_feynman/reference/nonplanar_three_point.py` (40)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (75)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (41) |
| `COMPUTE_OPTIONS` | constant |  | 61 | `src/pinn_feynman/data/generation.py` (24, 48)<br>`src/pinn_feynman/problems/_registry.py` (71)<br>`src/pinn_feynman/reference/box_massless.py` (37)<br>`src/pinn_feynman/reference/nonplanar_three_point.py` (44)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (79)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (45) |
| `CAUCHY_POINTS` | constant |  | 63 | `src/pinn_feynman/reference/box_massless.py` (29, 33, 84)<br>`src/pinn_feynman/reference/box_one_mass.py` (176, 181) |
| `CAUCHY_RADIUS` | constant |  | 64 | `src/pinn_feynman/reference/box_massless.py` (30, 33, 84)<br>`src/pinn_feynman/reference/box_one_mass.py` (175) |
| `WORKING_DPS` | constant |  | 65 | `src/pinn_feynman/reference/box_massless.py` (31, 34, 98)<br>`src/pinn_feynman/reference/box_one_mass.py` (173) |
| `CONSTANTS_DPS` | constant |  | 67 | `src/pinn_feynman/reference/box_one_mass.py` (199) |
| `NIELSEN` | constant |  | 69 | `src/pinn_feynman/reference/box_one_mass.py` (239, 240, 252) |
| `box` | function | `(s0, s1, s2, eps)` | 129 | `src/pinn_feynman/reference/box_massless.py` (78)<br>`src/pinn_feynman/reference/box_one_mass.py` (155) |
| `boni_basis` | function | `(s1, s2, eps, s0=1)` | 144 | `src/pinn_feynman/reference/box_massless.py` (100)<br>`src/pinn_feynman/reference/box_one_mass.py` (177) |
| `laurent_coefficients` | function | `(point, orders=ORDERS, s0=1)` | 162 | `src/pinn_feynman/reference/box_massless.py` (113) |
| `nielsen_polylogs` | function | `(z, one_minus_z=None)` | 227 | `src/pinn_feynman/reference/box_one_mass.py` (251) |
| `hypergeometric_series` | function | `(z, one_minus_z=None)` | 246 | `src/pinn_feynman/reference/box_one_mass.py` (279) |
| `reference_values` | function | `(points, orders=ORDERS)` | 297 | `src/pinn_feynman/problems/_registry.py` (65) |

### `pinn_feynman.reference.nonplanar_three_point`

[`src/pinn_feynman/reference/nonplanar_three_point.py`](../src/pinn_feynman/reference/nonplanar_three_point.py) — Reference values of the non-planar three-point masters (m10, m11) (Boni sec. 6.3.3), x = -s/m^2, s -> s + i0.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `ORDERS` | constant |  | 29 | `src/pinn_feynman/providers/box_massless.py` (13, 20, 28)<br>`src/pinn_feynman/providers/box_one_mass.py` (12, 20, 30)<br>`src/pinn_feynman/providers/nonplanar_three_point.py` (11, 17, 18, 24)<br>`src/pinn_feynman/providers/sunrise_equal_mass.py` (11, 18, 23)<br>`src/pinn_feynman/reference/box_massless.py` (27, 33, 87, 96, 107)<br>`src/pinn_feynman/reference/box_one_mass.py` (47, 58, 162, 287, 297, 305, 307)<br>`src/pinn_feynman/reference/nonplanar_three_point.py` (40, 235, 238, 239, 241, 242)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (34, 75, 90, 161, 164, 165, 167)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (30, 41, 99, 102, 103, 105) |
| `N_MASTERS` | constant |  | 30 | `src/pinn_feynman/evaluation/predict.py` (26)<br>`src/pinn_feynman/reference/box_one_mass.py` (48, 178, 179, 287, 307)<br>`src/pinn_feynman/reference/nonplanar_three_point.py` (220, 241)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (35, 167)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (31, 105)<br>`src/pinn_feynman/systems/box_massless.py` (26, 51)<br>`src/pinn_feynman/systems/box_massless_canonical.py` (22)<br>`src/pinn_feynman/systems/box_one_mass.py` (32, 90)<br>`src/pinn_feynman/systems/nonplanar_three_point.py` (34)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (25, 65) |
| `LEADING_ORDER` | constant |  | 32 | `src/pinn_feynman/reference/nonplanar_three_point.py` (242) |
| `X_RANGE` | constant |  | 33 | `src/pinn_feynman/reference/nonplanar_three_point.py` (41, 173, 209) |
| `X_SMALLEST` | constant |  | 34 | `src/pinn_feynman/reference/nonplanar_three_point.py` (42, 169, 170, 210, 212) |
| `STARTS` | constant |  | 35 | `src/pinn_feynman/reference/nonplanar_three_point.py` (41, 180)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (37, 42, 62, 73) |
| `THRESHOLD` | constant |  | 36 | `src/pinn_feynman/reference/nonplanar_three_point.py` (89, 121, 145, 225, 226)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (38, 112) |
| `START_DIGITS` | constant |  | 38 | `src/pinn_feynman/reference/nonplanar_three_point.py` (41, 138) |
| `CACHE_PARAMS` | constant |  | 40 | `src/pinn_feynman/problems/_registry.py` (68)<br>`src/pinn_feynman/reference/box_massless.py` (33)<br>`src/pinn_feynman/reference/box_one_mass.py` (58)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (75)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (41) |
| `COMPUTE_OPTIONS` | constant |  | 44 | `src/pinn_feynman/data/generation.py` (24, 48)<br>`src/pinn_feynman/problems/_registry.py` (71)<br>`src/pinn_feynman/reference/box_massless.py` (37)<br>`src/pinn_feynman/reference/box_one_mass.py` (61)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (79)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (45) |
| `ZETA2` | constant |  | 46 | `src/pinn_feynman/reference/nonplanar_three_point.py` (99, 105) |
| `clausen2` | function | `(theta)` | 61 | `src/pinn_feynman/reference/nonplanar_three_point.py` (93) |
| `forcing` | function | `(x)` | 81 | `src/pinn_feynman/physics/bottom_up.py` (83)<br>`src/pinn_feynman/physics/coefficients.py` (20, 24, 30)<br>`src/pinn_feynman/physics/losses.py` (34, 35, 46)<br>`src/pinn_feynman/providers/nonplanar_three_point.py` (15, 18)<br>`src/pinn_feynman/reference/nonplanar_three_point.py` (163)<br>`src/pinn_feynman/training/recorder.py` (301)<br>`src/pinn_feynman/training/trainer.py` (137, 209, 222) |
| `variation_of_constants` | function | `(x, digits=START_DIGITS)` | 138 | `src/pinn_feynman/reference/nonplanar_three_point.py` (180) |
| `start_values` | function | `()` | 178 | `src/pinn_feynman/reference/nonplanar_three_point.py` (188)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (74) |
| `leading_values` | function | `(x)` | 217 | `src/pinn_feynman/reference/nonplanar_three_point.py` (242) |
| `reference_values` | function | `(x, orders=ORDERS)` | 235 | `src/pinn_feynman/problems/_registry.py` (65) |

### `pinn_feynman.reference.sunrise_equal_mass`

[`src/pinn_feynman/reference/sunrise_equal_mass.py`](../src/pinn_feynman/reference/sunrise_equal_mass.py) — Reference values of the equal-mass sunrise master integrals (m^2 = 3, s < 0), by integration of Boni's (A.10).

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `ORDERS` | constant |  | 34 | `src/pinn_feynman/providers/box_massless.py` (13, 20, 28)<br>`src/pinn_feynman/providers/box_one_mass.py` (12, 20, 30)<br>`src/pinn_feynman/providers/nonplanar_three_point.py` (11, 17, 18, 24)<br>`src/pinn_feynman/providers/sunrise_equal_mass.py` (11, 18, 23)<br>`src/pinn_feynman/reference/box_massless.py` (27, 33, 87, 96, 107)<br>`src/pinn_feynman/reference/box_one_mass.py` (47, 58, 162, 287, 297, 305, 307)<br>`src/pinn_feynman/reference/nonplanar_three_point.py` (29, 40, 235, 238, 239, 241, 242)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (75, 90, 161, 164, 165, 167)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (30, 41, 99, 102, 103, 105) |
| `N_MASTERS` | constant |  | 35 | `src/pinn_feynman/evaluation/predict.py` (26)<br>`src/pinn_feynman/reference/box_one_mass.py` (48, 178, 179, 287, 307)<br>`src/pinn_feynman/reference/nonplanar_three_point.py` (30, 220, 241)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (167)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (31, 105)<br>`src/pinn_feynman/systems/box_massless.py` (26, 51)<br>`src/pinn_feynman/systems/box_massless_canonical.py` (22)<br>`src/pinn_feynman/systems/box_one_mass.py` (32, 90)<br>`src/pinn_feynman/systems/nonplanar_three_point.py` (34)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (25, 65) |
| `MASS_SQUARED` | constant |  | 36 | `src/pinn_feynman/reference/sunrise_equal_mass.py` (89, 97, 99, 100)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (32)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (22) |
| `STARTING_VALUES` | constant |  | 38 | `src/pinn_feynman/reference/sunrise_equal_mass.py` (47, 52, 58, 62) |
| `INTEGRATED_MASTERS` | constant |  | 40 | `src/pinn_feynman/reference/sunrise_equal_mass.py` (61, 62, 64, 65) |
| `read_starts` | function | `(problem)` | 43 | `src/pinn_feynman/reference/sunrise_equal_mass.py` (70)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (37) |
| `S_RANGE` | constant |  | 72 | `src/pinn_feynman/reference/sunrise_equal_mass.py` (76, 138, 153) |
| `CACHE_PARAMS` | constant |  | 75 | `src/pinn_feynman/problems/_registry.py` (68)<br>`src/pinn_feynman/reference/box_massless.py` (33)<br>`src/pinn_feynman/reference/box_one_mass.py` (58)<br>`src/pinn_feynman/reference/nonplanar_three_point.py` (40)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (41) |
| `COMPUTE_OPTIONS` | constant |  | 79 | `src/pinn_feynman/data/generation.py` (24, 48)<br>`src/pinn_feynman/problems/_registry.py` (71)<br>`src/pinn_feynman/reference/box_massless.py` (37)<br>`src/pinn_feynman/reference/box_one_mass.py` (61)<br>`src/pinn_feynman/reference/nonplanar_three_point.py` (44)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (45) |
| `tadpole_series` | function | `()` | 83 | `src/pinn_feynman/providers/sunrise_equal_mass.py` (15, 17)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (117, 168)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (51, 106) |
| `pole_parts` | function | `(s)` | 94 | `src/pinn_feynman/reference/sunrise_equal_mass.py` (127)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (63) |
| `reference_values` | function | `(s, orders=ORDERS)` | 161 | `src/pinn_feynman/problems/_registry.py` (65) |

### `pinn_feynman.reference.sunrise_equal_mass_positive`

[`src/pinn_feynman/reference/sunrise_equal_mass_positive.py`](../src/pinn_feynman/reference/sunrise_equal_mass_positive.py) — Reference values of the equal-mass sunrise master integrals (m^2 = 3) for s > 0, between the singular points of

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `ORDERS` | constant |  | 30 | `src/pinn_feynman/providers/box_massless.py` (13, 20, 28)<br>`src/pinn_feynman/providers/box_one_mass.py` (12, 20, 30)<br>`src/pinn_feynman/providers/nonplanar_three_point.py` (11, 17, 18, 24)<br>`src/pinn_feynman/providers/sunrise_equal_mass.py` (11, 18, 23)<br>`src/pinn_feynman/reference/box_massless.py` (27, 33, 87, 96, 107)<br>`src/pinn_feynman/reference/box_one_mass.py` (47, 58, 162, 287, 297, 305, 307)<br>`src/pinn_feynman/reference/nonplanar_three_point.py` (29, 40, 235, 238, 239, 241, 242)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (34, 75, 90, 161, 164, 165, 167)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (41, 99, 102, 103, 105) |
| `N_MASTERS` | constant |  | 31 | `src/pinn_feynman/evaluation/predict.py` (26)<br>`src/pinn_feynman/reference/box_one_mass.py` (48, 178, 179, 287, 307)<br>`src/pinn_feynman/reference/nonplanar_three_point.py` (30, 220, 241)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (35, 167)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (105)<br>`src/pinn_feynman/systems/box_massless.py` (26, 51)<br>`src/pinn_feynman/systems/box_massless_canonical.py` (22)<br>`src/pinn_feynman/systems/box_one_mass.py` (32, 90)<br>`src/pinn_feynman/systems/nonplanar_three_point.py` (34)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (25, 65) |
| `MASS_SQUARED` | constant |  | 32 | `src/pinn_feynman/reference/sunrise_equal_mass.py` (36, 89, 97, 99, 100)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (22) |
| `S_RANGES` | constant |  | 34 | `src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (43, 72, 92, 95) |
| `STARTS` | constant |  | 37 | `src/pinn_feynman/reference/nonplanar_three_point.py` (35, 41, 180)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (42, 62, 73) |
| `THRESHOLD` | constant |  | 38 | `src/pinn_feynman/reference/nonplanar_three_point.py` (36, 89, 121, 145, 225, 226)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (112) |
| `CACHE_PARAMS` | constant |  | 41 | `src/pinn_feynman/problems/_registry.py` (68)<br>`src/pinn_feynman/reference/box_massless.py` (33)<br>`src/pinn_feynman/reference/box_one_mass.py` (58)<br>`src/pinn_feynman/reference/nonplanar_three_point.py` (40)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (75) |
| `COMPUTE_OPTIONS` | constant |  | 45 | `src/pinn_feynman/data/generation.py` (24, 48)<br>`src/pinn_feynman/problems/_registry.py` (71)<br>`src/pinn_feynman/reference/box_massless.py` (37)<br>`src/pinn_feynman/reference/box_one_mass.py` (61)<br>`src/pinn_feynman/reference/nonplanar_three_point.py` (44)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (79) |
| `start_values` | function | `(index)` | 60 | `src/pinn_feynman/reference/nonplanar_three_point.py` (188)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (74) |
| `reference_values` | function | `(s, orders=ORDERS)` | 99 | `src/pinn_feynman/problems/_registry.py` (65) |

### `pinn_feynman.runtime`

[`src/pinn_feynman/runtime.py`](../src/pinn_feynman/runtime.py) — Where data and runs are stored.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `STORAGE_ENV` | constant |  | 17 | `src/pinn_feynman/runtime.py` (45) |
| `DEFAULT_STORAGE` | constant |  | 19 | `src/pinn_feynman/runtime.py` (51) |
| `_LOOKUP` | constant |  | 22 | `src/pinn_feynman/runtime.py` (27, 31) |
| `repository_root` | function | `()` | 34 | `src/pinn_feynman/paths.py` (11, 19)<br>`src/pinn_feynman/runtime.py` (51, 67) |
| `StorageNotFound` | class |  | 39 | — |
| `storage_root` | function | `()` | 43 | `src/pinn_feynman/paths.py` (11, 25, 30)<br>`src/pinn_feynman/runtime.py` (68) |
| `summary` | function | `()` | 54 | `src/pinn_feynman/evaluation/evaluate.py` (37)<br>`src/pinn_feynman/evaluation/predict.py` (51, 95, 96)<br>`src/pinn_feynman/training/run.py` (105, 110, 111, 177, 178, 179, 184, 185, …)<br>`src/pinn_feynman/tuning/publish.py` (42, 44, 46, 47, 48, 70, 71)<br>`src/pinn_feynman/tuning/tune.py` (74, 84, 86, 91, 137, 138, 139, 140, …) |

### `pinn_feynman.systems`

[`src/pinn_feynman/systems/__init__.py`](../src/pinn_feynman/systems/__init__.py) — Differential-equation systems of master integrals (connection matrices).

### `pinn_feynman.systems.analysis`

[`src/pinn_feynman/systems/analysis.py`](../src/pinn_feynman/systems/analysis.py) — Local analysis of a system dM/dx = A(x, eps) M at its singular points (Frobenius theory).

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `SYSTEM_MODULES` | constant |  | 60 | `src/pinn_feynman/systems/analysis.py` (141, 163)<br>`src/pinn_feynman/visualization/singularities.py` (31, 35) |
| `INFINITY` | constant |  | 61 | `src/pinn_feynman/systems/analysis.py` (81, 143) |
| `PointAnalysis` | class |  | 65 | `src/pinn_feynman/systems/analysis.py` (135) |
| `PointAnalysis.fuchsian` | method | `(self)` | 74 | `src/pinn_feynman/systems/analysis.py` (151) |
| `local_matrix` | function | `(a, x, point)` | 78 | `src/pinn_feynman/systems/analysis.py` (131) |
| `pole_order` | function | `(a, t)` | 86 | `src/pinn_feynman/systems/analysis.py` (67, 75, 132, 152, 156) |
| `residue` | function | `(a, t)` | 99 | `src/pinn_feynman/systems/analysis.py` (68, 133, 156) |
| `resonances` | function | `(exponents)` | 104 | `src/pinn_feynman/systems/analysis.py` (70, 136, 155) |
| `jordan_blocks` | function | `(m)` | 116 | `src/pinn_feynman/systems/analysis.py` (71, 136, 157) |
| `analyse_point` | function | `(a, x, point)` | 130 | `src/pinn_feynman/systems/analysis.py` (144) |
| `analyse` | function | `(name, include_infinity=True)` | 139 | `src/pinn_feynman/systems/analysis.py` (165) |
| `report` | function | `(analyses)` | 147 | `src/pinn_feynman/evaluation/evaluate.py` (33, 40, 152, 153, 154, 157)<br>`src/pinn_feynman/systems/analysis.py` (165)<br>`src/pinn_feynman/tuning/objective.py` (17) |
| `main` | function | `(argv=None)` | 161 | `src/pinn_feynman/data/build.py` (76)<br>`src/pinn_feynman/data/check_interpolation.py` (46)<br>`src/pinn_feynman/devtools/benchmark.py` (68)<br>`src/pinn_feynman/devtools/codemap.py` (318)<br>`src/pinn_feynman/devtools/list_configs.py` (97)<br>`src/pinn_feynman/evaluation/evaluate.py` (167)<br>`src/pinn_feynman/systems/analysis.py` (171)<br>`src/pinn_feynman/training/queue.py` (379)<br>`src/pinn_feynman/training/run.py` (212)<br>`src/pinn_feynman/tuning/tune.py` (253) |

### `pinn_feynman.systems.blocks`

[`src/pinn_feynman/systems/blocks.py`](../src/pinn_feynman/systems/blocks.py) — Blocks of a system dM/dz_k = A_k M: dependency graph, bottom-up order, split into unknowns U and knowns K.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `EPS` | constant |  | 22 | `src/pinn_feynman/systems/blocks.py` (25)<br>`src/pinn_feynman/systems/box_massless.py` (18, 36, 49, 50, 51)<br>`src/pinn_feynman/systems/box_massless_canonical.py` (19, 31, 37, 38)<br>`src/pinn_feynman/systems/box_one_mass.py` (26, 49, 88, 89, 90)<br>`src/pinn_feynman/systems/nonplanar_three_point.py` (26, 66)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (17, 42, 63, 64) |
| `epsilon_orders` | function | `(matrix, eps=EPS)` | 25 | — |
| `is_zero` | function | `(entry, n_checks=3, seed=0)` | 47 | `src/pinn_feynman/systems/blocks.py` (118, 235) |
| `Partition` | class |  | 85 | `src/pinn_feynman/systems/blocks.py` (194, 210) |
| `Partition.top_blocks` | method | `(self)` | 93 | — |
| `Partition.closure_below` | method | `(self, indices)` | 98 | — |
| `partition` | function | `(orders_by_variable)` | 167 | `src/pinn_feynman/problems/_registry.py` (20)<br>`src/pinn_feynman/systems/blocks.py` (210, 213, 216, 217, 241) |
| `Split` | class |  | 203 | `src/pinn_feynman/systems/blocks.py` (241) |
| `Split.report` | method | `(self)` | 212 | `src/pinn_feynman/evaluation/evaluate.py` (33, 40, 152, 153, 154, 157)<br>`src/pinn_feynman/systems/analysis.py` (165)<br>`src/pinn_feynman/tuning/objective.py` (17) |
| `split` | function | `(orders_by_variable, unknowns)` | 221 | `src/pinn_feynman/collocation/r3.py` (79)<br>`src/pinn_feynman/collocation/sampler.py` (86, 145)<br>`src/pinn_feynman/devtools/codemap.py` (107, 121, 170, 238)<br>`src/pinn_feynman/evaluation/predict.py` (107)<br>`src/pinn_feynman/physics/system.py` (147, 153)<br>`src/pinn_feynman/training/environment.py` (20, 29)<br>`src/pinn_feynman/training/queue.py` (160, 167, 172, 188)<br>`src/pinn_feynman/training/trainer.py` (143)<br>`src/pinn_feynman/tuning/space.py` (37)<br>`src/pinn_feynman/visualization/data_io.py` (50) |

### `pinn_feynman.systems.box_massless`

[`src/pinn_feynman/systems/box_massless.py`](../src/pinn_feynman/systems/box_massless.py) — One-loop massless box: differential-equation system in Boni's basis.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `C_X_BONI` | constant |  | 21 | `src/pinn_feynman/systems/box_massless.py` (64, 69) |
| `BONI_DOMAIN` | constant |  | 23 | — |
| `SINGULAR_POINTS` | constant |  | 25 | `src/pinn_feynman/systems/analysis.py` (143)<br>`src/pinn_feynman/systems/box_massless_canonical.py` (21)<br>`src/pinn_feynman/systems/nonplanar_three_point.py` (29)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (24) |
| `N_MASTERS` | constant |  | 26 | `src/pinn_feynman/evaluation/predict.py` (26)<br>`src/pinn_feynman/reference/box_one_mass.py` (48, 178, 179, 287, 307)<br>`src/pinn_feynman/reference/nonplanar_three_point.py` (30, 220, 241)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (35, 167)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (31, 105)<br>`src/pinn_feynman/systems/box_massless.py` (51)<br>`src/pinn_feynman/systems/box_massless_canonical.py` (22)<br>`src/pinn_feynman/systems/box_one_mass.py` (32, 90)<br>`src/pinn_feynman/systems/nonplanar_three_point.py` (34)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (25, 65) |
| `VARIABLES` | constant |  | 28 | `src/pinn_feynman/physics/system.py` (69, 132, 151)<br>`src/pinn_feynman/systems/box_massless.py` (46, 47)<br>`src/pinn_feynman/systems/box_one_mass.py` (30, 98)<br>`src/pinn_feynman/systems/nonplanar_three_point.py` (36, 71, 72)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (28, 60, 61)<br>`src/pinn_feynman/visualization/singularities.py` (55) |
| `TITLE` | constant |  | 30 | `src/pinn_feynman/evaluation/evaluate.py` (53, 104, 117)<br>`src/pinn_feynman/systems/box_one_mass.py` (35)<br>`src/pinn_feynman/systems/nonplanar_three_point.py` (44)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (30) |
| `VARIABLE_LABELS` | constant |  | 31 | `src/pinn_feynman/systems/box_one_mass.py` (36)<br>`src/pinn_feynman/systems/nonplanar_three_point.py` (45)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (31) |
| `connection_matrix` | function | `()` | 34 | `src/pinn_feynman/systems/analysis.py` (142)<br>`src/pinn_feynman/systems/box_massless.py` (48)<br>`src/pinn_feynman/systems/box_one_mass.py` (85)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (62) |
| `connection_matrix_orders` | function | `(variable='x')` | 44 | `src/pinn_feynman/physics/system.py` (70, 153)<br>`src/pinn_feynman/systems/box_massless.py` (58)<br>`src/pinn_feynman/systems/box_one_mass.py` (99)<br>`src/pinn_feynman/systems/nonplanar_three_point.py` (78)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (72)<br>`src/pinn_feynman/weighting/distance.py` (57) |
| `numeric_orders` | function | `(modules='numpy')` | 56 | `src/pinn_feynman/reference/nonplanar_three_point.py` (159)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (110) |
| `to_boni_x` | function | `(x)` | 62 | — |
| `from_boni_x` | function | `(x_boni)` | 67 | — |

### `pinn_feynman.systems.box_massless_canonical`

[`src/pinn_feynman/systems/box_massless_canonical.py`](../src/pinn_feynman/systems/box_massless_canonical.py) — One-loop massless box in a canonical (epsilon-factorised, dlog) basis.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `SINGULAR_POINTS` | constant |  | 21 | `src/pinn_feynman/systems/analysis.py` (143)<br>`src/pinn_feynman/systems/box_massless.py` (25)<br>`src/pinn_feynman/systems/nonplanar_three_point.py` (29)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (24) |
| `N_MASTERS` | constant |  | 22 | `src/pinn_feynman/evaluation/predict.py` (26)<br>`src/pinn_feynman/reference/box_one_mass.py` (48, 178, 179, 287, 307)<br>`src/pinn_feynman/reference/nonplanar_three_point.py` (30, 220, 241)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (35, 167)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (31, 105)<br>`src/pinn_feynman/systems/box_massless.py` (26, 51)<br>`src/pinn_feynman/systems/box_one_mass.py` (32, 90)<br>`src/pinn_feynman/systems/nonplanar_three_point.py` (34)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (25, 65) |
| `S` | constant |  | 23 | `src/pinn_feynman/systems/box_massless_canonical.py` (36, 38)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (18, 20, 42, 72) |
| `A_RES` | constant |  | 25 | `src/pinn_feynman/systems/box_massless_canonical.py` (31) |
| `B_RES` | constant |  | 26 | `src/pinn_feynman/systems/box_massless_canonical.py` (31) |
| `connection_matrix` | function | `()` | 29 | `src/pinn_feynman/systems/analysis.py` (142)<br>`src/pinn_feynman/systems/box_massless.py` (48)<br>`src/pinn_feynman/systems/box_one_mass.py` (85)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (62) |
| `gauge_from_boni` | function | `()` | 34 | — |

### `pinn_feynman.systems.box_one_mass`

[`src/pinn_feynman/systems/box_one_mass.py`](../src/pinn_feynman/systems/box_one_mass.py) — One-loop one-mass box: differential-equation system in Boni's basis, physical region.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `INVARIANTS` | constant |  | 28 | `src/pinn_feynman/systems/box_one_mass.py` (77) |
| `VARIABLES` | constant |  | 30 | `src/pinn_feynman/physics/system.py` (69, 132, 151)<br>`src/pinn_feynman/systems/box_massless.py` (28, 46, 47)<br>`src/pinn_feynman/systems/box_one_mass.py` (98)<br>`src/pinn_feynman/systems/nonplanar_three_point.py` (36, 71, 72)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (28, 60, 61)<br>`src/pinn_feynman/visualization/singularities.py` (55) |
| `S0_VALUE` | constant |  | 31 | `src/pinn_feynman/systems/box_one_mass.py` (80) |
| `N_MASTERS` | constant |  | 32 | `src/pinn_feynman/evaluation/predict.py` (26)<br>`src/pinn_feynman/reference/box_one_mass.py` (48, 178, 179, 287, 307)<br>`src/pinn_feynman/reference/nonplanar_three_point.py` (30, 220, 241)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (35, 167)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (31, 105)<br>`src/pinn_feynman/systems/box_massless.py` (26, 51)<br>`src/pinn_feynman/systems/box_massless_canonical.py` (22)<br>`src/pinn_feynman/systems/box_one_mass.py` (90)<br>`src/pinn_feynman/systems/nonplanar_three_point.py` (34)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (25, 65) |
| `MASTER_NAMES` | constant |  | 33 | `src/pinn_feynman/systems/nonplanar_three_point.py` (35)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (26) |
| `TITLE` | constant |  | 35 | `src/pinn_feynman/evaluation/evaluate.py` (53, 104, 117)<br>`src/pinn_feynman/systems/box_massless.py` (30)<br>`src/pinn_feynman/systems/nonplanar_three_point.py` (44)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (30) |
| `VARIABLE_LABELS` | constant |  | 36 | `src/pinn_feynman/systems/box_massless.py` (31)<br>`src/pinn_feynman/systems/nonplanar_three_point.py` (45)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (31) |
| `EDGE_LABELS` | constant |  | 38 | — |
| `DEFAULT_MARGIN` | constant |  | 40 | `src/pinn_feynman/systems/box_one_mass.py` (132) |
| `MARGIN_TOLERANCE` | constant |  | 42 | `src/pinn_feynman/systems/box_one_mass.py` (129) |
| `DENOMINATOR_FACTORS` | constant |  | 44 | `src/pinn_feynman/visualization/singularities.py` (84) |
| `connection_matrix` | function | `(variable)` | 47 | `src/pinn_feynman/systems/analysis.py` (142)<br>`src/pinn_feynman/systems/box_massless.py` (48)<br>`src/pinn_feynman/systems/box_one_mass.py` (85)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (62) |
| `connection_matrix_orders` | function | `(variable, s0=S0_VALUE)` | 80 | `src/pinn_feynman/physics/system.py` (70, 153)<br>`src/pinn_feynman/systems/box_massless.py` (58)<br>`src/pinn_feynman/systems/box_one_mass.py` (99)<br>`src/pinn_feynman/systems/nonplanar_three_point.py` (78)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (72)<br>`src/pinn_feynman/weighting/distance.py` (57) |
| `numeric_orders` | function | `(modules='numpy')` | 95 | `src/pinn_feynman/reference/nonplanar_three_point.py` (159)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (110) |
| `edge_distance` | function | `(points)` | 105 | `src/pinn_feynman/systems/box_one_mass.py` (128) |
| `edge_distances` | function | `(points)` | 113 | `src/pinn_feynman/evaluation/evaluate.py` (120) |
| `in_domain` | function | `(points, margin=0.0)` | 121 | `src/pinn_feynman/data/domains.py` (171) |
| `vertices` | function | `(margin=DEFAULT_MARGIN)` | 132 | `src/pinn_feynman/data/domains.py` (117, 121, 130, 146, 161, 166)<br>`src/pinn_feynman/evaluation/evaluate.py` (114) |

### `pinn_feynman.systems.nonplanar_three_point`

[`src/pinn_feynman/systems/nonplanar_three_point.py`](../src/pinn_feynman/systems/nonplanar_three_point.py) — Non-planar two-loop three-point function with a massive loop (Boni sec. 6.3.3): the two masters of the top sector.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `EPS` | constant |  | 26 | `src/pinn_feynman/systems/blocks.py` (22, 25)<br>`src/pinn_feynman/systems/box_massless.py` (18, 36, 49, 50, 51)<br>`src/pinn_feynman/systems/box_massless_canonical.py` (19, 31, 37, 38)<br>`src/pinn_feynman/systems/box_one_mass.py` (26, 49, 88, 89, 90)<br>`src/pinn_feynman/systems/nonplanar_three_point.py` (66)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (17, 42, 63, 64) |
| `X` | constant |  | 27 | `src/pinn_feynman/systems/analysis.py` (144)<br>`src/pinn_feynman/systems/box_massless.py` (18, 36, 59)<br>`src/pinn_feynman/systems/box_massless_canonical.py` (19, 31, 36)<br>`src/pinn_feynman/systems/nonplanar_three_point.py` (55, 78)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (20) |
| `SINGULAR_POINTS` | constant |  | 29 | `src/pinn_feynman/systems/analysis.py` (143)<br>`src/pinn_feynman/systems/box_massless.py` (25)<br>`src/pinn_feynman/systems/box_massless_canonical.py` (21)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (24) |
| `FORCING_BRANCH_POINTS` | constant |  | 33 | — |
| `N_MASTERS` | constant |  | 34 | `src/pinn_feynman/evaluation/predict.py` (26)<br>`src/pinn_feynman/reference/box_one_mass.py` (48, 178, 179, 287, 307)<br>`src/pinn_feynman/reference/nonplanar_three_point.py` (30, 220, 241)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (35, 167)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (31, 105)<br>`src/pinn_feynman/systems/box_massless.py` (26, 51)<br>`src/pinn_feynman/systems/box_massless_canonical.py` (22)<br>`src/pinn_feynman/systems/box_one_mass.py` (32, 90)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (25, 65) |
| `MASTER_NAMES` | constant |  | 35 | `src/pinn_feynman/systems/box_one_mass.py` (33)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (26) |
| `VARIABLES` | constant |  | 36 | `src/pinn_feynman/physics/system.py` (69, 132, 151)<br>`src/pinn_feynman/systems/box_massless.py` (28, 46, 47)<br>`src/pinn_feynman/systems/box_one_mass.py` (30, 98)<br>`src/pinn_feynman/systems/nonplanar_three_point.py` (71, 72)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (28, 60, 61)<br>`src/pinn_feynman/visualization/singularities.py` (55) |
| `DEFAULT_DOMAIN` | constant |  | 38 | — |
| `SHOWN_ORDERS` | constant |  | 40 | — |
| `NONZERO_ORDERS` | constant |  | 43 | — |
| `TITLE` | constant |  | 44 | `src/pinn_feynman/evaluation/evaluate.py` (53, 104, 117)<br>`src/pinn_feynman/systems/box_massless.py` (30)<br>`src/pinn_feynman/systems/box_one_mass.py` (35)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (30) |
| `VARIABLE_LABELS` | constant |  | 45 | `src/pinn_feynman/systems/box_massless.py` (31)<br>`src/pinn_feynman/systems/box_one_mass.py` (36)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (31) |
| `FIGURE_LABELS` | constant |  | 46 | `src/pinn_feynman/systems/sunrise_equal_mass.py` (33) |
| `homogeneous_matrices` | function | `()` | 53 | `src/pinn_feynman/systems/nonplanar_three_point.py` (65, 73) |
| `connection_matrix` | function | `()` | 63 | `src/pinn_feynman/systems/analysis.py` (142)<br>`src/pinn_feynman/systems/box_massless.py` (48)<br>`src/pinn_feynman/systems/box_one_mass.py` (85)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (62) |
| `connection_matrix_orders` | function | `(variable='x')` | 69 | `src/pinn_feynman/physics/system.py` (70, 153)<br>`src/pinn_feynman/systems/box_massless.py` (58)<br>`src/pinn_feynman/systems/box_one_mass.py` (99)<br>`src/pinn_feynman/systems/nonplanar_three_point.py` (78)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (72)<br>`src/pinn_feynman/weighting/distance.py` (57) |
| `numeric_orders` | function | `(modules='numpy')` | 76 | `src/pinn_feynman/reference/nonplanar_three_point.py` (159)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (110) |

### `pinn_feynman.systems.singular_kinds`

[`src/pinn_feynman/systems/singular_kinds.py`](../src/pinn_feynman/systems/singular_kinds.py) — Kind of every singular point (one variable) and singular line (two variables) of the systems: one place for it.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `KINDS` | constant |  | 20 | `src/pinn_feynman/systems/singular_kinds.py` (52)<br>`src/pinn_feynman/visualization/singularities.py` (27, 54) |
| `LINE_KINDS` | constant |  | 43 | `src/pinn_feynman/systems/singular_kinds.py` (60)<br>`src/pinn_feynman/visualization/singularities.py` (27, 80) |
| `point_kind` | function | `(problem, point)` | 49 | — |
| `line_kind` | function | `(problem, factor)` | 57 | — |

### `pinn_feynman.systems.sunrise_equal_mass`

[`src/pinn_feynman/systems/sunrise_equal_mass.py`](../src/pinn_feynman/systems/sunrise_equal_mass.py) — Two-loop equal-mass sunrise: differential-equation system in Boni's basis, m^2 = 3.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `EPS` | constant |  | 17 | `src/pinn_feynman/systems/blocks.py` (22, 25)<br>`src/pinn_feynman/systems/box_massless.py` (18, 36, 49, 50, 51)<br>`src/pinn_feynman/systems/box_massless_canonical.py` (19, 31, 37, 38)<br>`src/pinn_feynman/systems/box_one_mass.py` (26, 49, 88, 89, 90)<br>`src/pinn_feynman/systems/nonplanar_three_point.py` (26, 66)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (42, 63, 64) |
| `S` | constant |  | 18 | `src/pinn_feynman/systems/box_massless_canonical.py` (23, 36, 38)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (20, 42, 72) |
| `X` | constant |  | 20 | `src/pinn_feynman/systems/analysis.py` (144)<br>`src/pinn_feynman/systems/box_massless.py` (18, 36, 59)<br>`src/pinn_feynman/systems/box_massless_canonical.py` (19, 31, 36)<br>`src/pinn_feynman/systems/nonplanar_three_point.py` (27, 55, 78) |
| `MASS_SQUARED` | constant |  | 22 | `src/pinn_feynman/reference/sunrise_equal_mass.py` (36, 89, 97, 99, 100)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (32) |
| `SINGULAR_POINTS` | constant |  | 24 | `src/pinn_feynman/systems/analysis.py` (143)<br>`src/pinn_feynman/systems/box_massless.py` (25)<br>`src/pinn_feynman/systems/box_massless_canonical.py` (21)<br>`src/pinn_feynman/systems/nonplanar_three_point.py` (29) |
| `N_MASTERS` | constant |  | 25 | `src/pinn_feynman/evaluation/predict.py` (26)<br>`src/pinn_feynman/reference/box_one_mass.py` (48, 178, 179, 287, 307)<br>`src/pinn_feynman/reference/nonplanar_three_point.py` (30, 220, 241)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (35, 167)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (31, 105)<br>`src/pinn_feynman/systems/box_massless.py` (26, 51)<br>`src/pinn_feynman/systems/box_massless_canonical.py` (22)<br>`src/pinn_feynman/systems/box_one_mass.py` (32, 90)<br>`src/pinn_feynman/systems/nonplanar_three_point.py` (34)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (65) |
| `MASTER_NAMES` | constant |  | 26 | `src/pinn_feynman/systems/box_one_mass.py` (33)<br>`src/pinn_feynman/systems/nonplanar_three_point.py` (35) |
| `VARIABLES` | constant |  | 28 | `src/pinn_feynman/physics/system.py` (69, 132, 151)<br>`src/pinn_feynman/systems/box_massless.py` (28, 46, 47)<br>`src/pinn_feynman/systems/box_one_mass.py` (30, 98)<br>`src/pinn_feynman/systems/nonplanar_three_point.py` (36, 71, 72)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (60, 61)<br>`src/pinn_feynman/visualization/singularities.py` (55) |
| `TITLE` | constant |  | 30 | `src/pinn_feynman/evaluation/evaluate.py` (53, 104, 117)<br>`src/pinn_feynman/systems/box_massless.py` (30)<br>`src/pinn_feynman/systems/box_one_mass.py` (35)<br>`src/pinn_feynman/systems/nonplanar_three_point.py` (44) |
| `VARIABLE_LABELS` | constant |  | 31 | `src/pinn_feynman/systems/box_massless.py` (31)<br>`src/pinn_feynman/systems/box_one_mass.py` (36)<br>`src/pinn_feynman/systems/nonplanar_three_point.py` (45) |
| `FIGURE_LABELS` | constant |  | 33 | `src/pinn_feynman/systems/nonplanar_three_point.py` (46) |
| `connection_matrix` | function | `()` | 40 | `src/pinn_feynman/systems/analysis.py` (142)<br>`src/pinn_feynman/systems/box_massless.py` (48)<br>`src/pinn_feynman/systems/box_one_mass.py` (85)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (62) |
| `connection_matrix_orders` | function | `(variable='s')` | 55 | `src/pinn_feynman/physics/system.py` (70, 153)<br>`src/pinn_feynman/systems/box_massless.py` (58)<br>`src/pinn_feynman/systems/box_one_mass.py` (99)<br>`src/pinn_feynman/systems/nonplanar_three_point.py` (78)<br>`src/pinn_feynman/systems/sunrise_equal_mass.py` (72)<br>`src/pinn_feynman/weighting/distance.py` (57) |
| `numeric_orders` | function | `(modules='numpy')` | 70 | `src/pinn_feynman/reference/nonplanar_three_point.py` (159)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (110) |

### `pinn_feynman.training`

[`src/pinn_feynman/training/__init__.py`](../src/pinn_feynman/training/__init__.py) — Training loop, optimizer/scheduler, experiment assembly and entry point.

### `pinn_feynman.training.environment`

[`src/pinn_feynman/training/environment.py`](../src/pinn_feynman/training/environment.py) — Where a run was made: library versions, CPU model, threads.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `cpu_model` | function | `()` | 15 | `src/pinn_feynman/training/environment.py` (29) |
| `runtime_environment` | function | `()` | 26 | `src/pinn_feynman/training/run.py` (32, 108) |

### `pinn_feynman.training.history`

[`src/pinn_feynman/training/history.py`](../src/pinn_feynman/training/history.py) — Per-epoch training record.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `History` | class |  | 6 | `src/pinn_feynman/training/__init__.py` (3)<br>`src/pinn_feynman/training/trainer.py` (38, 94) |
| `History.append` | method | `(self, row)` | 12 | `src/pinn_feynman/data/build.py` (34)<br>`src/pinn_feynman/data/generation.py` (60, 65)<br>`src/pinn_feynman/devtools/codemap.py` (82, 85, 89, 94, 113, 165, 174, 219, …)<br>`src/pinn_feynman/devtools/list_configs.py` (55, 70, 74)<br>`src/pinn_feynman/models/mlp.py` (43)<br>`src/pinn_feynman/physics/derivatives.py` (26)<br>`src/pinn_feynman/providers/base.py` (85, 137)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (64, 142)<br>`src/pinn_feynman/reference/sunrise_equal_mass_positive.py` (80)<br>`src/pinn_feynman/systems/analysis.py` (40, 112, 124, 126, 152, 156)<br>`src/pinn_feynman/systems/blocks.py` (38, 42, 106, 133, 142, 144, 160, 163, …)<br>`src/pinn_feynman/training/history.py` (13)<br>`src/pinn_feynman/training/recorder.py` (190, 207, 217, 240, 241, 242, 243, 265, …)<br>`src/pinn_feynman/training/trainer.py` (282)<br>`src/pinn_feynman/visualization/singularities.py` (67, 93)<br>`src/pinn_feynman/visualization/style.py` (53)<br>`src/pinn_feynman/visualization/triangle.py` (187)<br>`src/pinn_feynman/visualization/tuning.py` (91) ⚠️ generic name |
| `History.column` | method | `(self, key)` | 15 | `src/pinn_feynman/training/recorder.py` (327, 331)<br>`src/pinn_feynman/visualization/triangle.py` (44, 45, 46, 47) ⚠️ generic name |
| `History.state_dict` | method | `(self)` | 21 | `src/pinn_feynman/training/recorder.py` (351)<br>`src/pinn_feynman/training/run.py` (78, 102)<br>`src/pinn_feynman/training/trainer.py` (267, 298, 299, 300, 301, 302, 303, 304, …) ⚠️ generic name |
| `History.load_state_dict` | method | `(self, state)` | 24 | `src/pinn_feynman/evaluation/predict.py` (92)<br>`src/pinn_feynman/training/run.py` (164)<br>`src/pinn_feynman/training/trainer.py` (315, 316, 317, 318, 319, 322, 323, 326) ⚠️ generic name |
| `History.save_csv` | method | `(self, path)` | 27 | `src/pinn_feynman/training/run.py` (79) |

### `pinn_feynman.training.optim`

[`src/pinn_feynman/training/optim.py`](../src/pinn_feynman/training/optim.py) — Optimizer and learning-rate scheduler from the `training` section of a configuration.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `ExponentialUntil` | class |  | 6 | `src/pinn_feynman/training/optim.py` (24) |
| `ExponentialUntil.get_lr` | method | `(self)` | 17 | `src/pinn_feynman/training/optim.py` (19) |
| `OPTIMIZERS` | constant |  | 23 | `src/pinn_feynman/training/optim.py` (41) |
| `SCHEDULERS` | constant |  | 24 | `src/pinn_feynman/training/optim.py` (47) |
| `build_optimizer` | function | `(config, parameters)` | 39 | `src/pinn_feynman/training/__init__.py` (4)<br>`src/pinn_feynman/training/setup.py` (17, 188) |
| `build_scheduler` | function | `(config, optimizer)` | 45 | `src/pinn_feynman/training/__init__.py` (4)<br>`src/pinn_feynman/training/setup.py` (17, 189) |

### `pinn_feynman.training.queue`

[`src/pinn_feynman/training/queue.py`](../src/pinn_feynman/training/queue.py) — Run a queue of trainings, idempotently.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `ConfigMismatch` | class |  | 54 | `src/pinn_feynman/training/queue.py` (87, 293) |
| `EXECUTION_KEYS` | constant |  | 68 | `src/pinn_feynman/training/queue.py` (77) |
| `EXECUTION_PREFIXES` | constant |  | 69 | `src/pinn_feynman/training/queue.py` (78) |
| `config_differences` | function | `(spec, directory)` | 72 | `src/pinn_feynman/training/queue.py` (85, 352) |
| `check_existing_run` | function | `(spec, directory)` | 81 | `src/pinn_feynman/training/queue.py` (254, 292) |
| `LocalSession` | class |  | 92 | `src/pinn_feynman/training/queue.py` (139) |
| `LocalSession.claim` | method | `(self, directory)` | 101 | `src/pinn_feynman/tuning/tune.py` (192) |
| `LocalSession.acquire` | method | `(self, spec, directory)` | 105 | `src/pinn_feynman/training/queue.py` (261) |
| `LocalSession.heartbeat` | method | `(self, directory)` | 109 | `src/pinn_feynman/training/queue.py` (274)<br>`src/pinn_feynman/tuning/tune.py` (212) |
| `LocalSession.release` | method | `(self, directory)` | 112 | `src/pinn_feynman/training/queue.py` (279)<br>`src/pinn_feynman/tuning/tune.py` (236) |
| `LocalSession.lost` | method | `(self, error)` | 115 | `src/pinn_feynman/training/queue.py` (277)<br>`src/pinn_feynman/tuning/tune.py` (214) |
| `LocalSession.listing` | method | `(self, base)` | 118 | `src/pinn_feynman/training/queue.py` (342, 354) |
| `LocalSession.status_note` | method | `(self, spec, directory, text, listing=None, tuning=False)` | 121 | `src/pinn_feynman/training/queue.py` (354) |
| `LocalSession.add_arguments` | method | `(parser)` | 125 | `src/pinn_feynman/training/queue.py` (365) |
| `LocalSession.options` | method | `(args)` | 129 | `src/pinn_feynman/data/generation.py` (48, 49, 50, 51)<br>`src/pinn_feynman/evaluation/evaluate.py` (61)<br>`src/pinn_feynman/training/queue.py` (98, 99, 234, 250, 310, 340, 367, 372, …)<br>`src/pinn_feynman/tuning/publish.py` (73, 74, 75)<br>`src/pinn_feynman/tuning/tune.py` (191) |
| `_SESSIONS` | constant |  | 134 | `src/pinn_feynman/training/queue.py` (139) |
| `session_class` | function | `()` | 137 | `src/pinn_feynman/training/queue.py` (234, 250, 310, 340, 364)<br>`src/pinn_feynman/tuning/tune.py` (183, 191) |
| `RunSpec` | class |  | 143 | `src/pinn_feynman/training/queue.py` (214, 297) |
| `RunSpec.explicit` | method | `(self)` | 152 | `src/pinn_feynman/runtime.py` (45, 46, 47)<br>`src/pinn_feynman/training/queue.py` (158, 167, 172) |
| `RunSpec.config_path` | method | `(self)` | 156 | `src/pinn_feynman/training/queue.py` (76, 83, 167, 172, 258, 259, 271, 350, …)<br>`src/pinn_feynman/training/run.py` (46, 129) |
| `RunSpec.problem` | method | `(self)` | 166 | `src/pinn_feynman/data/cache.py` (18, 30, 52, 72)<br>`src/pinn_feynman/data/dataset.py` (26, 47)<br>`src/pinn_feynman/data/generation.py` (19, 20, 24, 31, 33, 37, 48, 54)<br>`src/pinn_feynman/devtools/list_configs.py` (89)<br>`src/pinn_feynman/evaluation/boni_reference.py` (60, 65)<br>`src/pinn_feynman/evaluation/evaluate.py` (38, 46, 47, 88, 129)<br>`src/pinn_feynman/evaluation/predict.py` (23, 24, 25, 26, 30, 33, 38, 39, …)<br>`src/pinn_feynman/paths.py` (44, 49)<br>`src/pinn_feynman/physics/system.py` (24, 141, 142, 150, 153, 156, 157)<br>`src/pinn_feynman/problems/_registry.py` (108, 109, 111, 112, 113, 114, 118, 119, …)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (49, 58, 62)<br>`src/pinn_feynman/systems/singular_kinds.py` (52, 54, 60, 62)<br>`src/pinn_feynman/training/queue.py` (173, 240)<br>`src/pinn_feynman/training/setup.py` (131, 132, 137, 140)<br>`src/pinn_feynman/tuning/publish.py` (30, 71)<br>`src/pinn_feynman/visualization/singularities.py` (35, 40, 51, 54, 57, 59, 76, 80, …)<br>`src/pinn_feynman/weighting/distance.py` (40, 41) |
| `RunSpec.run_name` | method | `(self)` | 170 | `src/pinn_feynman/training/queue.py` (87, 176, 290, 294, 295, 318, 319, 320, …) |
| `RunSpec.directory` | method | `(self, base=None)` | 175 | `src/pinn_feynman/data/cache.py` (29, 30, 69, 78, 81)<br>`src/pinn_feynman/evaluation/evaluate.py` (34, 45, 48, 149, 150, 154, 158)<br>`src/pinn_feynman/evaluation/predict.py` (47, 88, 89, 92, 94, 96)<br>`src/pinn_feynman/training/queue.py` (75, 83, 85, 87, 235, 240, 253, 254, …)<br>`src/pinn_feynman/training/recorder.py` (406)<br>`src/pinn_feynman/training/run.py` (53, 54, 56, 57, 77, 78, 79, 91, …)<br>`src/pinn_feynman/tuning/tune.py` (71, 82, 86, 88, 90, 113, 154, 155, …) |
| `TuneSpec` | class |  | 180 | `src/pinn_feynman/training/queue.py` (213, 251, 317, 345, 354) |
| `TuneSpec.problem` | method | `(self)` | 187 | `src/pinn_feynman/data/cache.py` (18, 30, 52, 72)<br>`src/pinn_feynman/data/dataset.py` (26, 47)<br>`src/pinn_feynman/data/generation.py` (19, 20, 24, 31, 33, 37, 48, 54)<br>`src/pinn_feynman/devtools/list_configs.py` (89)<br>`src/pinn_feynman/evaluation/boni_reference.py` (60, 65)<br>`src/pinn_feynman/evaluation/evaluate.py` (38, 46, 47, 88, 129)<br>`src/pinn_feynman/evaluation/predict.py` (23, 24, 25, 26, 30, 33, 38, 39, …)<br>`src/pinn_feynman/paths.py` (44, 49)<br>`src/pinn_feynman/physics/system.py` (24, 141, 142, 150, 153, 156, 157)<br>`src/pinn_feynman/problems/_registry.py` (108, 109, 111, 112, 113, 114, 118, 119, …)<br>`src/pinn_feynman/reference/sunrise_equal_mass.py` (49, 58, 62)<br>`src/pinn_feynman/systems/singular_kinds.py` (52, 54, 60, 62)<br>`src/pinn_feynman/training/queue.py` (173, 240)<br>`src/pinn_feynman/training/setup.py` (131, 132, 137, 140)<br>`src/pinn_feynman/tuning/publish.py` (30, 71)<br>`src/pinn_feynman/visualization/singularities.py` (35, 40, 51, 54, 57, 59, 76, 80, …)<br>`src/pinn_feynman/weighting/distance.py` (40, 41) |
| `TuneSpec.tuning_file` | method | `(self)` | 191 | `src/pinn_feynman/training/queue.py` (197, 240) |
| `TuneSpec.tuning_path` | method | `(self)` | 196 | `src/pinn_feynman/training/queue.py` (205, 235)<br>`src/pinn_feynman/tuning/publish.py` (46, 73, 76)<br>`src/pinn_feynman/tuning/tune.py` (147, 184, 186) |
| `TuneSpec.run_name` | method | `(self)` | 200 | `src/pinn_feynman/training/queue.py` (87, 176, 290, 294, 295, 318, 319, 320, …) |
| `TuneSpec.directory` | method | `(self, base=None)` | 203 | `src/pinn_feynman/data/cache.py` (29, 30, 69, 78, 81)<br>`src/pinn_feynman/evaluation/evaluate.py` (34, 45, 48, 149, 150, 154, 158)<br>`src/pinn_feynman/evaluation/predict.py` (47, 88, 89, 92, 94, 96)<br>`src/pinn_feynman/training/queue.py` (75, 83, 85, 87, 235, 240, 253, 254, …)<br>`src/pinn_feynman/training/recorder.py` (406)<br>`src/pinn_feynman/training/run.py` (53, 54, 56, 57, 77, 78, 79, 91, …)<br>`src/pinn_feynman/tuning/tune.py` (71, 82, 86, 88, 90, 113, 154, 155, …) |
| `load_queue` | function | `(path)` | 208 | `src/pinn_feynman/training/queue.py` (368) |
| `with_diagnostics` | function | `(config, diagnostics)` | 218 | `src/pinn_feynman/training/queue.py` (271) |
| `run_tune` | function | `(spec, base=None, max_minutes=None, session=None, **options)` | 226 | `src/pinn_feynman/training/queue.py` (252) |
| `run_one` | function | `(spec, base=None, max_minutes=None, session=None, **options)` | 247 | `src/pinn_feynman/training/queue.py` (325) |
| `finished_runs` | function | `(specs, base=None, workers=16)` | 282 | `src/pinn_feynman/training/queue.py` (314) |
| `run_queue` | function | `(specs, base=None, max_minutes=None, **options)` | 308 | `src/pinn_feynman/training/queue.py` (375) |
| `print_table` | function | `(status)` | 333 | `src/pinn_feynman/training/queue.py` (329, 373) |
| `queue_status` | function | `(specs, base=None, **options)` | 338 | `src/pinn_feynman/training/queue.py` (372) |
| `main` | function | `(argv=None)` | 358 | `src/pinn_feynman/data/build.py` (76)<br>`src/pinn_feynman/data/check_interpolation.py` (46)<br>`src/pinn_feynman/devtools/benchmark.py` (68)<br>`src/pinn_feynman/devtools/codemap.py` (318)<br>`src/pinn_feynman/devtools/list_configs.py` (97)<br>`src/pinn_feynman/evaluation/evaluate.py` (167)<br>`src/pinn_feynman/systems/analysis.py` (171)<br>`src/pinn_feynman/training/queue.py` (379)<br>`src/pinn_feynman/training/run.py` (212)<br>`src/pinn_feynman/tuning/tune.py` (253) |

### `pinn_feynman.training.recorder`

[`src/pinn_feynman/training/recorder.py`](../src/pinn_feynman/training/recorder.py) — Training recorder: what happens during the training, observed without changing it.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `TAIL` | constant |  | 72 | `src/pinn_feynman/training/recorder.py` (142, 143, 290, 402) |
| `DEFAULTS` | constant |  | 73 | `src/pinn_feynman/training/recorder.py` (131, 135, 141, 459, 461) |
| `DIRECTORY` | constant |  | 75 | `src/pinn_feynman/training/recorder.py` (406) |
| `CELLS_FILE` | constant |  | 76 | `src/pinn_feynman/training/recorder.py` (411) |
| `GRADIENTS_FILE` | constant |  | 77 | `src/pinn_feynman/training/recorder.py` (413) |
| `POPULATION_GRADIENTS_FILE` | constant |  | 78 | `src/pinn_feynman/training/recorder.py` (415) |
| `POPULATION_KEYS` | constant |  | 79 | `src/pinn_feynman/training/recorder.py` (163, 422, 450) |
| `GRADIENT_KEYS` | constant |  | 80 | `src/pinn_feynman/training/recorder.py` (155, 215, 447) |
| `snapshot_name` | function | `(epoch)` | 83 | `src/pinn_feynman/training/recorder.py` (409) |
| `log_abs_range` | function | `(domain)` | 87 | `src/pinn_feynman/training/recorder.py` (99, 138) |
| `log_quadrature_grid` | function | `(domain, n)` | 95 | `src/pinn_feynman/training/recorder.py` (140) |
| `effective_points` | function | `(c)` | 110 | — |
| `TrainingRecorder` | class |  | 123 | `src/pinn_feynman/training/recorder.py` (462) |
| `TrainingRecorder.bind` | method | `(self, trainer)` | 166 | `src/pinn_feynman/training/trainer.py` (93) |
| `TrainingRecorder.on_batch` | method | `(self, idx, z, norms, weights)` | 185 | `src/pinn_feynman/training/trainer.py` (162) |
| `TrainingRecorder.wants_gradients` | method | `(self)` | 194 | `src/pinn_feynman/training/trainer.py` (174) |
| `TrainingRecorder.on_gradients` | method | `(self, trainer, terms)` | 198 | `src/pinn_feynman/training/trainer.py` (175) |
| `TrainingRecorder.grid_residual` | method | `(self, model)` | 297 | `src/pinn_feynman/training/recorder.py` (306) |
| `TrainingRecorder.end_epoch` | method | `(self, trainer, trained_population=None, trained_points=None)` | 304 | `src/pinn_feynman/training/trainer.py` (252, 281) |
| `TrainingRecorder.snapshot` | method | `(self, trainer, grid_r, trained_population=None, trained_points=None)` | 344 | `src/pinn_feynman/training/recorder.py` (340) |
| `TrainingRecorder.cell_arrays` | method | `(self)` | 394 | `src/pinn_feynman/training/recorder.py` (411) |
| `TrainingRecorder.flush` | method | `(self, directory)` | 404 | `src/pinn_feynman/progress.py` (46, 93)<br>`src/pinn_feynman/training/run.py` (77) |
| `TrainingRecorder.population_arrays` | method | `(self)` | 417 | `src/pinn_feynman/training/recorder.py` (415) |
| `TrainingRecorder.gradient_arrays` | method | `(self)` | 424 | `src/pinn_feynman/training/recorder.py` (413) |
| `TrainingRecorder.state_dict` | method | `(self)` | 430 | `src/pinn_feynman/training/recorder.py` (351)<br>`src/pinn_feynman/training/run.py` (78, 102)<br>`src/pinn_feynman/training/trainer.py` (267, 298, 299, 300, 301, 302, 303, 304, …) ⚠️ generic name |
| `TrainingRecorder.load_state_dict` | method | `(self, state)` | 437 | `src/pinn_feynman/evaluation/predict.py` (92)<br>`src/pinn_feynman/training/run.py` (164)<br>`src/pinn_feynman/training/trainer.py` (315, 316, 317, 318, 319, 322, 323, 326) ⚠️ generic name |
| `build_recorder` | function | `(config, domain, input_scale, epochs)` | 453 | `src/pinn_feynman/training/setup.py` (18, 196) |

### `pinn_feynman.training.run`

[`src/pinn_feynman/training/run.py`](../src/pinn_feynman/training/run.py) — Train one PINN (real or imaginary part) from a configuration, with checkpoints.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `CHECKPOINT` | constant |  | 35 | `src/pinn_feynman/training/run.py` (56, 78, 125) |
| `SUMMARY` | constant |  | 36 | `src/pinn_feynman/training/run.py` (54, 109) |
| `TimeBudgetReached` | class |  | 39 | `src/pinn_feynman/training/run.py` (93, 174) |
| `run_directory` | function | `(config, config_path, part, out=None)` | 43 | `src/pinn_feynman/training/run.py` (152) |
| `run_status` | function | `(directory)` | 49 | `src/pinn_feynman/training/queue.py` (51, 83, 255, 265, 289, 349)<br>`src/pinn_feynman/training/run.py` (148, 202) |
| `atomic_torch_save` | function | `(obj, path)` | 62 | `src/pinn_feynman/training/run.py` (78) |
| `save_checkpoint` | function | `(directory, trainer)` | 69 | `src/pinn_feynman/training/run.py` (91, 101) |
| `checkpoint_callback` | function | `(directory, every, deadline=None, before_checkpoint=None)` | 82 | `src/pinn_feynman/training/run.py` (172) |
| `save_run` | function | `(directory, experiment, elapsed)` | 97 | `src/pinn_feynman/training/run.py` (177) |
| `write_config` | function | `(directory, config, part)` | 114 | `src/pinn_feynman/training/queue.py` (51, 272)<br>`src/pinn_feynman/training/run.py` (153) |
| `load_run_state` | function | `(directory)` | 120 | `src/pinn_feynman/training/queue.py` (51, 269)<br>`src/pinn_feynman/training/run.py` (141, 149) |
| `new_config` | function | `(config_path, epochs=None, seed=None)` | 128 | `src/pinn_feynman/training/queue.py` (51, 76, 271)<br>`src/pinn_feynman/training/run.py` (145) |
| `prepare` | function | `(args)` | 137 | `src/pinn_feynman/training/run.py` (205) |
| `train` | function | `(config, part, directory, state=None, max_minutes=None, before_checkpoint=None)` | 157 | `src/pinn_feynman/training/queue.py` (51, 273)<br>`src/pinn_feynman/training/run.py` (207)<br>`src/pinn_feynman/training/trainer.py` (251, 262, 268) |
| `print_summary` | function | `(summary)` | 182 | `src/pinn_feynman/training/run.py` (178)<br>`src/pinn_feynman/tuning/tune.py` (233) |
| `main` | function | `(argv=None)` | 190 | `src/pinn_feynman/data/build.py` (76)<br>`src/pinn_feynman/data/check_interpolation.py` (46)<br>`src/pinn_feynman/devtools/benchmark.py` (68)<br>`src/pinn_feynman/devtools/codemap.py` (318)<br>`src/pinn_feynman/devtools/list_configs.py` (97)<br>`src/pinn_feynman/evaluation/evaluate.py` (167)<br>`src/pinn_feynman/systems/analysis.py` (171)<br>`src/pinn_feynman/training/queue.py` (379)<br>`src/pinn_feynman/training/run.py` (212)<br>`src/pinn_feynman/tuning/tune.py` (253) |

### `pinn_feynman.training.setup`

[`src/pinn_feynman/training/setup.py`](../src/pinn_feynman/training/setup.py) — Assemble all the pieces of an experiment from a configuration.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `SEED_OFFSETS` | constant |  | 22 | `src/pinn_feynman/training/setup.py` (178, 185, 187) |
| `Experiment` | class |  | 26 | `src/pinn_feynman/training/setup.py` (198) |
| `resolve_device` | function | `(name)` | 33 | `src/pinn_feynman/training/setup.py` (192) |
| `to_point_set` | function | `(dataset, part, input_scale, dtype, columns=None)` | 45 | `src/pinn_feynman/training/setup.py` (174, 175) |
| `load_datasets` | function | `(config, names=('boundary', 'validation'))` | 56 | `src/pinn_feynman/training/setup.py` (172)<br>`src/pinn_feynman/tuning/tune.py` (33, 226) |
| `check_collocation` | function | `(config)` | 61 | `src/pinn_feynman/training/setup.py` (165) |
| `check_variables` | function | `(config)` | 70 | `src/pinn_feynman/training/setup.py` (163) |
| `check_part` | function | `(config, part)` | 94 | `src/pinn_feynman/training/setup.py` (166) |
| `check_active_orders` | function | `(config)` | 106 | `src/pinn_feynman/training/setup.py` (168) |
| `check_provider_cost` | function | `(config)` | 128 | `src/pinn_feynman/training/setup.py` (167) |
| `check_multi_variable` | function | `(config)` | 143 | `src/pinn_feynman/training/setup.py` (164) |
| `build_experiment` | function | `(config, part, datasets=None, seed=None)` | 161 | `src/pinn_feynman/devtools/benchmark.py` (18, 55)<br>`src/pinn_feynman/training/__init__.py` (5)<br>`src/pinn_feynman/training/run.py` (33, 160)<br>`src/pinn_feynman/tuning/objective.py` (6, 32) |

### `pinn_feynman.training.trainer`

[`src/pinn_feynman/training/trainer.py`](../src/pinn_feynman/training/trainer.py) — Training loop of Boni's PINN (thesis sec. 4.4, 6.1.2, Table 6.1).

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `PointSet` | class |  | 42 | `src/pinn_feynman/training/__init__.py` (6)<br>`src/pinn_feynman/training/setup.py` (19, 53)<br>`src/pinn_feynman/training/trainer.py` (49) |
| `PointSet.to` | method | `(self, device)` | 48 | `src/pinn_feynman/balancing/fixed.py` (25)<br>`src/pinn_feynman/balancing/relobralo.py` (93, 100)<br>`src/pinn_feynman/collocation/r3.py` (55, 73, 112, 124)<br>`src/pinn_feynman/collocation/sampler.py` (62, 124, 129)<br>`src/pinn_feynman/data/domains.py` (75, 147)<br>`src/pinn_feynman/models/factory.py` (47)<br>`src/pinn_feynman/physics/bottom_up.py` (102)<br>`src/pinn_feynman/training/recorder.py` (207, 225, 265, 274, 279, 280, 292, 380, …)<br>`src/pinn_feynman/training/trainer.py` (49, 57, 58, 59, 60, 67, 142, 152)<br>`src/pinn_feynman/weighting/distance.py` (168, 169) ⚠️ generic name |
| `Trainer` | class |  | 52 | `src/pinn_feynman/training/__init__.py` (6)<br>`src/pinn_feynman/training/setup.py` (19, 29, 190) |
| `Trainer.a0_base` | method | `(self)` | 109 | — |
| `Trainer.a1_base` | method | `(self)` | 113 | — |
| `Trainer.a0_grid` | method | `(self)` | 117 | — |
| `Trainer.a1_grid` | method | `(self)` | 121 | — |
| `Trainer.a0_val` | method | `(self)` | 125 | — |
| `Trainer.a1_val` | method | `(self)` | 129 | — |
| `Trainer.residual` | method | `(self, idx)` | 133 | `src/pinn_feynman/physics/losses.py` (30, 33, 35, 36, 51, 57, 58)<br>`src/pinn_feynman/training/trainer.py` (143, 154, 156, 168) |
| `Trainer.residual_norms` | method | `(self, idx, chunk=16384)` | 140 | `src/pinn_feynman/training/trainer.py` (201) |
| `Trainer.losses` | method | `(self, idx)` | 145 | `src/pinn_feynman/balancing/fixed.py` (25)<br>`src/pinn_feynman/balancing/relobralo.py` (65, 97, 99, 100)<br>`src/pinn_feynman/training/trainer.py` (172) ⚠️ generic name |
| `Trainer.train_step` | method | `(self, idx)` | 170 | `src/pinn_feynman/devtools/benchmark.py` (26, 30)<br>`src/pinn_feynman/training/trainer.py` (194) |
| `Trainer.train_epoch` | method | `(self)` | 180 | `src/pinn_feynman/training/trainer.py` (251) |
| `Trainer.residual_stats` | method | `(self)` | 205 | `src/pinn_feynman/training/trainer.py` (279) |
| `Trainer.validate` | method | `(self)` | 213 | `src/pinn_feynman/training/trainer.py` (258) ⚠️ generic name |
| `Trainer.validate_physics` | method | `(self)` | 218 | `src/pinn_feynman/training/trainer.py` (259) |
| `Trainer.fit` | method | `(self, epochs, on_epoch_end=None, desc='train', leave=True)` | 225 | `src/pinn_feynman/training/run.py` (171)<br>`src/pinn_feynman/tuning/objective.py` (33) ⚠️ generic name |
| `Trainer.state_dict` | method | `(self)` | 295 | `src/pinn_feynman/training/recorder.py` (351)<br>`src/pinn_feynman/training/run.py` (78, 102)<br>`src/pinn_feynman/training/trainer.py` (267, 298, 299, 300, 301, 302, 303, 304, …) ⚠️ generic name |
| `Trainer.load_state_dict` | method | `(self, state)` | 313 | `src/pinn_feynman/evaluation/predict.py` (92)<br>`src/pinn_feynman/training/run.py` (164)<br>`src/pinn_feynman/training/trainer.py` (315, 316, 317, 318, 319, 322, 323, 326) ⚠️ generic name |

### `pinn_feynman.tuning`

[`src/pinn_feynman/tuning/__init__.py`](../src/pinn_feynman/tuning/__init__.py) — Hyperparameter tuning with Optuna (per-problem YAML search space).

### `pinn_feynman.tuning.figures`

[`src/pinn_feynman/tuning/figures.py`](../src/pinn_feynman/tuning/figures.py) — Re-export of `pinn_feynman.visualization.tuning.tuning_history`.

### `pinn_feynman.tuning.objective`

[`src/pinn_feynman/tuning/objective.py`](../src/pinn_feynman/tuning/objective.py) — Objective of a tuning trial: short training, best validation loss, pruning.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `PruningCallback` | class |  | 10 | `src/pinn_feynman/tuning/__init__.py` (3)<br>`src/pinn_feynman/tuning/objective.py` (33) |
| `PruningCallback.__call__` | method | `(self, trainer)` | 16 | — |
| `make_objective` | function | `(base_config, budget, space, part, datasets)` | 22 | `src/pinn_feynman/tuning/__init__.py` (3)<br>`src/pinn_feynman/tuning/tune.py` (34, 225) |

### `pinn_feynman.tuning.publish`

[`src/pinn_feynman/tuning/publish.py`](../src/pinn_feynman/tuning/publish.py) — Write the best configuration of a finished study as a NEW configuration file.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `TUNED_PATH` | constant |  | 19 | `src/pinn_feynman/tuning/publish.py` (34) |
| `_REMOTE` | constant |  | 21 | `src/pinn_feynman/tuning/publish.py` (72, 73) |
| `PublishError` | class |  | 24 | `src/pinn_feynman/training/queue.py` (232, 241)<br>`src/pinn_feynman/tuning/publish.py` (35, 60) |
| `tuned_config_path` | function | `(problem, study_name)` | 28 | `src/pinn_feynman/tuning/publish.py` (71) |
| `check_path` | function | `(path)` | 33 | `src/pinn_feynman/tuning/publish.py` (57) |
| `tuned_config_text` | function | `(study_dir, tuning_path)` | 39 | `src/pinn_feynman/tuning/publish.py` (76) |
| `write_local` | function | `(path, text, root=None)` | 55 | `src/pinn_feynman/tuning/publish.py` (76) |
| `publish_study` | function | `(study_dir, tuning_path, problem, root=None, **options)` | 67 | `src/pinn_feynman/training/queue.py` (232, 240)<br>`src/pinn_feynman/tuning/publish.py` (73) |

### `pinn_feynman.tuning.space`

[`src/pinn_feynman/tuning/space.py`](../src/pinn_feynman/tuning/space.py) — Search space of a tuning study: YAML description -> Optuna suggestions -> configuration.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `DISTRIBUTIONS` | constant |  | 16 | `src/pinn_feynman/tuning/space.py` (31) |
| `suggest` | function | `(trial, space)` | 19 | `src/pinn_feynman/tuning/__init__.py` (4)<br>`src/pinn_feynman/tuning/objective.py` (7, 28) |
| `set_path` | function | `(config, path, value)` | 35 | `src/pinn_feynman/tuning/__init__.py` (4)<br>`src/pinn_feynman/tuning/space.py` (48) |
| `apply` | function | `(config, values)` | 44 | `src/pinn_feynman/tuning/__init__.py` (4)<br>`src/pinn_feynman/tuning/space.py` (59)<br>`src/pinn_feynman/tuning/tune.py` (36, 80) |
| `trial_config` | function | `(base_config, budget, values)` | 57 | `src/pinn_feynman/tuning/__init__.py` (4)<br>`src/pinn_feynman/tuning/objective.py` (7, 29) |

### `pinn_feynman.tuning.tune`

[`src/pinn_feynman/tuning/tune.py`](../src/pinn_feynman/tuning/tune.py) — Run (or resume) a tuning study, idempotently.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `FINISHED` | constant |  | 38 | `src/pinn_feynman/tuning/tune.py` (63) |
| `SAMPLERS` | constant |  | 40 | `src/pinn_feynman/data/sampling.py` (33, 43, 45)<br>`src/pinn_feynman/tuning/tune.py` (54) |
| `PRUNERS` | constant |  | 41 | `src/pinn_feynman/tuning/tune.py` (56) |
| `load_tuning_config` | function | `(path)` | 44 | `src/pinn_feynman/tuning/tune.py` (184) |
| `create_study` | function | `(study_cfg, storage)` | 53 | `src/pinn_feynman/tuning/tune.py` (57, 202) |
| `n_finished` | function | `(study)` | 61 | `src/pinn_feynman/tuning/tune.py` (73, 131, 222, 223) |
| `write_outputs` | function | `(study, directory, base, budget, full_training, target=None)` | 66 | `src/pinn_feynman/tuning/tune.py` (231) |
| `trial_line` | function | `(study, trial, target)` | 123 | `src/pinn_feynman/tuning/tune.py` (209) |
| `print_summary` | function | `(summary)` | 135 | `src/pinn_feynman/training/run.py` (178)<br>`src/pinn_feynman/tuning/tune.py` (233) |
| `study_directory` | function | `(tuning_path, out=None)` | 145 | `src/pinn_feynman/training/queue.py` (204, 205)<br>`src/pinn_feynman/tuning/tune.py` (186) |
| `study_status` | function | `(directory)` | 152 | `src/pinn_feynman/training/queue.py` (346, 347)<br>`src/pinn_feynman/tuning/tune.py` (188) |
| `fail_stale_trials` | function | `(study)` | 161 | `src/pinn_feynman/tuning/tune.py` (203) |
| `copy_atomic` | function | `(source, target)` | 169 | `src/pinn_feynman/tuning/tune.py` (204, 210, 230) |
| `run_study` | function | `(tuning_path, out=None, n_trials=None, max_minutes=None, session=None, **options)` | 176 | `src/pinn_feynman/training/queue.py` (233, 235)<br>`src/pinn_feynman/tuning/tune.py` (248) |
| `main` | function | `(argv=None)` | 240 | `src/pinn_feynman/data/build.py` (76)<br>`src/pinn_feynman/data/check_interpolation.py` (46)<br>`src/pinn_feynman/devtools/benchmark.py` (68)<br>`src/pinn_feynman/devtools/codemap.py` (318)<br>`src/pinn_feynman/devtools/list_configs.py` (97)<br>`src/pinn_feynman/evaluation/evaluate.py` (167)<br>`src/pinn_feynman/systems/analysis.py` (171)<br>`src/pinn_feynman/training/queue.py` (379)<br>`src/pinn_feynman/training/run.py` (212)<br>`src/pinn_feynman/tuning/tune.py` (253) |

### `pinn_feynman.visualization`

[`src/pinn_feynman/visualization/__init__.py`](../src/pinn_feynman/visualization/__init__.py) — Every figure of the project: pure functions, data in, PNG out, one style (`style`).

### `pinn_feynman.visualization.accuracy`

[`src/pinn_feynman/visualization/accuracy.py`](../src/pinn_feynman/visualization/accuracy.py) — Accuracy of the network against the exact solution (Boni's Fig. 6.3-6.7): curves, errors along x and by

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `error_histograms` | function | `(errors, orders, path, title, xlabel, reference_mean=None)` | 19 | `src/pinn_feynman/evaluation/evaluate.py` (66, 68, 109, 111)<br>`src/pinn_feynman/evaluation/plots.py` (5) |
| `real_vs_imag` | function | `(abs_real, abs_imag, path)` | 44 | `src/pinn_feynman/evaluation/evaluate.py` (162)<br>`src/pinn_feynman/evaluation/plots.py` (7) |
| `solution_curves` | function | `(x_boni, prediction, target, orders, path, title, shown_orders=(-2, -1, 0), xlabel=None, master_labels=None, singular=None)` | 62 | `src/pinn_feynman/evaluation/evaluate.py` (73)<br>`src/pinn_feynman/evaluation/plots.py` (7) |
| `error_vs_x` | function | `(x_boni, errors, orders, path, title, n_bins=40, xlabel=None, singular=None)` | 89 | `src/pinn_feynman/evaluation/evaluate.py` (77)<br>`src/pinn_feynman/evaluation/plots.py` (6) |
| `component_heatmap` | function | `(errors, orders, path, title, master_labels=None)` | 111 | `src/pinn_feynman/evaluation/evaluate.py` (63, 107)<br>`src/pinn_feynman/evaluation/plots.py` (5) |
| `error_vs_distance` | function | `(x_boni, abs_err, rel_err, orders, path, title, n_bins=30, xlabel=None)` | 141 | `src/pinn_feynman/evaluation/evaluate.py` (79)<br>`src/pinn_feynman/evaluation/plots.py` (6) |
| `error_map` | function | `(points, errors, orders, path, title, outline=None, axis_labels=None, gridsize=30, colorbar_label='mean absolute error', singular_lines=None)` | 165 | `src/pinn_feynman/evaluation/evaluate.py` (116)<br>`src/pinn_feynman/evaluation/plots.py` (6) |
| `error_vs_edge_distance` | function | `(distances, abs_err, rel_err, orders, path, title, edge_labels=None, n_bins=20)` | 205 | `src/pinn_feynman/evaluation/evaluate.py` (120)<br>`src/pinn_feynman/evaluation/plots.py` (6) |
| `compare_errors` | function | `(rows, path)` | 245 | `src/pinn_feynman/evaluation/plots.py` (5) |
| `compare_error_vs_x` | function | `(curves, path, n_bins=40, singular=None)` | 261 | `src/pinn_feynman/evaluation/plots.py` (4) |
| `compare_residuals` | function | `(curves, path, n_bins=80, singular=None)` | 278 | `src/pinn_feynman/evaluation/plots.py` (5) |
| `compare_error_vs_distance` | function | `(curves, path, n_bins=30)` | 296 | `src/pinn_feynman/evaluation/plots.py` (3) |
| `compare_error_maps` | function | `(points, per_group, path, outline=None, axis_labels=None, gridsize=30, colorbar_label='mean absolute error', singular_lines=None)` | 315 | `src/pinn_feynman/evaluation/plots.py` (3) |
| `compare_error_vs_edge_distance` | function | `(distances, per_group, path, edge_labels=None, n_bins=20, colours=None)` | 353 | `src/pinn_feynman/evaluation/plots.py` (4) |

### `pinn_feynman.visualization.components`

[`src/pinn_feynman/visualization/components.py`](../src/pinn_feynman/visualization/components.py) — Figures per component M_i^(k) of the network output: the masters and the orders in epsilon separately.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `ORDER_LABELS` | constant |  | 14 | `src/pinn_feynman/visualization/components.py` (69, 115, 227) |
| `component_grid` | function | `(x, groups, path, title, reference=None, ylog=False, xlog=False, ylabel='', max_points=1500, order_labels=ORDER_LABELS, master_labels=None, singular=None, signed=False, mean=False, xlabel=None)` | 68 | — |
| `per_order_cells` | function | `(centres, groups, path, title, ylabel, ylog=True, xlog=False, references=None, order_labels=ORDER_LABELS, xlabel='|x|', singular=None)` | 114 | — |
| `segment_grid` | function | `(curves, path, master_labels, order_labels, singular=None, shade=None, xlabel='x', network=None, ylabel='')` | 149 | — |
| `shade_spans` | function | `(ax, shade)` | 192 | `src/pinn_feynman/visualization/components.py` (164, 208)<br>`src/pinn_feynman/visualization/triangle.py` (250, 258) |
| `block_curves` | function | `(x, series, path, title, ylabel, ylog=True, references=None, xlabel='x', bands=None, singular=None, shade=None)` | 200 | — |
| `per_order_blocks` | function | `(x, groups, path, title, ylabel, ylog=True, order_labels=ORDER_LABELS, xlabel='x', bands=None, singular=None)` | 227 | — |
| `cell_curves` | function | `(centres, series, path, title, ylabel, ylog=True, xlog=False, references=None, xlabel='|x|', singular=None)` | 261 | — |
| `component_errors` | function | `(errors, orders, path, title, master_labels=None, ylabel='mean absolute error')` | 283 | — |

### `pinn_feynman.visualization.data_io`

[`src/pinn_feynman/visualization/data_io.py`](../src/pinn_feynman/visualization/data_io.py) — The data behind the figures, saved next to a report (`report/figure_data/`) so that a figure can be redrawn without

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `JSON_KEY` | constant |  | 13 | `src/pinn_feynman/visualization/data_io.py` (42, 63, 64) |
| `POINTWISE` | constant |  | 15 | `src/pinn_feynman/visualization/data_io.py` (71) |
| `save_data` | function | `(path, data)` | 36 | — |
| `load_data` | function | `(path)` | 60 | — |
| `slim` | function | `(data)` | 68 | `src/pinn_feynman/visualization/data_io.py` (71, 73) |
| `history_columns` | function | `(rows)` | 77 | `src/pinn_feynman/training/trainer.py` (289) |
| `history_rows` | function | `(columns)` | 83 | `src/pinn_feynman/visualization/training.py` (15, 16, 17, 29, 30, 34, 35, 36, …) |

### `pinn_feynman.visualization.methods`

[`src/pinn_feynman/visualization/methods.py`](../src/pinn_feynman/visualization/methods.py) — Colours and styles of the training methods, shared by the figures.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `METHOD_COLOUR` | constant |  | 13 | `src/pinn_feynman/visualization/methods.py` (22, 79) |
| `METHOD_LABEL` | constant |  | 14 | `src/pinn_feynman/visualization/methods.py` (23) |
| `FAMILIES` | constant |  | 16 | `src/pinn_feynman/visualization/methods.py` (24, 45) |
| `_EXTRA` | constant |  | 19 | `src/pinn_feynman/visualization/methods.py` (20, 21, 88)<br>`src/pinn_feynman/visualization/training.py` (73, 74, 75) |
| `LINESTYLES` | constant |  | 27 | — |
| `MARKERS` | constant |  | 28 | `src/pinn_feynman/visualization/methods.py` (85) |
| `PART_COLOUR` | constant |  | 29 | `src/pinn_feynman/visualization/components.py` (156, 167, 174)<br>`src/pinn_feynman/visualization/methods.py` (72, 73) |
| `REFERENCE_COLOUR` | constant |  | 30 | `src/pinn_feynman/visualization/methods.py` (70) |
| `TINTS` | constant |  | 32 | `src/pinn_feynman/visualization/methods.py` (78) |
| `set_reference` | function | `(label)` | 37 | — |
| `family_of` | function | `(label)` | 43 | `src/pinn_feynman/visualization/methods.py` (75) |
| `tint` | function | `(colour, toward_white=0.0, toward_black=0.0)` | 53 | `src/pinn_feynman/visualization/methods.py` (79) |
| `group_styles` | function | `(labels, reference=None)` | 61 | `src/pinn_feynman/visualization/components.py` (18, 19)<br>`src/pinn_feynman/visualization/training.py` (109, 110)<br>`src/pinn_feynman/visualization/triangle.py` (251, 252) |
| `group_markers` | function | `(labels)` | 83 | — |

### `pinn_feynman.visualization.singularities`

[`src/pinn_feynman/visualization/singularities.py`](../src/pinn_feynman/visualization/singularities.py) — Dashed vertical lines at the singular points of a problem, on every figure with the domain on the x axis.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `PHYSICAL` | constant |  | 22 | `src/pinn_feynman/visualization/singularities.py` (98) |
| `SPURIOUS` | constant |  | 23 | `src/pinn_feynman/visualization/singularities.py` (98) |
| `DASHES` | constant |  | 24 | `src/pinn_feynman/visualization/singularities.py` (113, 137) |
| `SYSTEM_MODULES` | constant |  | 31 | `src/pinn_feynman/systems/analysis.py` (60, 141, 163)<br>`src/pinn_feynman/visualization/singularities.py` (35) |
| `positions` | function | `(problem)` | 38 | `src/pinn_feynman/models/factory.py` (53, 54, 56)<br>`src/pinn_feynman/visualization/singularities.py` (57) |
| `singular_points` | function | `(problem, scale=1.0, absolute=False, sign=-1.0, name=None)` | 45 | `src/pinn_feynman/evaluation/evaluate.py` (86, 88) |
| `singular_lines` | function | `(problem, s0=1)` | 71 | `src/pinn_feynman/evaluation/evaluate.py` (128, 129)<br>`src/pinn_feynman/visualization/accuracy.py` (188, 337) |
| `colour` | function | `(kind)` | 97 | `src/pinn_feynman/visualization/accuracy.py` (373, 379, 380, 383)<br>`src/pinn_feynman/visualization/components.py` (27, 30, 32, 43, 46, 50, 57, 167, …)<br>`src/pinn_feynman/visualization/methods.py` (55)<br>`src/pinn_feynman/visualization/singularities.py` (113, 115, 137, 140)<br>`src/pinn_feynman/visualization/training.py` (135, 142, 146, 148)<br>`src/pinn_feynman/visualization/triangle.py` (221, 225, 227, 261, 267, 269) |
| `mark` | function | `(ax, points, label_y=0.97)` | 101 | `src/pinn_feynman/visualization/accuracy.py` (9, 80, 102, 271, 289)<br>`src/pinn_feynman/visualization/components.py` (11, 97, 137, 178, 216, 249, 272) |
| `mark_lines` | function | `(ax, lines)` | 120 | `src/pinn_feynman/visualization/accuracy.py` (9, 188, 337) |

### `pinn_feynman.visualization.style`

[`src/pinn_feynman/visualization/style.py`](../src/pinn_feynman/visualization/style.py) — Style shared by every figure of the project: colours, matplotlib settings, saving, warnings.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `SERIES` | constant |  | 15 | `src/pinn_feynman/evaluation/plots.py` (8)<br>`src/pinn_feynman/visualization/accuracy.py` (10, 27, 50, 76, 100, 155, 229, 250, …)<br>`src/pinn_feynman/visualization/components.py` (12, 133, 167, 174, 211, 212, 241, 268)<br>`src/pinn_feynman/visualization/methods.py` (10, 13, 29)<br>`src/pinn_feynman/visualization/training.py` (8, 16, 17, 35, 36, 40, 46, 48, …)<br>`src/pinn_feynman/visualization/triangle.py` (16, 221)<br>`src/pinn_feynman/visualization/tuning.py` (6, 16, 18, 24, 36, 63, 64, 139) |
| `STYLE` | constant |  | 18 | `src/pinn_feynman/evaluation/plots.py` (8)<br>`src/pinn_feynman/visualization/accuracy.py` (10, 21, 46, 69, 95, 121, 149, 177, …)<br>`src/pinn_feynman/visualization/components.py` (12, 85, 124, 159, 206, 236, 265, 292)<br>`src/pinn_feynman/visualization/training.py` (8, 13, 31, 60, 84, 179, 209, 263, …)<br>`src/pinn_feynman/visualization/triangle.py` (16, 34, 146, 193, 217, 254)<br>`src/pinn_feynman/visualization/tuning.py` (6, 14, 34, 51, 94, 114, 135) |
| `order_label` | function | `(order)` | 27 | `src/pinn_feynman/evaluation/plots.py` (8)<br>`src/pinn_feynman/visualization/accuracy.py` (10, 32, 79, 100, 125, 155, 189, 229)<br>`src/pinn_feynman/visualization/triangle.py` (16, 61) |
| `set_domain` | function | `(lo=None, hi=None)` | 35 | — |
| `domain_box` | function | `(ax)` | 41 | `src/pinn_feynman/visualization/components.py` (12, 98, 138, 217, 250, 273) |
| `collect` | function | `()` | 64 | — |
| `quiet` | function | `()` | 75 | — |

### `pinn_feynman.visualization.training`

[`src/pinn_feynman/visualization/training.py`](../src/pinn_feynman/visualization/training.py) — Figures along the epochs: losses, training dynamics, adaptive-method columns.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `loss_curves` | function | `(history_rows, path, title)` | 11 | `src/pinn_feynman/evaluation/evaluate.py` (53, 104)<br>`src/pinn_feynman/evaluation/plots.py` (9) |
| `training_dynamics` | function | `(history_rows, path, title)` | 26 | `src/pinn_feynman/evaluation/evaluate.py` (54, 105)<br>`src/pinn_feynman/evaluation/plots.py` (9) |
| `compare_val_loss` | function | `(histories, path)` | 58 | `src/pinn_feynman/evaluation/plots.py` (9) |
| `ADAPTIVE_COLUMNS` | constant |  | 72 | `src/pinn_feynman/visualization/training.py` (75, 80) |
| `_EXTRA` | constant |  | 73 | `src/pinn_feynman/visualization/methods.py` (19, 20, 21, 88)<br>`src/pinn_feynman/visualization/training.py` (74, 75) |
| `adaptive_columns` | function | `(history, path, title)` | 78 | — |
| `moving_average` | function | `(epochs, values, window, log=True)` | 113 | `src/pinn_feynman/visualization/training.py` (137) |
| `MAX_GROUPS_WITH_SEEDS` | constant |  | 125 | `src/pinn_feynman/visualization/training.py` (144) |
| `TRAINING_LOSSES` | constant |  | 170 | `src/pinn_feynman/visualization/training.py` (183, 194) |
| `loss_training` | function | `(groups, path, title, smooth=1)` | 175 | — |
| `loss_validation` | function | `(groups, path, title, markers=None, smooth=1)` | 203 | — |
| `relobralo` | function | `(groups, path, title, smooth=1)` | 254 | — |
| `learning_rate` | function | `(groups, path, title)` | 276 | — |
| `r3_retained` | function | `(groups, sizes, path, title)` | 288 | — |
| `pde_sampling_bias` | function | `(groups, path, title)` | 313 | — |
| `gradients` | function | `(groups, path, title, smooth=1)` | 332 | `src/pinn_feynman/training/recorder.py` (149, 151, 152, 155, 217, 326, 330, 427, …) |

### `pinn_feynman.visualization.triangle`

[`src/pinn_feynman/visualization/triangle.py`](../src/pinn_feynman/visualization/triangle.py) — Maps on a two-variable domain with a grid of panels: rows = groups (e.g. the real and imaginary parts), columns =

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `order_maps` | function | `(points, rows, orders, path, outline=None, axis_labels=None, colorbar_label='', gridsize=30, overlays=None, empty_text='identically zero')` | 19 | — |
| `cell_maps` | function | `(polygons, rows, columns, path, outline=None, overlays=None, lines=None, axis_labels=None, colorbar_label='', shared=False, empty_text='identically zero')` | 124 | — |
| `ratio_maps` | function | `(polygons, panels, path, outline=None, overlays=None, lines=None, axis_labels=None, colorbar_label='')` | 175 | — |
| `edge_curves` | function | `(centres, groups, path, edge_labels, ylabel, ylog=True, references=None, colours=None)` | 210 | — |
| `cut_curves` | function | `(cuts, groups, path, ylabel, ylog=True, references=None, shade=None, reference=None, xlabel='s1')` | 245 | — |

### `pinn_feynman.visualization.tuning`

[`src/pinn_feynman/visualization/tuning.py`](../src/pinn_feynman/visualization/tuning.py) — Figures of a tuning study (log scale: values span many decades).

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `tuning_history` | function | `(study, path, reference=None)` | 9 | `src/pinn_feynman/tuning/figures.py` (3)<br>`src/pinn_feynman/tuning/tune.py` (35, 88) |
| `importance_bars` | function | `(importances, path, title)` | 32 | — |
| `slices` | function | `(trials, params, path, title)` | 49 | — |
| `parallel_coordinates` | function | `(trials, params, path, title, top=None)` | 76 | — |
| `contour_pair` | function | `(trials, pair, path, title)` | 107 | — |
| `category_boxes` | function | `(trials, name, path, title)` | 128 | — |

### `pinn_feynman.weighting`

[`src/pinn_feynman/weighting/__init__.py`](../src/pinn_feynman/weighting/__init__.py) — Per-point weights of the physics loss (distance from the singular points); default: none.

### `pinn_feynman.weighting.distance`

[`src/pinn_feynman/weighting/distance.py`](../src/pinn_feynman/weighting/distance.py) — Distance weighting (DW): physics residual multiplied by the distance from the nearest pole of the block A_UU.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `NORMALISATION_POINTS` | constant |  | 33 | `src/pinn_feynman/weighting/distance.py` (161) |
| `block_poles` | function | `(system)` | 70 | `src/pinn_feynman/weighting/distance.py` (112) |
| `block_lines` | function | `(system)` | 84 | `src/pinn_feynman/weighting/distance.py` (122) |
| `distance_function` | function | `(system)` | 106 | `src/pinn_feynman/weighting/distance.py` (160) |
| `DistanceWeighting` | class |  | 143 | `src/pinn_feynman/weighting/__init__.py` (3)<br>`src/pinn_feynman/weighting/factory.py` (5, 8) |
| `DistanceWeighting.build` | method | `(cls, params, n_base=None, domain=None, system=None, input_scale=1.0)` | 152 | `src/pinn_feynman/devtools/codemap.py` (302)<br>`src/pinn_feynman/weighting/factory.py` (32) |
| `DistanceWeighting.weights` | method | `(self, z)` | 166 | `src/pinn_feynman/balancing/fixed.py` (11, 12, 13, 16, 19, 22, 25)<br>`src/pinn_feynman/balancing/relobralo.py` (35, 70, 74, 76, 79, 83, 87, 93, …)<br>`src/pinn_feynman/physics/losses.py` (56, 58)<br>`src/pinn_feynman/training/recorder.py` (103, 105, 106, 107, 191, 202, 271, 275, …)<br>`src/pinn_feynman/training/trainer.py` (157, 160, 162, 163, 168, 274, 275, 276)<br>`src/pinn_feynman/weighting/distance.py` (172) |
| `DistanceWeighting.__call__` | method | `(self, idx, z, residual_norm)` | 171 | — |
| `DistanceWeighting.end_epoch` | method | `(self)` | 174 | `src/pinn_feynman/training/trainer.py` (252, 281) |
| `DistanceWeighting.state_dict` | method | `(self)` | 177 | `src/pinn_feynman/training/recorder.py` (351)<br>`src/pinn_feynman/training/run.py` (78, 102)<br>`src/pinn_feynman/training/trainer.py` (267, 298, 299, 300, 301, 302, 303, 304, …) ⚠️ generic name |
| `DistanceWeighting.load_state_dict` | method | `(self, state)` | 180 | `src/pinn_feynman/evaluation/predict.py` (92)<br>`src/pinn_feynman/training/run.py` (164)<br>`src/pinn_feynman/training/trainer.py` (315, 316, 317, 318, 319, 322, 323, 326) ⚠️ generic name |

### `pinn_feynman.weighting.factory`

[`src/pinn_feynman/weighting/factory.py`](../src/pinn_feynman/weighting/factory.py) — Point weighting from the optional `point_weighting` section of a configuration.

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `POINT_WEIGHTING` | constant |  | 8 | `src/pinn_feynman/weighting/__init__.py` (4)<br>`src/pinn_feynman/weighting/factory.py` (11, 18, 20) |
| `point_weighting_class` | function | `(config)` | 14 | `src/pinn_feynman/training/setup.py` (16, 63, 150)<br>`src/pinn_feynman/weighting/__init__.py` (4)<br>`src/pinn_feynman/weighting/factory.py` (29) |
| `build_point_weighting` | function | `(config, n_base, domain, system=None, input_scale=1.0)` | 23 | `src/pinn_feynman/training/setup.py` (16, 182)<br>`src/pinn_feynman/weighting/__init__.py` (4) |

### `pinn_feynman.weighting.point_weights`

[`src/pinn_feynman/weighting/point_weights.py`](../src/pinn_feynman/weighting/point_weights.py) — Per-point weights of the physics loss: the interface, and `NoWeighting` (Boni's plain mean).

| Symbol | Kind | Signature | Line | Used in |
|---|---|---|---|---|
| `NoWeighting` | class |  | 12 | `src/pinn_feynman/training/setup.py` (16, 153)<br>`src/pinn_feynman/training/trainer.py` (37, 65)<br>`src/pinn_feynman/weighting/__init__.py` (5)<br>`src/pinn_feynman/weighting/factory.py` (6, 8) |
| `NoWeighting.build` | method | `(cls, params, n_base=None, domain=None, system=None, input_scale=1.0)` | 18 | `src/pinn_feynman/devtools/codemap.py` (302)<br>`src/pinn_feynman/weighting/factory.py` (32) |
| `NoWeighting.__call__` | method | `(self, idx, z, residual_norm)` | 21 | — |
| `NoWeighting.end_epoch` | method | `(self)` | 24 | `src/pinn_feynman/training/trainer.py` (252, 281) |
| `NoWeighting.state_dict` | method | `(self)` | 27 | `src/pinn_feynman/training/recorder.py` (351)<br>`src/pinn_feynman/training/run.py` (78, 102)<br>`src/pinn_feynman/training/trainer.py` (267, 298, 299, 300, 301, 302, 303, 304, …) ⚠️ generic name |
| `NoWeighting.load_state_dict` | method | `(self, state)` | 30 | `src/pinn_feynman/evaluation/predict.py` (92)<br>`src/pinn_feynman/training/run.py` (164)<br>`src/pinn_feynman/training/trainer.py` (315, 316, 317, 318, 319, 322, 323, 326) ⚠️ generic name |

## 3. Imports between package modules

| Module | Imports |
|---|---|
| `pinn_feynman.balancing` | `pinn_feynman.balancing.factory`, `pinn_feynman.balancing.fixed`, `pinn_feynman.balancing.relobralo` |
| `pinn_feynman.balancing.factory` | `pinn_feynman.balancing.fixed`, `pinn_feynman.balancing.relobralo` |
| `pinn_feynman.collocation` | `pinn_feynman.collocation.factory`, `pinn_feynman.collocation.r3`, `pinn_feynman.collocation.sampler` |
| `pinn_feynman.collocation.factory` | `pinn_feynman.collocation.extra`, `pinn_feynman.collocation.r3`, `pinn_feynman.collocation.sampler` |
| `pinn_feynman.collocation.r3` | `pinn_feynman.data.domains` |
| `pinn_feynman.collocation.sampler` | `pinn_feynman.data.domains` |
| `pinn_feynman.data` | `pinn_feynman.data.cache`, `pinn_feynman.data.dataset` |
| `pinn_feynman.data.build` | `pinn_feynman.config`, `pinn_feynman.data.cache`, `pinn_feynman.data.dataset`, `pinn_feynman.progress` |
| `pinn_feynman.data.cache` | `pinn_feynman.data.dataset`, `pinn_feynman.data.generation`, `pinn_feynman.data.interpolation`, `pinn_feynman.data.sampling`, `pinn_feynman.paths` |
| `pinn_feynman.data.check_interpolation` | `pinn_feynman.config`, `pinn_feynman.data.build`, `pinn_feynman.data.cache` |
| `pinn_feynman.data.dataset` | `pinn_feynman.data.domains` |
| `pinn_feynman.data.generation` | `pinn_feynman.problems`, `pinn_feynman.progress` |
| `pinn_feynman.data.sampling` | `pinn_feynman.data.domains`, `pinn_feynman.data.interpolation` |
| `pinn_feynman.devtools.benchmark` | `pinn_feynman.config`, `pinn_feynman.training.setup` |
| `pinn_feynman.devtools.codemap` | `pinn_feynman.paths` |
| `pinn_feynman.devtools.list_configs` | `pinn_feynman.paths` |
| `pinn_feynman.evaluation` | `pinn_feynman.evaluation.metrics`, `pinn_feynman.evaluation.predict` |
| `pinn_feynman.evaluation.boni_reference` | `pinn_feynman.problems` |
| `pinn_feynman.evaluation.evaluate` | `pinn_feynman.data.build`, `pinn_feynman.data.cache`, `pinn_feynman.evaluation.boni_reference`, `pinn_feynman.evaluation.metrics`, `pinn_feynman.evaluation.plots`, `pinn_feynman.evaluation.predict`, `pinn_feynman.physics`, `pinn_feynman.visualization.singularities` |
| `pinn_feynman.evaluation.plots` | `pinn_feynman.visualization.accuracy`, `pinn_feynman.visualization.style`, `pinn_feynman.visualization.training` |
| `pinn_feynman.evaluation.predict` | `pinn_feynman.config`, `pinn_feynman.data.generation`, `pinn_feynman.models`, `pinn_feynman.physics`, `pinn_feynman.problems` |
| `pinn_feynman.models` | `pinn_feynman.models.factory`, `pinn_feynman.models.mlp`, `pinn_feynman.models.order_mask` |
| `pinn_feynman.models.factory` | `pinn_feynman.models.mlp`, `pinn_feynman.models.order_mask` |
| `pinn_feynman.paths` | `pinn_feynman.runtime` |
| `pinn_feynman.physics` | `pinn_feynman.physics.coefficients`, `pinn_feynman.physics.derivatives`, `pinn_feynman.physics.losses`, `pinn_feynman.physics.propagation`, `pinn_feynman.physics.system` |
| `pinn_feynman.physics.bottom_up` | `pinn_feynman.physics.coefficients` |
| `pinn_feynman.physics.coefficients` | `pinn_feynman.physics.losses` |
| `pinn_feynman.physics.losses` | `pinn_feynman.physics.derivatives` |
| `pinn_feynman.physics.system` | `pinn_feynman.data.generation`, `pinn_feynman.physics.bottom_up`, `pinn_feynman.physics.coefficients`, `pinn_feynman.problems`, `pinn_feynman.systems.blocks` |
| `pinn_feynman.problems` | `pinn_feynman.problems._registry` |
| `pinn_feynman.problems.box_massless` | `pinn_feynman.problems._registry` |
| `pinn_feynman.problems.box_massless_bu` | `pinn_feynman.problems._registry` |
| `pinn_feynman.problems.box_one_mass` | `pinn_feynman.problems._registry` |
| `pinn_feynman.problems.box_one_mass_bu` | `pinn_feynman.problems._registry` |
| `pinn_feynman.problems.nonplanar_three_point` | `pinn_feynman.problems._registry` |
| `pinn_feynman.problems.nonplanar_three_point_bu` | `pinn_feynman.problems._registry` |
| `pinn_feynman.problems.sunrise_equal_mass_positive` | `pinn_feynman.problems._registry` |
| `pinn_feynman.problems.sunrise_equal_mass_positive_bu` | `pinn_feynman.problems._registry` |
| `pinn_feynman.providers` | `pinn_feynman.providers.base` |
| `pinn_feynman.providers.box_massless` | `pinn_feynman.providers.base`, `pinn_feynman.reference.box_one_mass` |
| `pinn_feynman.providers.box_one_mass` | `pinn_feynman.providers.base`, `pinn_feynman.reference.box_one_mass` |
| `pinn_feynman.providers.nonplanar_three_point` | `pinn_feynman.providers.base`, `pinn_feynman.reference.nonplanar_three_point` |
| `pinn_feynman.providers.sunrise_equal_mass` | `pinn_feynman.providers.base`, `pinn_feynman.reference.sunrise_equal_mass` |
| `pinn_feynman.reference.nonplanar_three_point` | `pinn_feynman.systems` |
| `pinn_feynman.reference.sunrise_equal_mass` | `pinn_feynman.systems` |
| `pinn_feynman.reference.sunrise_equal_mass_positive` | `pinn_feynman.reference.sunrise_equal_mass` |
| `pinn_feynman.runtime` | `pinn_feynman.paths` |
| `pinn_feynman.systems.box_massless_canonical` | `pinn_feynman.systems.box_massless` |
| `pinn_feynman.training` | `pinn_feynman.training.history`, `pinn_feynman.training.optim`, `pinn_feynman.training.setup`, `pinn_feynman.training.trainer` |
| `pinn_feynman.training.queue` | `pinn_feynman.config`, `pinn_feynman.paths`, `pinn_feynman.progress`, `pinn_feynman.training.run`, `pinn_feynman.tuning.publish`, `pinn_feynman.tuning.tune` |
| `pinn_feynman.training.recorder` | `pinn_feynman.physics.derivatives`, `pinn_feynman.physics.losses` |
| `pinn_feynman.training.run` | `pinn_feynman.config`, `pinn_feynman.data.dataset`, `pinn_feynman.paths`, `pinn_feynman.progress`, `pinn_feynman.training.environment`, `pinn_feynman.training.setup` |
| `pinn_feynman.training.setup` | `pinn_feynman.balancing`, `pinn_feynman.collocation`, `pinn_feynman.config`, `pinn_feynman.data.build`, `pinn_feynman.data.cache`, `pinn_feynman.data.domains`, `pinn_feynman.evaluation.predict`, `pinn_feynman.models`, `pinn_feynman.models.factory`, `pinn_feynman.physics`, `pinn_feynman.problems`, `pinn_feynman.training.optim`, `pinn_feynman.training.recorder`, `pinn_feynman.training.trainer`, `pinn_feynman.weighting` |
| `pinn_feynman.training.trainer` | `pinn_feynman.physics.derivatives`, `pinn_feynman.physics.losses`, `pinn_feynman.physics.propagation`, `pinn_feynman.progress`, `pinn_feynman.training.history`, `pinn_feynman.weighting` |
| `pinn_feynman.tuning` | `pinn_feynman.tuning.objective`, `pinn_feynman.tuning.space` |
| `pinn_feynman.tuning.figures` | `pinn_feynman.visualization.tuning` |
| `pinn_feynman.tuning.objective` | `pinn_feynman.training.setup`, `pinn_feynman.tuning.space` |
| `pinn_feynman.tuning.publish` | `pinn_feynman.paths`, `pinn_feynman.progress` |
| `pinn_feynman.tuning.space` | `pinn_feynman.config` |
| `pinn_feynman.tuning.tune` | `pinn_feynman.config`, `pinn_feynman.paths`, `pinn_feynman.progress`, `pinn_feynman.training.queue`, `pinn_feynman.training.setup`, `pinn_feynman.tuning.figures`, `pinn_feynman.tuning.objective`, `pinn_feynman.tuning.space` |
| `pinn_feynman.visualization.accuracy` | `pinn_feynman.visualization.singularities`, `pinn_feynman.visualization.style` |
| `pinn_feynman.visualization.components` | `pinn_feynman.visualization.methods`, `pinn_feynman.visualization.singularities`, `pinn_feynman.visualization.style` |
| `pinn_feynman.visualization.methods` | `pinn_feynman.visualization.extra`, `pinn_feynman.visualization.style` |
| `pinn_feynman.visualization.singularities` | `pinn_feynman.systems.singular_kinds` |
| `pinn_feynman.visualization.training` | `pinn_feynman.visualization.methods`, `pinn_feynman.visualization.style` |
| `pinn_feynman.visualization.triangle` | `pinn_feynman.visualization.components`, `pinn_feynman.visualization.methods`, `pinn_feynman.visualization.style` |
| `pinn_feynman.visualization.tuning` | `pinn_feynman.visualization.style` |
| `pinn_feynman.weighting` | `pinn_feynman.weighting.distance`, `pinn_feynman.weighting.factory`, `pinn_feynman.weighting.point_weights` |
| `pinn_feynman.weighting.distance` | `pinn_feynman.problems` |
| `pinn_feynman.weighting.factory` | `pinn_feynman.weighting.distance`, `pinn_feynman.weighting.extra`, `pinn_feynman.weighting.point_weights` |

## 4. Entry point (`python -m ...`)

- `python -m pinn_feynman.data.build`
- `python -m pinn_feynman.data.check_interpolation`
- `python -m pinn_feynman.devtools.benchmark`
- `python -m pinn_feynman.devtools.codemap`
- `python -m pinn_feynman.devtools.list_configs`
- `python -m pinn_feynman.evaluation.evaluate`
- `python -m pinn_feynman.systems.analysis`
- `python -m pinn_feynman.training.queue`
- `python -m pinn_feynman.training.run`
- `python -m pinn_feynman.tuning.tune`
