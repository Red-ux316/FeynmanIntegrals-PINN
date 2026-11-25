# configs/

YAML configuration files, **one folder per problem** (Feynman diagram), one file per training configuration, and the
training queues that use them (`<problem>/queues/`, and `queues/` for the queues that span several problems). They are
passed as arguments to the package's entry points; no path is hardcoded in the code (`pinn_feynman.paths`).

| Folder | Problem |
|---|---|
| [box_massless/](box_massless/README.md) | massless box |
| [box_one_mass/](box_one_mass/README.md) | one-mass box |
| [sunrise_equal_mass/](sunrise_equal_mass/README.md) | equal-mass sunrise |
| [nonplanar_three_point/](nonplanar_three_point/README.md) | non-planar three-point function |
| [queues/](queues/README.md) | queues over several problems |

**Inheritance.** A configuration can start with `base: <file>.yaml` and write **only what changes**
(`pinn_feynman.config.load_config`). Sections are merged recursively; a section whose `name` differs from the base one
(another component, e.g. `collocation: {name: r3_continuous, ...}`) replaces the base section entirely.

**Names.** The name of a configuration is the name of its base plus the traits that change
(`bu_phys_3_27` + network 128x4 + exponential scheduler on 1000 epochs = `bu_phys_3_27_w128d4_exp_1000ep`); the name of
a queue says what it holds.

**Running.** One training: `python -m pinn_feynman.training.run configs/<problem>/<name>.yaml --part real`. A queue:
`python -m pinn_feynman.training.queue configs/<problem>/queues/<queue>.yaml` (or `notebooks/run_local.ipynb`). The
datasets go to `<storage>/data/` and the runs to `<storage>/runs/`, with `<storage>` = `$PINN_FEYNMAN_STORAGE` or the
folder `storage/` at the root of the repository.

## Sections of a training configuration

| Section | Content |
|---|---|
| `data` | problem, domain, `boundary`, `validation`, `test` datasets (sampler, number of points, seed) |
| `dtype` | tensor precision |
| `model` | network: name in the registry, hidden layers, activation, `in_features`, optional `active_orders` |
| `physics` | system of equations and `input_scale` (network variable $z=c\,x$) |
| `balancing` | loss weights (ReLoBRaLo or fixed) |
| `collocation` | collocation points: `base_points` (default), `resample` (new points every epoch), `dirichlet`, `r3_continuous` |
| `point_weighting` | optional: `none` (default), `distance` (distance weighting near the singular points) |
| `training` | epochs, seed, threads, `device`, `checkpoint_every`, `log_every`, `residual_stats_points`, `diagnostics` (training recorder), optimizer, scheduler |
