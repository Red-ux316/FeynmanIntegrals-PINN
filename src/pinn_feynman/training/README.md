# pinn_feynman.training

Training loop of Boni's PINN (thesis §4.4, §6.1.2, Tab. 6.1), experiment construction from configuration, entry point.

## A step and an epoch
- **Step** (`Trainer.train_step`): $L_{data}$ on the 2 boundary points (6.11); $L_{DE}$ on a batch of 64 collocation points (6.12); total loss weighted by ReLoBRaLo (6.13); backward; Adam step.
- **Epoch** (`train_epoch`): 1024 steps, i.e. the 65,536 points drawn by the sampler.
- **At the end of an epoch:**
  - $L_{val}$ = MSE on the validation points (5.16);
  - `val_physics_loss` = unweighted physics loss $\mathrm{mean}(r^2)$ (6.12) on the same validation points (configuration `training.validation_physics`, default `true`). The matrices at the validation points are computed once; no gradient, no random numbers, so the training is the same with or without it (test);
  - the scheduler receives the **average weighted loss** of the epoch;
  - if $L_{val}$ is the lowest so far, the network state is copied (Boni p. 81).
- **Matrix:** $A^{(0)}$, $A^{(1)}$ are computed **once** on all base points, like Boni's collocation set (6.7); at each step the extracted rows are taken. Samplers without fixed points (`resample`, `r3_continuous`) change their points and their `version`: the trainer recomputes the matrix after the draw of an epoch, **right after every R3 refresh** and **after loading a checkpoint** (the coordinates change, the version number may not tell).

## Modules

| Module | Name | What it does |
|---|---|---|
| `optim.py` | `OPTIMIZERS`, `SCHEDULERS`, `build_optimizer`, `build_scheduler` | Adam; ReduceLROnPlateau (`eps: 0` lifts the 1e-8 default), cosine, exponential, `exponential_until` (`ExponentialUntil`: exponential until `stop_epoch`, then the lr of that epoch; identical to exponential before it), from configuration. The trainer passes the loss only to the plateau scheduler |
| `history.py` | `History` | rows per epoch; `column(key)`, `save_csv(path)` with the union of the columns (a run resumed from an older checkpoint may gain new columns) |
| `trainer.py` | `PointSet(z, y)` | inputs in the network's variable and targets |
| | `Trainer` | `losses(idx)`, `train_step`, `train_epoch`, `validate`, `validate_physics`, `fit(epochs, on_epoch_end, desc, leave)`; attributes `history`, `best_state`, `best_val`, `best_epoch`. `fit` shows a progress bar on the epochs (last train and val loss, best val, lr); every `log_every` epochs a full line stays above it. With `residual_stats_points=N` (configuration `training.residual_stats_points`, default 1024, 0 = off) every epoch adds `residual_mean`, `residual_max`, `residual_skewness`, `residual_kurtosis` to the history, from the residual on a fixed uniform grid of N points (`residual_stats()`): no gradient and no random numbers, so the training is unchanged; about 10 ms per epoch |
| `environment.py` | `runtime_environment`, `cpu_model` | the session that made a run (Python, torch, numpy, CPU model, threads), written in the `environment` key of `summary.json`: the same configuration and seed can give different results on different machines |
| `recorder.py` | `TrainingRecorder`, `build_recorder`, `effective_points`, `log_quadrature_grid` | optional record of what happens during the training; see below |
| `setup.py` | `build_experiment(config, part, datasets, seed)` | all the pieces: dataset converted to $z=10x$, network, system, sampler, balancer, optimizer, scheduler. Before anything else `check_variables` verifies that the dimension of the domain, `model.in_features` (default 1; the one-mass configuration writes `in_features: 2`) and `system.n_variables` agree (`ValueError`; the configuration is never modified), and with more than one variable `check_multi_variable` refuses, with a `NotImplementedError` naming it, every component written for one variable only (the recorder among them). `check_part` refuses a part the problem does not train (`Problem.parts`). The scaled domain is `as_domain(domain).scaled(scale)` (interval or triangle) |
| | `SEED_OFFSETS` | a single configuration seed, separate random streams for weights, collocation and $\rho$ |
| `run.py` | `python -m pinn_feynman.training.run <config> --part real\|imag [--epochs] [--seed] [--out]`, or `--resume <run>` | one training run; writes `runs/<problem>-<config>_<part>_<date-time>/` |
| `queue.py` | `python -m pinn_feynman.training.queue <queue.yaml> [--max-minutes M] [--only ...] [--status]` | a list of trainings and tuning studies, run in order; see below |

## One code for one and two variables
The trainer, the samplers and `setup` are the same for the massless box ($d=1$) and the one-mass box ($d=2$); a comparison with frozen numbers checked that the massless box gives the same numbers as before on the same machine. The only branch in the trainer is the grid of the residual statistics (`residual_stats_points`): for $d=1$ a linear grid on the interval (or between the base-point extremes), for $d>1$ the fixed barycentric lattice `Triangle.grid(n)` (no random numbers: the generator of the sampler is not touched).

## Hooks for the training methods
The trainer has two optional hooks; the default is Boni's baseline.
- `point_weighting`: per-point weights in the physics loss. Called at each step with the residual norms, returns $w$ or `None` (`weighting/`: distance weighting). A sampler with `loss_weights(idx)` multiplies its own weights into $w$; without it `losses` is unchanged. A weighting with `history_columns()` adds its own columns to `history.csv`, and one with `snapshot_arrays()` adds its state to the recorder's snapshots.
- `sampler.refresh(residual_fn)`: if the sampler has it (R3 on the continuum), it is called at the end of the epoch with `Trainer.residual_norms`, i.e. the gradient-free residuals over many points. A sampler with `blocks_per_epoch` > 1 (`r3_continuous` with a small `population`) is also refreshed between the blocks of the epoch (`train_epoch`); the history columns of the epoch are means over all its steps, and `r3_retained` is the number of points kept by the last refresh.

A new method is added by writing its component and registering it, **without touching the trainer**.

## Training recorder
Off unless the configuration has a `training.diagnostics` block (`{}` = on with the defaults):
```yaml
training:
  diagnostics:
    snapshot_epochs: [1, 10, 50, 100, 200, 400]   # the final epoch is always added
    log_cells: 24                                  # equal cells in log10|x| over the domain (the domain must not contain x = 0)
    grid_points: 2048                              # fixed grid uniform in log10|x|, trapezoid weights in x
```
- **Per step** (`on_batch`, called by `Trainer.losses` with the residual norms $r_i$ and the weights $w_i$ it already has): $c_i = w_i^2 r_i^2$ ($r_i^2$ without weights), effective points $n_{eff} = (\sum c)^2/\sum c^2$ weighted and unweighted, $\mathrm{mean}(r_i^2)$, per $\log|x|$ cell $\sum r^2$, $\sum c$ and the number of points; with fixed base points a visit counter per base point.
- **Per epoch**, extra `history.csv` columns: `physics_loss_unweighted`, `n_eff_weighted`, `n_eff_unweighted` (means over the steps), and, when the domain reaches the tail $-0.1 < x < 0$: `tail_share_r2` (share of $\int r^2\,dx$ in the tail, on the grid, network at the end of the epoch) and `tail_share_w2r2` (share of the epoch's $\sum c$ in the tail: the loss the optimiser saw). `diagnostics/cells.npz`: the per-cell sums of every epoch (`sum_r2`, `sum_c`, `count`, `grid_int_r2`), cell edges, grid and weights.
- **Snapshots** `diagnostics/epoch_XXXX.npz` (end of the epoch, after the R3 refresh): network parameters (`model.*`), residual on the grid, visit counters, the point weighting's own state if it has any; R3: `population_x` (coordinates in $x$, float64, after the refresh; the first `last_retained` are the retained points), `population_trained_x` (the population trained on in the epoch; with a small population refreshed within the epoch, the last block), `last_retained`. About 1-3 MB each, compressed.
- **Files and resume:** snapshots wait in memory; `run.save_checkpoint` writes them and `cells.npz` (temporary file + rename) **before** `checkpoint.pt`, so an interruption can only make the resume recompute identical files. Cell sums and visit counters are in the checkpoint (`recorder` key).
- **Gradients per loss term**, `gradients: {every: 64}` inside `diagnostics` (off by default): every 64 steps, before the optimiser step, the gradient of each weighted term ($\lambda_k c_k L_k$, data and PDE) with `torch.autograd.grad` on the step's graph (kept with `retain_graph`; `.grad` untouched). Saved: norms, cosine between the two, and the norms after Adam's division $g/(\sqrt{\hat v}+\epsilon)$ with the optimiser's second moment before the step (NaN before the first step): about the step each term alone would cause. `diagnostics/gradients.npz` (one row per sampled step) and per-epoch medians `grad_norm_data`, `grad_norm_pde`, `grad_cosine`, `adam_step_data`, `adam_step_pde` in `history.csv`. The training is identical with it on or off and resumes exactly; cost about 3%.
- **Gradients on the whole population**, `population_gradient: {stride: 8}` inside `diagnostics` (off by default): at the end of every epoch, with the network of that moment, the gradient of the plain L_PDE on the points trained on in the epoch (P_t, every 8th) and on the population after the R3 refresh (P_{t+1}), and of L_data. Columns `pop_grad_norm_pde`, `pop_grad_norm_data`, `pop_grad_cos_data_pde`, `pop_grad_cos_refresh` (NaN without a population); vectors in `diagnostics/population_gradients.npz` (float32, one row per epoch). Snapshots also keep Adam's moments (`adam.exp_avg`, `adam.exp_avg_sq`, `adam.step`) and the ReLoBRaLo state (`balancer.weights`, `previous`, `initial`).
- **Training unchanged:** the recorder never calls the point weighting (it would advance a weighting with a state), draws no random numbers and uses detached tensors without gradients. Checked: same parameters and history with the recorder on and off; a paused and resumed run writes the same files as a straight one.

## Deterministic runs, pause and queue
- `run.py --run-dir DIR`: fixed folder. If the run is complete it is skipped, if there is a checkpoint it resumes, otherwise it starts there. `run_status(dir)` → `done` / `partial` / `new`.
- `run.py --max-minutes M`: at the first checkpoint after M minutes the training **stops cleanly** (`TimeBudgetReached`) and resumes by relaunching the same command (sessions with a time limit, memory limits).
- `train(config, part, directory, state, max_minutes)` → `("done", summary)` or `("paused", None)`: the core reused by the queue.
- `queue.py`:
  - `RunSpec(config, part, seed)`, with `config` = `problem/name` or a `.yaml` path; folder `runs/<problem>-<name>_<part>[_seed<k>]`;
  - `load_queue`, `run_one`, `run_queue`, `queue_status`, `print_table(status)`;
  - an optional `diagnostics:` block at the top of the queue file turns the training recorder on for the new runs of the queue (`with_diagnostics`: merged into `training.diagnostics`, the configuration's own values win); the run keeps its name;
  - output: a header per entry (`== [3/8] <run> (training) ==`), then the run's own lines and bar; at the end a table of the outcomes.

  The queue is **idempotent**: it skips completed runs, resumes interrupted ones, stops at the first pause.
  An existing run (complete or interrupted) whose saved `config.yaml` differs from the configuration the queue would use stops the queue with `ConfigMismatch` (the differing keys in the message; `--status` marks it `DIFFERENT CONFIG`), instead of skipping or resuming it. Execution settings (`device`, `threads`, `log_every`, `checkpoint_every`, `diagnostics`) are not compared. `config_differences(spec, directory)`, `check_existing_run(spec, directory)`.
  - `TuneSpec(tune, n_trials)`: entry `{tune: <problem>/<name>}`, a tuning study (`tuning/tune.py`); `run_tune` runs or resumes it and, once finished, publishes its best configuration as `configs/<problem>/tuned/<study>.yaml` (`tuning/publish.py`). Outcome `publish_failed` if the publication fails: the file is written locally anyway.
  - `config` may contain subfolders (`box_massless/tuned/<study>`); the problem is the first element, and the run name is `<problem>-<rest with _>_<part>[_seed<k>]`.
  - A training whose configuration does not exist yet is `waiting` and skipped.
  - `LocalSession`: the queue runs alone on its storage. `session_class()` is the hook for a session that shares the storage with others (lock before training, heartbeat at each checkpoint, release at the end).

## Checkpoint and resume
- The run's folder is created **at the start**, with `config.yaml`.
- Every `training.checkpoint_every` epochs, `checkpoint.pt` and `history.csv` are rewritten. `checkpoint.pt` contains the network, optimizer, scheduler, state of ReLoBRaLo and of the sampler, including the random generators, history and best network. Writing goes through a temporary file that is then renamed, so an interruption does not leave broken files.
- `python -m pinn_feynman.training.run --resume runs/<run>` resumes from the last checkpoint. The result is **bit-for-bit identical** to that of an uninterrupted training run (test).
- `Trainer.state_dict()` / `load_state_dict()`; `Trainer.fit(epochs, on_epoch_end)` continues from `trainer.epoch`.

## Device
`training.device`: `cpu`, `cuda` or `auto` (`setup.resolve_device`). Network, dataset, base points and matrices go on the device; batch indices are extracted on CPU. `cuda` without a visible GPU gives an explicit error.

## Outputs of a run
`config.yaml` (configuration used, with `part`), `checkpoint.pt`, `history.csv` (per epoch: `train_loss`, `data_loss`, `physics_loss`, `val_loss`, `lr`, `seconds`, `val_physics_loss`, `weight_0`, `weight_1`, plus the optional columns above), `best_model.pt`, `final_model.pt`, `summary.json`.

Measured time: about 6.3 s per epoch with 1 thread, i.e. about 42 minutes for 400 epochs.

With `training.diagnostics` also the folder `diagnostics/` (see "Training recorder").

