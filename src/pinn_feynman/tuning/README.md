# pinn_feynman.tuning

Hyperparameter tuning with **Optuna**. The search space is **per problem**, in `configs/<problem>/tune.yaml`, and starts from a training configuration of the problem.

## The tuning file

```yaml
base_config: baseline.yaml         # training configuration to start from
study:   {name, part, n_trials, sampler: tpe|random, seed, pruner: {name: median|none, ...}}   # n_trials = TOTAL
budget:  {...}                     # overrides the base for the short trials (epochs, points per epoch)
full_training: {epochs: 400}       # epochs written to best_config.yaml
search_space:                      # dotted paths in the configuration
  training.optimizer.lr: {type: float, low: 1.0e-3, high: 3.0e-2, log: true}
  model.width: {type: categorical, choices: [16, 32, 64]}
  model.depth: {type: int, low: 2, high: 4}
```
`model.width` and `model.depth` are special: they become `model.hidden = [width] * depth`.

## Modules

| Module | Name | What it does |
|---|---|---|
| `space.py` | `suggest(trial, space)` | values drawn from Optuna: `float` (also log), `int`, `categorical` |
| | `set_path`, `apply(config, values)` | writes the values into the dotted paths; width/depth → hidden |
| | `trial_config(base, budget, values)` | configuration of a trial: base + budget + values |
| `objective.py` | `make_objective(base, budget, space, part, datasets)` | one trial = short training run; value = **best $L_{val}$**; a divergent training run is marked "pruned" |
| | `PruningCallback(trial)` | reports $L_{val}$ to Optuna at the end of each epoch; interrupts the trial if the pruner decides to |
| `figures.py` | `tuning_history(study, path, reference)` | re-export of `visualization.tuning`: value of every trial (completed and pruned) and the best so far, on a log scale, with optional reference lines |
| `tune.py` | `python -m pinn_feynman.tuning.tune <tune.yaml> [--n-trials N] [--out DIR] [--max-minutes M]` | runs or resumes the study |
| | `run_study(tuning_path, out, n_trials, max_minutes, session)` → `(outcome, directory)` | `done`, `paused` (time budget spent); `session` is the queue's (`training.queue.session_class()`) |
| | `study_directory`, `study_status` (`done`/`partial`/`new`), `n_finished`, `fail_stale_trials`, `copy_atomic` | pieces used by the queue |
| | `trial_line(study, trial, target)`, `print_summary(summary)` | output: during a trial a bar on its epochs (removed at the end); then one line per trial (number, finished/target, state, value, best so far, parameters); a short summary at the end |
| | `load_tuning_config`, `create_study`, `write_outputs` | reusable parts |
| `publish.py` | `publish_study(study_dir, tuning_file, problem, root)` | writes `configs/<problem>/tuned/<study>.yaml` (best configuration with a header saying where it comes from); never overwrites a different file |
| | `check_path`, `tuned_config_path`, `tuned_config_text`, `write_local`, `PublishError` | only `configs/<problem>/tuned/<name>.yaml` is accepted |

## Outputs (`runs/tuning/<study>/`)
| File | Content |
|---|---|
| `study.db` | the Optuna study (SQLite). Optuna works on a local copy (a network drive is not safe for SQLite), copied back after every trial |
| `trials.csv` | all trials, with parameters, value and status |
| `best_config.yaml` | **complete training configuration** with the best values and the full epochs: launched with `training.run` |
| `summary.json` | number of trials (completed, pruned, finished), target, `done`, best value and parameters |
| `optimization_history.png` | history of the trials on a log scale (`figures.tuning_history`) |
| `param_importances.png`, `slice.png` | Optuna figures. Importance is computed with PED-ANOVA, without scikit-learn |

## Resuming and publishing
- `n_trials` is the **total** of finished trials (complete + pruned). Relaunching continues up to it; a finished study is skipped. `--n-trials` sets a different total.
- `--max-minutes`: the study stops after the running trial once the budget is spent (`paused`).
- Trials left `RUNNING` by a session that died are marked `FAIL` at start and do not count.
- In a queue (`training/queue.py`, entry `{tune: <problem>/<name>}`), a finished study is published as `configs/<problem>/tuned/<study>.yaml`, so that the next entries of the queue can train it (`{config: <problem>/tuned/<study>, ...}`).

**Note:** the $L_{val}$ of a short trial is not the final error. Tuning **ranks** the configurations; the best one is then trained fully and evaluated on the test set.

