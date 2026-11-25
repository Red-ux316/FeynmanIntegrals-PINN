"""Run (or resume) a tuning study, idempotently.

    python -m pinn_feynman.tuning.tune configs/box_massless/bu_tune.yaml [--n-trials N] [--out DIR] [--max-minutes M]

The study lives in <out>/<study name>/ (default runs/tuning/): study.db,
trials.csv, best_config.yaml (a full training configuration), summary.json and figures.

- `n_trials` (tune.yaml or --n-trials) is the TOTAL of finished trials (complete + pruned): running
  again after an interruption continues up to it, and a finished study is left as it is.
- --max-minutes stops cleanly after the running trial once the time budget is spent ('paused').
- Trials left RUNNING by a session that died are marked FAIL at start (they do not count).
- SQLite works on a local working copy of study.db (a network drive is not safe for SQLite); the
  copy is written back to the study folder after every trial.
- In a queue the study is taken through the queue's session (training.queue.session_class): one session
  at a time per study.
"""

import argparse
import json
import os
import shutil
import tempfile
import time
import warnings
from pathlib import Path

import optuna
import yaml

from .. import progress as console
from ..config import load_config
from ..paths import runs_dir
from ..training.setup import load_datasets
from .objective import make_objective
from .figures import tuning_history
from .space import apply

FINISHED = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)

SAMPLERS = {"tpe": optuna.samplers.TPESampler, "random": optuna.samplers.RandomSampler}
PRUNERS = {"median": optuna.pruners.MedianPruner, "none": optuna.pruners.NopPruner}


def load_tuning_config(path):
    """Tuning file + its base training configuration (`base_config`, relative path)."""
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        tuning = yaml.safe_load(stream)
    base = load_config(path.parent / tuning["base_config"])
    return tuning, base


def create_study(study_cfg, storage):
    sampler = SAMPLERS[study_cfg.get("sampler", "tpe")](seed=study_cfg.get("seed"))
    pruner_cfg = dict(study_cfg.get("pruner", {"name": "median"}))
    pruner = PRUNERS[pruner_cfg.pop("name")](**pruner_cfg)
    return optuna.create_study(study_name=study_cfg["name"], storage=storage, load_if_exists=True,
                               direction="minimize", sampler=sampler, pruner=pruner)


def n_finished(study):
    """Complete + pruned trials (failed and running ones do not count towards n_trials)."""
    return sum(t.state in FINISHED for t in study.trials)


def write_outputs(study, directory, base, budget, full_training, target=None):
    """trials.csv, best_config.yaml, summary.json and the figures.

    summary.json has `done` = True once `target` finished trials are reached (with at least one complete).
    """
    study.trials_dataframe().to_csv(directory / "trials.csv", index=False)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    finished = n_finished(study)
    summary = {"study": study.study_name, "n_trials": len(study.trials), "n_complete": len(complete),
               "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
               "n_finished": finished, "target": target,
               "done": bool(complete) and target is not None and finished >= target}
    if complete:
        best = study.best_trial
        best_config = apply(base, best.params)           # full-size training with the best values
        best_config["training"]["epochs"] = full_training.get("epochs", best_config["training"]["epochs"])
        with open(directory / "best_config.yaml", "w", encoding="utf-8") as stream:
            yaml.safe_dump(best_config, stream, sort_keys=False)
        summary.update({"best_value": best.value, "best_params": best.params, "best_trial": best.number,
                        "best_hidden": best.user_attrs.get("hidden"), "budget": budget})
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if complete:
        tuning_history(study, directory / "optimization_history.png")
    if len(complete) >= 2:
        _figures(study, directory)
    return summary


def _figures(study, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from optuna.visualization import matplotlib as ovm

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")        # optuna marks its matplotlib backend as experimental
        # PED-ANOVA importances: no scikit-learn needed (optuna's default fANOVA needs it)
        importances = lambda s: ovm.plot_param_importances(
            s, evaluator=optuna.importance.PedAnovaImportanceEvaluator())
        for name, fn in (("param_importances", importances),
                         ("slice", ovm.plot_slice)):
            try:
                ax = fn(study)
                fig = (ax.flatten()[0] if hasattr(ax, "flatten") else ax).get_figure()
                fig.set_size_inches(10, 5)
                fig.tight_layout()
                fig.savefig(directory / f"{name}.png", dpi=130)
                plt.close(fig)
            except (ValueError, RuntimeError, ImportError) as err:   # e.g. importances need variance
                console.note(f"  figure {name} skipped: {err}")


def _short(value):
    return f"{value:.3g}" if isinstance(value, float) else str(value)


def trial_line(study, trial, target):
    """`  trial   7  [8/30]  COMPLETE  val 1.83e-04  best 1.21e-04  lr=0.0175 width=64 ...`"""
    value = f"val {trial.value:.2e}" if trial.value is not None else "val -"
    try:
        best = f"best {study.best_value:.2e}"
    except ValueError:                                   # no complete trial yet
        best = "best -"
    params = " ".join(f"{k}={_short(v)}" for k, v in trial.params.items())
    return (f"  trial {trial.number:3d}  [{n_finished(study)}/{target}]  {trial.state.name:8s}  {value}  {best}  "
            f"{params}")


def print_summary(summary):
    """A few lines instead of summary.json (which is on disk)."""
    state = "done" if summary["done"] else "not finished"
    console.note(f"  study {summary['study']} {state}: {summary['n_finished']}/{summary['target']} finished "
                 f"({summary['n_complete']} complete, {summary['n_pruned']} pruned)")
    if "best_value" in summary:
        params = " ".join(f"{k}={_short(v)}" for k, v in summary["best_params"].items())
        console.note(f"  best: trial {summary['best_trial']}, val {summary['best_value']:.3e}, {params}")


def study_directory(tuning_path, out=None):
    """<out or runs_dir()/tuning>/<study name>: where the study and its outputs live."""
    with Path(tuning_path).open(encoding="utf-8") as stream:
        name = yaml.safe_load(stream)["study"]["name"]
    return (Path(out) if out else runs_dir() / "tuning") / name


def study_status(directory):
    """'done' (summary.json says so), 'partial' (study.db exists) or 'new'."""
    directory = Path(directory)
    summary = directory / "summary.json"
    if summary.exists() and json.loads(summary.read_text(encoding="utf-8")).get("done"):
        return "done"
    return "partial" if (directory / "study.db").exists() else "new"


def fail_stale_trials(study):
    """Mark as FAIL the trials left RUNNING by a session that died (one session at a time works on a study)."""
    stale = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
    for trial in stale:
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    return len(stale)


def copy_atomic(source, target):
    """Copy to a temporary name next to `target`, then rename: readers never see a partial file."""
    tmp = Path(target).with_name(f"{Path(target).name}.{os.getpid()}.tmp")
    shutil.copyfile(source, tmp)
    os.replace(tmp, target)


def run_study(tuning_path, out=None, n_trials=None, max_minutes=None, session=None, **options):
    """Run or resume a study up to its total of finished trials. Returns (outcome, directory).

    outcome: 'done' (target reached, outputs written), 'paused' (time budget spent, or the session lost the
    study), 'busy' (another session holds the study). `session`: the queue's session (default
    `training.queue.session_class()(**options)`).
    """
    from ..training.queue import session_class
    tuning, base = load_tuning_config(tuning_path)
    study_cfg = tuning["study"]
    directory = study_directory(tuning_path, out)
    target = n_trials if n_trials is not None else study_cfg.get("n_trials", 20)
    if study_status(directory) == "done" and n_trials is None:
        console.note(f"  study {study_cfg['name']}: already complete")
        return "done", directory
    session = session or session_class()(**options)
    if not session.claim(directory):
        console.note(f"  study {study_cfg['name']}: held by another session (busy)")
        return "busy", directory
    directory.mkdir(parents=True, exist_ok=True)       # only the session that holds the study creates the folder
    workdir = Path(tempfile.mkdtemp(prefix="pinn_tuning_"))
    local_db = workdir / "study.db"
    try:
        if (directory / "study.db").exists():
            shutil.copyfile(directory / "study.db", local_db)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = create_study(study_cfg, f"sqlite:///{local_db.as_posix()}")
        if fail_stale_trials(study):
            copy_atomic(local_db, directory / "study.db")
        deadline = time.perf_counter() + 60 * max_minutes if max_minutes is not None else None
        state = {"lost": False}

        def after_trial(study, trial):
            console.note(trial_line(study, trial, target))
            copy_atomic(local_db, directory / "study.db")
            try:
                session.heartbeat(directory)
            except session.Lost as lost:
                console.note(f"  study {study_cfg['name']}: taken by another session ({lost})")
                state["lost"] = True
                study.stop()
                return
            if deadline is not None and time.perf_counter() > deadline:
                console.note(f"  study {study_cfg['name']}: time budget of {max_minutes:g} min spent, pausing")
                study.stop()

        remaining = target - n_finished(study)
        console.note(f"  study {study_cfg['name']}: {n_finished(study)}/{target} trials finished, {max(remaining, 0)} to run")
        if remaining > 0:
            objective = make_objective(base, tuning.get("budget"), tuning["search_space"],
                                       study_cfg.get("part", "real"), load_datasets(base))
            study.optimize(objective, n_trials=remaining, callbacks=[after_trial])
        if state["lost"]:
            return "paused", directory
        copy_atomic(local_db, directory / "study.db")
        summary = write_outputs(study, directory, base, tuning.get("budget"), tuning.get("full_training", {}),
                                target=target)
        print_summary(summary)
        return ("done" if summary["done"] else "paused"), directory
    finally:
        session.release(directory)
        shutil.rmtree(workdir, ignore_errors=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tuning_config")
    parser.add_argument("--n-trials", type=int, help="total of finished trials (default: study.n_trials)")
    parser.add_argument("--out", help="parent folder (default runs/tuning)")
    parser.add_argument("--max-minutes", type=float, help="pause cleanly after the running trial")
    args = parser.parse_args(argv)
    console.header(f"tune: {args.tuning_config}")
    _, directory = run_study(args.tuning_config, args.out, args.n_trials, args.max_minutes)
    return directory


if __name__ == "__main__":
    main()
