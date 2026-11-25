"""Train one PINN (real or imaginary part) from a configuration, with checkpoints.

    python -m pinn_feynman.training.run configs/box_massless/baseline.yaml --part real [--epochs N] [--seed S] [--out DIR]
    python -m pinn_feynman.training.run --resume runs/<run>                  # continue an interrupted run
    python -m pinn_feynman.training.run <config> --part real --run-dir runs/<name> [--max-minutes M]

The run directory (default runs/<problem>-<config>_<part>_<timestamp>/) is created at the
start with config.yaml; every training.checkpoint_every epochs checkpoint.pt and history.csv
are rewritten; at the end best_model.pt, final_model.pt and summary.json are added.

--run-dir makes the directory deterministic and the command idempotent: a complete run is
skipped, a run with a checkpoint is resumed, otherwise a new run starts there.
--max-minutes pauses cleanly at the first checkpoint after the time budget (sessions with a time
limit, memory limits); running the same command again continues.
"""

import argparse
import copy
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

from .. import progress as console
from ..config import load_config
from ..data.dataset import PARTS
from ..paths import runs_dir
from .environment import runtime_environment
from .setup import build_experiment

CHECKPOINT = "checkpoint.pt"
SUMMARY = "summary.json"


class TimeBudgetReached(Exception):
    """Raised after a checkpoint when the run has used its time budget (the run is paused)."""


def run_directory(config, config_path, part, out=None):
    base = Path(out) if out else runs_dir()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return base / f"{config['data']['problem']}-{Path(config_path).stem}_{part}_{stamp}"


def run_status(directory):
    """'done' (summary.json), 'partial' (a non-empty checkpoint.pt and its config.yaml) or 'new'. A folder that an
    interrupted session left with an empty checkpoint, or without config.yaml, is 'new': the run starts again there
    instead of stopping the queue."""
    directory = Path(directory)
    if (directory / SUMMARY).exists():
        return "done"
    checkpoint = directory / CHECKPOINT
    if checkpoint.exists() and checkpoint.stat().st_size > 0 and (directory / "config.yaml").exists():
        return "partial"
    return "new"


def atomic_torch_save(obj, path):
    """Write to a temporary file, then rename: an interruption never leaves a broken file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(directory, trainer):
    """Recorder files first (diagnostics/), then checkpoint.pt and history.csv.

    Writing the diagnostics before the checkpoint means an interruption in between only makes the resume
    recompute (identically) what is already on disk, never lose a snapshot.
    """
    recorder = getattr(trainer, "recorder", None)
    if recorder is not None:
        recorder.flush(directory)
    atomic_torch_save(trainer.state_dict(), directory / CHECKPOINT)
    trainer.history.save_csv(directory / "history.csv")


def checkpoint_callback(directory, every, deadline=None, before_checkpoint=None):
    """Checkpoint every `every` epochs; after a checkpoint, stop if past `deadline` (perf_counter).

    `before_checkpoint()` runs before each save and may raise to stop without writing (queue sessions).
    """
    def callback(trainer):
        if every and trainer.epoch % every == 0:
            if before_checkpoint is not None:
                before_checkpoint()
            save_checkpoint(directory, trainer)
            if deadline is not None and time.perf_counter() > deadline:
                raise TimeBudgetReached(f"paused at epoch {trainer.epoch}")
    return callback


def save_run(directory, experiment, elapsed):
    """Final outputs: history, final and best networks, summary."""
    trainer = experiment.trainer
    directory.mkdir(parents=True, exist_ok=True)
    save_checkpoint(directory, trainer)
    torch.save(trainer.model.state_dict(), directory / "final_model.pt")
    torch.save(trainer.best_state, directory / "best_model.pt")
    last = trainer.history.rows[-1]
    summary = {"part": experiment.part, "epochs": len(trainer.history), "best_epoch": trainer.best_epoch,
               "best_val_loss": trainer.best_val, "final_train_loss": last["train_loss"],
               "final_val_loss": last["val_loss"], "final_lr": last["lr"], "seconds": elapsed,
               "device": str(trainer.device), "environment": runtime_environment()}
    with open(directory / SUMMARY, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    return summary


def write_config(directory, config, part):
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "config.yaml", "w", encoding="utf-8") as stream:
        yaml.safe_dump({**config, "part": part}, stream, sort_keys=False)


def load_run_state(directory):
    """(config, part, checkpoint state) of an existing run directory."""
    directory = Path(directory)
    config = load_config(directory / "config.yaml")
    part = config.pop("part")
    return config, part, torch.load(directory / CHECKPOINT, weights_only=False)


def new_config(config_path, epochs=None, seed=None):
    config = copy.deepcopy(load_config(config_path))
    if epochs is not None:
        config["training"]["epochs"] = epochs
    if seed is not None:
        config["training"]["seed"] = seed
    return config


def prepare(args):
    """(config, part, directory, checkpoint state or None) for a new or resumed run."""
    if args.resume:
        directory = Path(args.resume)
        config, part, state = load_run_state(directory)
        return config, part, directory, state
    if not args.config or not args.part:
        raise SystemExit("give a configuration and --part, or --resume <run directory>")
    config = new_config(args.config, args.epochs, args.seed)
    if getattr(args, "run_dir", None):
        directory = Path(args.run_dir)
        if run_status(directory) == "partial":
            config, part, state = load_run_state(directory)
            return config, part, directory, state
    else:
        directory = run_directory(config, args.config, args.part, args.out)
    write_config(directory, config, args.part)
    return config, args.part, directory, None


def train(config, part, directory, state=None, max_minutes=None, before_checkpoint=None):
    """Run (or continue) a training in `directory`: 'done' with its summary, or 'paused'."""
    torch.set_num_threads(config["training"].get("threads", 1))
    experiment = build_experiment(config, part)
    trainer = experiment.trainer
    epochs = config["training"]["epochs"]
    if state is not None:
        trainer.load_state_dict(state)
        console.note(f"  resumed at epoch {trainer.epoch}/{epochs}: {directory}")
    else:
        console.note(f"  new run, {epochs} epochs: {directory}")
    start = time.perf_counter()
    deadline = start + 60 * max_minutes if max_minutes is not None else None
    try:
        trainer.fit(epochs, desc=f"train {part}",
                    on_epoch_end=checkpoint_callback(directory, config["training"].get("checkpoint_every", 10), deadline,
                                                 before_checkpoint))
    except TimeBudgetReached as pause:
        console.note(f"  {pause} (time budget of {max_minutes:g} min); run the same command to continue")
        return "paused", None
    summary = save_run(directory, experiment, time.perf_counter() - start)
    print_summary(summary)
    return "done", summary


def print_summary(summary):
    """A few lines instead of summary.json (which is on disk)."""
    console.note(f"  done in {console.duration(summary['seconds'])}: best val {summary['best_val_loss']:.3e} "
                 f"at epoch {summary['best_epoch']}/{summary['epochs']}")
    console.note(f"  final: train {summary['final_train_loss']:.3e}  val {summary['final_val_loss']:.3e}  "
                 f"lr {summary['final_lr']:.1e}  ({summary['device']})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?")
    parser.add_argument("--part", choices=PARTS)
    parser.add_argument("--epochs", type=int, help="override training.epochs")
    parser.add_argument("--seed", type=int, help="override training.seed")
    parser.add_argument("--out", help="parent directory of the run (default: runs/)")
    parser.add_argument("--resume", help="run directory to continue from its checkpoint")
    parser.add_argument("--run-dir", help="fixed run directory: skip if complete, resume if checkpointed")
    parser.add_argument("--max-minutes", type=float, help="pause cleanly after this many minutes")
    args = parser.parse_args(argv)

    if args.run_dir and run_status(args.run_dir) == "done":
        console.note(f"{args.run_dir}: already complete")
        return Path(args.run_dir)
    config, part, directory, state = prepare(args)
    console.header(f"train: {args.config or args.resume} [{part}]")
    train(config, part, directory, state, args.max_minutes)
    return directory


if __name__ == "__main__":
    main()
