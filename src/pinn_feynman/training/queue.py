"""Run a queue of trainings, idempotently.

    python -m pinn_feynman.training.queue configs/box_massless/queues/<name>.yaml [--max-minutes M] [--only NAME ...]
        [--status]

Queue file:
    runs:
      - {tune: box_massless/tune}                                    # a tuning study (tuning/tune.py)
      - {config: box_massless/tuned/box_massless_mlp_v1, part: real, seed: 1}   # its best configuration
      - {config: box_massless/r3, part: real, seed: 1}
      - {config: box_massless/baseline, part: imag}

`config` is a path under configs/ without .yaml (subfolders allowed; the first element is the problem),
or an explicit .yaml path. Each training gets a deterministic directory under runs_dir():
<problem>-<config path after the problem, "/" -> "_">_<part>[_seed<k>]. A training whose configuration
does not exist yet (e.g. a tuned configuration of a study not finished) is 'waiting' and skipped.

An optional top-level `diagnostics:` block of the queue turns the recording of the training dynamics on for the NEW
runs of the queue, as an execution option and not as another experiment: its content is merged into `training.diagnostics`
of the configuration (the configuration's own values win), the run keeps its name (no `_gradrec` configuration needed) and
`training.diagnostics.*` stays out of the identity of the run (EXECUTION_PREFIXES). Runs already started keep the
configuration they were made with. Example:
    diagnostics: {gradients: {every: 64}}
    runs: [...]

A `tune` entry runs or resumes the study up to its total of trials; once finished, its best
configuration is published as configs/<problem>/tuned/<study>.yaml (tuning/publish.py).

Running the queue again skips complete runs and studies and resumes interrupted ones; with
--max-minutes the current run or study pauses cleanly when the budget is spent (and the queue stops).

A run directory that already exists (complete or interrupted) must have been made with the configuration
the queue would use now: otherwise the queue stops with ConfigMismatch instead of skipping or resuming it
(e.g. old runs of a configuration changed since).
"""

import argparse
import copy
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

import yaml

from .. import progress as console
from ..config import load_config
from ..paths import configs_dir, runs_dir
from .run import load_run_state, new_config, run_status, train, write_config


class ConfigMismatch(RuntimeError):
    """An existing run directory was made with a configuration different from the queue's."""


def _flatten(tree, prefix=""):
    if not isinstance(tree, dict):
        return {prefix: tree}
    out = {}
    for key, value in tree.items():
        out.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    return out


#: execution settings: they do not change the experiment
EXECUTION_KEYS = {"training.device", "training.threads", "training.log_every", "training.checkpoint_every"}
EXECUTION_PREFIXES = ("training.diagnostics.",)      # recording of the training dynamics


def config_differences(spec, directory):
    """Dotted keys where the configuration saved in `directory` differs from the one the queue would use
    (execution settings excluded)."""
    saved = _flatten(load_config(Path(directory) / "config.yaml"))
    wanted = _flatten({**new_config(spec.config_path, seed=spec.seed), "part": spec.part})
    return sorted(k for k in (saved.keys() | wanted.keys()) - EXECUTION_KEYS
                  if not k.startswith(EXECUTION_PREFIXES) and saved.get(k) != wanted.get(k))


def check_existing_run(spec, directory):
    """Raise ConfigMismatch if an existing run of `spec` was made with another configuration."""
    if run_status(directory) == "new" or not spec.config_path.exists():
        return
    diff = config_differences(spec, directory)
    if diff:
        raise ConfigMismatch(f"{spec.run_name}: the run in {directory} was made with a different configuration "
                             f"({', '.join(diff[:8])}{', ...' if len(diff) > 8 else ''}); move it away or rename "
                             "the configuration")


class LocalSession:
    """The queue runs alone on its storage: every run is taken at once, nothing to lock."""

    Lost = ()                                           # exceptions meaning "stop without writing": none

    def __init__(self, **options):
        if options:
            raise TypeError(f"unknown queue options {sorted(options)}")

    def claim(self, directory):
        """True if this session may work in `directory` (a run or a tuning study)."""
        return True

    def acquire(self, spec, directory):
        """None to train the run now, or the outcome of skipping it."""
        return None

    def heartbeat(self, directory):
        """Called before every checkpoint."""

    def release(self, directory):
        """Called when the run completes or pauses."""

    def lost(self, error):
        return "lost"

    def listing(self, base):
        return None

    def status_note(self, spec, directory, text, listing=None, tuning=False):
        return text

    @staticmethod
    def add_arguments(parser):
        """Command-line options of the session."""

    @staticmethod
    def options(args):
        return {}


#: optional module that lets several sessions share one queue; absent from the public repository
_SESSIONS = f"{__package__}.queue_sessions"


def session_class():
    """The session class of the optional module if it is there, else `LocalSession`."""
    return import_module(_SESSIONS).SESSION if find_spec(_SESSIONS) is not None else LocalSession


@dataclass(frozen=True)
class RunSpec:
    """One training: configuration '<problem>/[<subfolder>/]<name>', part, optional seed override."""

    config: str
    part: str
    seed: int | None = None
    diagnostics: dict | None = field(default=None, compare=False, hash=False)   # the `diagnostics:` block of the queue

    @property
    def explicit(self):
        return self.config.endswith((".yaml", ".yml"))

    @property
    def config_path(self):
        """configs/<problem>/[...]/<name>.yaml, or the file itself if `config` is a path to a .yaml."""
        if self.explicit:
            return Path(self.config)
        parts = self.config.split("/")
        if len(parts) < 2:
            raise ValueError(f"config {self.config!r}: expected '<problem>/<name>'")
        return configs_dir().joinpath(*parts[:-1], parts[-1] + ".yaml")

    @property
    def problem(self):
        return self.config_path.parent.name if self.explicit else self.config.split("/")[0]

    @property
    def run_name(self):
        seed = f"_seed{self.seed}" if self.seed is not None else ""
        stem = self.config_path.stem if self.explicit else "_".join(self.config.split("/")[1:])
        return f"{self.problem}-{stem}_{self.part}{seed}"

    def directory(self, base=None):
        return Path(base or runs_dir()) / self.run_name


@dataclass(frozen=True)
class TuneSpec:
    """One tuning study: configs/<problem>/<name>.yaml given as '<problem>/<name>', optional total of trials."""

    tune: str
    n_trials: int | None = None

    @property
    def problem(self):
        return self.tune.split("/")[0]

    @property
    def tuning_file(self):
        """Repository-relative path of the tuning file (written in the header of the tuned configuration)."""
        return f"configs/{self.tune}.yaml"

    @property
    def tuning_path(self):
        return configs_dir().parent / self.tuning_file

    @property
    def run_name(self):
        return f"tune:{self.tune}"

    def directory(self, base=None):
        from ..tuning.tune import study_directory
        return study_directory(self.tuning_path, Path(base) / "tuning" if base else None)


def load_queue(path):
    """RunSpecs and TuneSpecs from a queue YAML file."""
    with open(path, encoding="utf-8") as stream:
        queue = yaml.safe_load(stream)
    diagnostics = queue.get("diagnostics") or None
    return [TuneSpec(tune=e["tune"], n_trials=e.get("n_trials")) if "tune" in e
            else RunSpec(config=e["config"], part=e["part"], seed=e.get("seed"), diagnostics=diagnostics)
            for e in queue["runs"]]


def with_diagnostics(config, diagnostics):
    """`config` with the queue's `diagnostics` block merged into `training.diagnostics` (the configuration's values win)."""
    if diagnostics:
        training = config.setdefault("training", {})
        training["diagnostics"] = {**copy.deepcopy(diagnostics), **(training.get("diagnostics") or {})}
    return config


def run_tune(spec, base=None, max_minutes=None, session=None, **options):
    """Run or resume a study; once finished, publish its best configuration.

    'done', 'paused', 'busy', or 'publish_failed' (the configuration is written locally anyway, so the
    trainings that use it can run).
    """
    from ..tuning.publish import PublishError, publish_study
    from ..tuning.tune import run_study
    session = session or session_class()(**options)
    outcome, directory = run_study(spec.tuning_path, Path(base) / "tuning" if base else None, spec.n_trials,
                                   max_minutes, session)
    if outcome != "done":
        return outcome
    try:
        publish_study(directory, spec.tuning_file, spec.problem, root=configs_dir().parent)
    except PublishError as err:
        console.note(f"  publication failed: {err}")
        return "publish_failed"
    return "done"


def run_one(spec, base=None, max_minutes=None, session=None, **options):
    """'done' (already or now), 'paused' (time budget spent) or 'waiting' (configuration not there yet); a session
    shared with others may also skip a run (see `session_class`)."""
    session = session or session_class()(**options)
    if isinstance(spec, TuneSpec):
        return run_tune(spec, base, max_minutes, session)
    directory = spec.directory(base)
    check_existing_run(spec, directory)
    if run_status(directory) == "done":
        console.note("  already complete: skipped")
        return "done"
    if not spec.config_path.exists():
        console.note(f"  waiting: {spec.config_path} does not exist yet")
        return "waiting"
    skipped = session.acquire(spec, directory)
    if skipped is not None:
        return skipped
    try:
        status = run_status(directory)  # re-read: another session may have finished it meanwhile
        if status == "done":
            return "done"
        if status == "partial":
            config, part, state = load_run_state(directory)
        else:
            config, part, state = with_diagnostics(new_config(spec.config_path, seed=spec.seed), spec.diagnostics), spec.part, None
            write_config(directory, config, part)
        outcome, _ = train(config, part, directory, state, max_minutes,
                           before_checkpoint=lambda: session.heartbeat(directory))
        return outcome
    except session.Lost as lost:
        return session.lost(lost)
    finally:
        session.release(directory)


def finished_runs(specs, base=None, workers=16):
    """{run name: ConfigMismatch or None} for the runs that are already complete, read with several threads.

    On a network or cloud folder every file access has a delay of its own, so the runs are probed in parallel. A complete run whose
    configuration differs from the queue's carries its ConfigMismatch, raised when the queue reaches it."""
    def probe(spec):
        directory = spec.directory(base)
        if run_status(directory) != "done":
            return spec.run_name, False, None
        try:
            check_existing_run(spec, directory)
        except ConfigMismatch as err:
            return spec.run_name, True, err
        return spec.run_name, True, None

    runs = [s for s in specs if isinstance(s, RunSpec)]
    found, started = {}, time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for k, (name, done, err) in enumerate(pool.map(probe, runs), 1):
            if done:
                found[name] = err
            if k % 50 == 0:
                console.note(f"  ... {k}/{len(runs)} checked ({console.duration(time.perf_counter() - started)})")
    return found


def run_queue(specs, base=None, max_minutes=None, **options):
    """Run the specs in order; stop at the first pause. Returns {run name: outcome}."""
    session = session_class()(**options)
    outcomes = {}
    start = time.perf_counter()
    console.note(f"checking which of the {len(specs)} entries are already complete...")
    finished = finished_runs(specs, base)
    console.note(f"  {len(finished)} complete, checked in {console.duration(time.perf_counter() - start)}")
    for k, spec in enumerate(specs, 1):
        kind = "tuning" if isinstance(spec, TuneSpec) else "training"
        console.header(f"[{k}/{len(specs)}] {spec.run_name} ({kind})")
        if spec.run_name in finished:
            if finished[spec.run_name] is not None:
                raise finished[spec.run_name]
            console.note("  already complete: skipped")
            outcomes[spec.run_name] = "done"
            continue
        outcomes[spec.run_name] = run_one(spec, base, max_minutes, session)
        if outcomes[spec.run_name] == "paused":
            break
    console.header(f"queue: {len(outcomes)}/{len(specs)} entries handled in {console.duration(time.perf_counter() - start)}")
    print_table(outcomes)
    return outcomes


def print_table(status):
    """{name: text} as two aligned columns (queue status or outcomes)."""
    console.table(status.items())


def queue_status(specs, base=None, **options):
    """{name: 'done' | 'partial' | 'new' | 'waiting'}, running nothing."""
    session = session_class()(**options)
    status = {}
    listing = session.listing(base)
    for spec in specs:
        directory = spec.directory(base)
        if isinstance(spec, TuneSpec):
            from ..tuning.tune import study_status
            text = study_status(directory)
        else:
            text = run_status(directory)
            if text == "new" and not spec.config_path.exists():
                text = "waiting"
            elif text != "new" and spec.config_path.exists() and config_differences(spec, directory):
                text += " (DIFFERENT CONFIG: the queue would stop here)"
        status[spec.run_name] = session.status_note(spec, directory, text, listing, tuning=isinstance(spec, TuneSpec))
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("queue")
    parser.add_argument("--max-minutes", type=float)
    parser.add_argument("--only", nargs="*", help="run names to include")
    parser.add_argument("--status", action="store_true", help="only print the status of each run")
    session = session_class()
    session.add_arguments(parser)
    args = parser.parse_args(argv)
    options = session.options(args)
    specs = load_queue(args.queue)
    if args.only:
        specs = [s for s in specs if s.run_name in args.only]
    if args.status:
        status = queue_status(specs, **{k: v for k, v in options.items() if k == "stale_minutes"})
        print_table(status)
        return status
    return run_queue(specs, max_minutes=args.max_minutes, **options)


if __name__ == "__main__":
    main()
