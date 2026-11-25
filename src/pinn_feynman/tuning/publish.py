"""Write the best configuration of a finished study as a NEW configuration file.

The file is configs/<problem>/tuned/<study>.yaml, in the local copy of the code, so that a queue can
train it at once. An existing file is never overwritten: same content -> kept, different content ->
error. Only paths configs/<problem>/tuned/<name>.yaml are accepted.
"""

import json
import re
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

import yaml

from .. import progress as console
from ..paths import project_root

TUNED_PATH = re.compile(r"^configs/[a-z0-9_]+/tuned/[A-Za-z0-9_.-]+\.yaml$")
#: optional module that also publishes the file elsewhere; absent from the public repository
_REMOTE = f"{__package__}.publish_github"


class PublishError(RuntimeError):
    """The file could not be written (refused path, different existing file)."""


def tuned_config_path(problem, study_name):
    """Repository-relative path of the tuned configuration of a study."""
    return f"configs/{problem}/tuned/{study_name}.yaml"


def check_path(path):
    if not TUNED_PATH.match(path):
        raise PublishError(f"refused path {path!r}: only configs/<problem>/tuned/<name>.yaml can be published")
    return path


def tuned_config_text(study_dir, tuning_path):
    """best_config.yaml of the study, with a header saying where it comes from."""
    study_dir = Path(study_dir)
    summary = json.loads((study_dir / "summary.json").read_text(encoding="utf-8"))
    best = yaml.safe_load((study_dir / "best_config.yaml").read_text(encoding="utf-8"))
    params = ", ".join(f"{k}={v}" for k, v in summary.get("best_params", {}).items())
    header = [
        f"# Best configuration of the tuning study {summary['study']} ({Path(tuning_path).as_posix()}),",
        f"# {summary['n_finished']} finished trials ({summary['n_complete']} complete, {summary['n_pruned']} pruned),"
        f" best validation loss {summary.get('best_value')} at trial {summary.get('best_trial')}.",
        f"# Best values: {params}.",
        "# Written automatically by pinn_feynman.tuning.publish: do not edit, run a new study.",
    ]
    return "\n".join(header) + "\n" + yaml.safe_dump(best, sort_keys=False)


def write_local(path, text, root=None):
    """Write the file in the local copy of the code; an existing different file is an error."""
    target = Path(root or project_root()) / check_path(path)
    if target.exists():
        if target.read_text(encoding="utf-8") != text:
            raise PublishError(f"{target} already exists with different content: not overwritten")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def publish_study(study_dir, tuning_path, problem, root=None, **options):
    """Write the tuned configuration of a finished study. Returns a record {path, status}."""
    study_dir = Path(study_dir)
    summary = json.loads((study_dir / "summary.json").read_text(encoding="utf-8"))
    path = tuned_config_path(problem, summary["study"])
    if find_spec(_REMOTE) is not None:
        return import_module(_REMOTE).publish_study(study_dir, tuning_path, path, root, **options)
    if options:
        raise TypeError(f"unknown publication options {sorted(options)}")
    write_local(path, tuned_config_text(study_dir, tuning_path), root)
    console.note(f"  tuned config {path}: written")
    return {"path": path, "status": "written"}
