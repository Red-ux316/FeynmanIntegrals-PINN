"""Where data and runs are stored.

The storage root is the folder that contains data/ (dataset cache) and runs/ (training outputs).
It is the environment variable

    PINN_FEYNMAN_STORAGE   explicit root, e.g. a folder on a larger disk

or, without it, the folder `storage/` at the repository root (not versioned). Code is never
looked up through this root: the package is installed with `pip install -e .`.
"""

import os
import platform
from importlib.util import find_spec
from pathlib import Path

STORAGE_ENV = "PINN_FEYNMAN_STORAGE"
#: name of the default storage folder at the repository root
DEFAULT_STORAGE = "storage"

#: optional module with another default storage lookup; absent from the public repository
_LOOKUP = f"{__package__}.runtime_drive"


def _lookup():
    """The optional storage-lookup module, or None."""
    if find_spec(_LOOKUP) is None:
        return None
    from importlib import import_module

    return import_module(_LOOKUP)


def repository_root():
    """Repository root: the package lives in <root>/src/pinn_feynman."""
    return Path(__file__).resolve().parents[2]


class StorageNotFound(RuntimeError):
    """No storage root could be found."""


def storage_root():
    """Root of data/ and runs/: $PINN_FEYNMAN_STORAGE, else <repository>/storage."""
    explicit = os.environ.get(STORAGE_ENV)
    if explicit:
        return Path(explicit)
    lookup = _lookup()
    if lookup is not None:
        return lookup.drive_storage()
    return repository_root() / DEFAULT_STORAGE


def summary():
    """Device, versions and paths, for the first cell of a notebook or a log."""
    import torch

    from .paths import data_dir, runs_dir

    lookup = _lookup()
    info = lookup.runtime_info() if lookup is not None else {}
    info.update({
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu_threads": torch.get_num_threads(),
        "repository": str(repository_root()),
        "storage": str(storage_root()),
        "data": str(data_dir()),
        "runs": str(runs_dir()),
    })
    return info
