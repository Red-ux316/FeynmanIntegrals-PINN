"""Project locations, derived from the package position and the storage root (no hard-coded paths).

Configurations and code live in the repository; datasets (data/) and training outputs (runs/)
live under the storage root of `pinn_feynman.runtime` ($PINN_FEYNMAN_STORAGE, else the
unversioned folder storage/ at the repository root).
"""

import os
from pathlib import Path

from .runtime import repository_root, storage_root

#: environment variable overriding the dataset cache directory only
DATA_DIR_ENV = "PINN_FEYNMAN_DATA_DIR"


def project_root():
    """Repository root: the package lives in <root>/src/pinn_feynman."""
    return repository_root()


def data_dir():
    """Dataset cache directory: $PINN_FEYNMAN_DATA_DIR, else <storage>/data (not versioned)."""
    override = os.environ.get(DATA_DIR_ENV)
    return Path(override) if override else storage_root() / "data"


def runs_dir():
    """Training outputs directory: <storage>/runs (not versioned)."""
    return storage_root() / "runs"


def configs_dir():
    """Configuration files: <root>/configs/<problem>/<name>.yaml (always in the repository)."""
    return project_root() / "configs"


#: problems whose configurations live in the folder of another problem (same system family)
CONFIG_FOLDERS = {"sunrise_equal_mass_positive": "sunrise_equal_mass"}


def config_folder(problem):
    """Folder of the configuration files of a problem (`configs/<problem>`, or the one of `CONFIG_FOLDERS`)."""
    return configs_dir() / CONFIG_FOLDERS.get(problem, problem)


def config_file(problem, name="baseline"):
    """Path of the configuration `name` of a problem."""
    return config_folder(problem) / f"{name}.yaml"
