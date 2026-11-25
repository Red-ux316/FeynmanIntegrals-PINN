"""YAML configuration files, with inheritance.

A configuration may start with `base: <file>.yaml` (path relative to its own folder): it is
then the base configuration updated with the keys written here. Dictionaries are merged
recursively, with one rule: a section whose `name` differs from the base one (a different
component, e.g. `collocation: {name: r3, ...}`) replaces the base section entirely, so no
parameter of the old component survives. Everything else is overwritten.
"""

from pathlib import Path

import torch
import yaml

DTYPES = {"float32": torch.float32, "float64": torch.float64}


def merge(base, override):
    """Recursive merge of two configuration dicts (see module docstring)."""
    out = dict(base)
    for key, value in override.items():
        old = out.get(key)
        if isinstance(value, dict) and isinstance(old, dict) and value.get("name", old.get("name")) == old.get("name"):
            out[key] = merge(old, value)
        else:
            out[key] = value
    return out


#: libyaml-based safe loader when available (about 10 times faster: the queue reads hundreds of configurations)
_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def load_config(path):
    """Read a YAML configuration file into a dict, resolving `base:` inheritance."""
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        config = yaml.load(stream, Loader=_YAML_LOADER) or {}
    base = config.pop("base", None)
    if base is not None:
        config = merge(load_config(path.parent / base), config)
    return config


def torch_dtype(name):
    """torch dtype from its name in the configuration ('float32' or 'float64')."""
    try:
        return DTYPES[name]
    except KeyError:
        raise ValueError(f"unsupported dtype {name!r}; available: {sorted(DTYPES)}") from None
