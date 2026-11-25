"""Search space of a tuning study: YAML description -> Optuna suggestions -> configuration.

Each entry of `search_space` is a dotted path into the training configuration with a
distribution:
    training.optimizer.lr: {type: float, low: 1.0e-3, high: 3.0e-2, log: true}
    model.activation:      {type: categorical, choices: [gelu, tanh, silu]}
    model.width:           {type: categorical, choices: [16, 32, 64]}
    model.depth:           {type: int, low: 2, high: 4}
`model.width` and `model.depth` are special: they become model.hidden = [width] * depth.
"""

import copy

from ..config import merge

DISTRIBUTIONS = ("float", "int", "categorical")


def suggest(trial, space):
    """{dotted path: value} drawn by an Optuna trial from the search space."""
    values = {}
    for path, spec in space.items():
        kind = spec["type"]
        if kind == "float":
            values[path] = trial.suggest_float(path, spec["low"], spec["high"], log=spec.get("log", False))
        elif kind == "int":
            values[path] = trial.suggest_int(path, spec["low"], spec["high"], log=spec.get("log", False))
        elif kind == "categorical":
            values[path] = trial.suggest_categorical(path, spec["choices"])
        else:
            raise ValueError(f"unknown distribution {kind!r} for {path}; available: {DISTRIBUTIONS}")
    return values


def set_path(config, path, value):
    """Set config[a][b][c] = value for path 'a.b.c' (creating dicts if needed)."""
    keys = path.split(".")
    node = config
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def apply(config, values):
    """Configuration with the suggested values; model.width/depth -> model.hidden."""
    out = copy.deepcopy(config)
    for path, value in values.items():
        set_path(out, path, value)
    model = out.get("model", {})
    if "width" in model or "depth" in model:
        width = model.pop("width", model["hidden"][0])
        depth = model.pop("depth", len(model["hidden"]))
        model["hidden"] = [int(width)] * int(depth)
    return out


def trial_config(base_config, budget, values):
    """Base configuration, reduced by the budget, with the trial's values."""
    return apply(merge(base_config, budget or {}), values)
