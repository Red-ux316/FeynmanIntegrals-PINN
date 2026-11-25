"""The data behind the figures, saved next to a report (`report/figure_data/`) so that a figure can be redrawn without
recomputing anything and a comparison can reuse what the run reports already computed.

A nested dict of numpy arrays, numbers, strings, None, lists and tuples is flattened to `a/b/c` keys: arrays go to a
compressed npz, everything else to a JSON entry inside the same file (key `__json__`), so one file per object.
"""

import json
from pathlib import Path

import numpy as np

JSON_KEY = "__json__"
# per-point arrays that the figures do not use (they draw cells and decades): left out of the saved profiles
POINTWISE = {"abs_x", "e2", "eabs", "ratio", "lam", "grid_x", "r", "population_x"}


def _flatten(value, prefix, arrays, rest):
    if isinstance(value, dict):
        rest[prefix + "#type"] = "dict"
        for key, item in value.items():
            _flatten(item, f"{prefix}/{key}" if prefix else str(key), arrays, rest)
    elif isinstance(value, (list, tuple)) and any(isinstance(v, (np.ndarray, dict, list, tuple)) for v in value):
        rest[prefix + "#type"] = "list"
        rest[prefix + "#len"] = len(value)
        for k, item in enumerate(value):
            _flatten(item, f"{prefix}/{k}", arrays, rest)
    elif isinstance(value, np.ndarray) and value.dtype != object:
        arrays[prefix] = value
    elif isinstance(value, np.generic):
        rest[prefix] = value.item()
    else:
        rest[prefix] = list(value) if isinstance(value, tuple) else value


def save_data(path, data):
    """Write `data` (nested dict) to `path` (.npz)."""
    arrays, rest = {}, {}
    _flatten(data, "", arrays, rest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays[JSON_KEY] = np.frombuffer(json.dumps(rest).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(path, **arrays)


def _build(prefix, arrays, rest):
    kind = rest.get(prefix + "#type")
    if kind == "dict":
        start = f"{prefix}/" if prefix else ""
        children = sorted({k[len(start):].split("/")[0].split("#")[0] for k in list(arrays) + list(rest)
                           if k.startswith(start) and k != prefix + "#type" and k[len(start):]})
        return {child: _build(start + child, arrays, rest) for child in children}
    if kind == "list":
        return [_build(f"{prefix}/{k}", arrays, rest) for k in range(rest[prefix + "#len"])]
    if prefix in arrays:
        return arrays[prefix]
    return rest.get(prefix)


def load_data(path):
    """The nested dict saved by `save_data`."""
    with np.load(path, allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files if k != JSON_KEY}
        rest = json.loads(bytes(npz[JSON_KEY]).decode("utf-8"))
    return _build("", arrays, rest)


def slim(data):
    """A copy of a zone profile or recorder data without the per-point arrays (the figures draw cells and decades)."""
    if isinstance(data, dict):
        return {k: slim(v) for k, v in data.items() if k not in POINTWISE}
    if isinstance(data, (list, tuple)):
        return [slim(v) for v in data]
    return data


def history_columns(rows):
    """History rows (list of dicts) -> {column: array}, NaN where a row has no value."""
    keys = sorted({k for r in rows for k in r})
    return {k: np.array([r.get(k, np.nan) for r in rows], dtype=float) for k in keys}


def history_rows(columns):
    """{column: array} -> history rows, dropping NaN cells (as `catalog.read_history_rows`)."""
    n = len(next(iter(columns.values()))) if columns else 0
    return [{k: float(v[i]) for k, v in columns.items() if np.isfinite(v[i])} for i in range(n)]
