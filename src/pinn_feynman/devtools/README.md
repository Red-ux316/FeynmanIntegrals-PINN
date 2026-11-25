# pinn_feynman.devtools

Tools for people working on the code.

## `codemap.py`
Generates `docs/code_map.md` via **static analysis** with `ast`: it neither imports nor executes anything.

| Map section | Content |
|---|---|
| 1 | repository files with a one-line description (docstring, first heading or first comment); generated files excluded |
| 2 | for each module: public functions, classes, methods and constants, with signature, line and **every point where the name occurs** in `src/` (and in `tests/`, if the repository has it) |
| 3 | imports between the package's modules |
| 4 | `python -m ...` entry points |
| 5 | public functions and classes that no test refers to by name |

Commands:
- `python -m pinn_feynman.devtools.codemap`: regenerates the map;
- `python -m pinn_feynman.devtools.codemap --check`: exit code 1 if it is not up to date.

Limitation: occurrences are found **by name**, so for methods with generic names (`update`, `to`, `epoch`...), marked with ⚠️, different objects with the same name also show up.

## `benchmark.py`
`python -m pinn_feynman.devtools.benchmark <config> --device cpu|cuda [--dtype] [--batch-size] [--steps] [--threads]`: milliseconds per training step with the real code, plus the estimated minutes for a full training run. One JSON line.

## `list_configs.py`
`python -m pinn_feynman.devtools.list_configs [problem] [--no-diff]`: read-only inheritance tree of the files in `configs/<problem>/` (`base:`), each file with the keys it overrides (dotted paths); a missing base is flagged. Index of the files: `configs/<problem>/README.md`.
