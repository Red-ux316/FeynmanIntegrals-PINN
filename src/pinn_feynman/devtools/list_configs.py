"""Inheritance tree of the configuration files, with what each one changes with respect to its base.

Usage:
    python -m pinn_feynman.devtools.list_configs [problem] [--no-diff]

Read-only: it parses the YAML files of `configs/<problem>/` (default: every problem) and prints
one line per file, indented under its `base:`, followed by the keys it overrides (dotted paths).
"""

import argparse
from pathlib import Path

import yaml

from pinn_feynman.paths import configs_dir


def read_raw(path):
    """The YAML file as written, without resolving `base:`."""
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def flatten(node, prefix=""):
    """Leaf values of a nested dict as {dotted.path: value}; lists are leaves."""
    if not isinstance(node, dict):
        return {prefix: node}
    out = {}
    for key, value in node.items():
        out.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    return out


def config_files(folder):
    """Configuration files directly inside `folder` (queues/ and other subfolders excluded)."""
    return sorted(Path(folder).glob("*.yaml"))


def base_of(path):
    """Name of the file a configuration inherits from (`base:`), or None for a root."""
    return read_raw(path).get("base")


def overrides(path):
    """{dotted.path: value} of the keys written in the file itself (not inherited)."""
    raw = read_raw(path)
    raw.pop("base", None)
    return flatten(raw)


def children_map(folder):
    """{base name or None: [file names that inherit from it]}; every file appears once."""
    out = {}
    for path in config_files(folder):
        out.setdefault(base_of(path), []).append(path.name)
    return out


def tree_lines(folder, with_diff=True):
    """Lines of the inheritance tree of one config folder."""
    folder = Path(folder)
    children = children_map(folder)
    names = {p.name for p in config_files(folder)}
    roots = sorted(children.get(None, []) + [b for b in children if b is not None and b not in names])
    lines = []

    def visit(name, depth):
        indent = "  " * depth
        marker = "" if name in names else "  (missing!)"
        lines.append(f"{indent}{name}{marker}")
        if with_diff and name in names:
            changed = overrides(folder / name)
            for key, value in changed.items():
                lines.append(f"{indent}    {key}: {value}")
        for child in sorted(children.get(name, [])):
            visit(child, depth + 1)

    for root in roots:
        visit(root, 0)
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("problem", nargs="?", help="folder name under configs/ (default: all)")
    parser.add_argument("--no-diff", action="store_true", help="only the tree, without the overridden keys")
    args = parser.parse_args(argv)
    root = configs_dir()
    folders = [root / args.problem] if args.problem else sorted(p for p in root.iterdir() if p.is_dir())
    for folder in folders:
        print(f"== {folder.name}")
        print("\n".join(tree_lines(folder, with_diff=not args.no_diff)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
