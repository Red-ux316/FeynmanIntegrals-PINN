"""Generate docs/code_map.md: files, public symbols, where each symbol is used, imports.

    python -m pinn_feynman.devtools.codemap            # (re)write the map
    python -m pinn_feynman.devtools.codemap --check    # exit 1 if the map is out of date

Static analysis with `ast` (nothing is imported or executed). Usages are found by name:
a function `f` is "used" wherever the identifier `f` appears (call, import, attribute).
For methods and common names this can include unrelated objects with the same name; the
map says so next to them.
"""

import argparse
import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ..paths import project_root

PACKAGE = "pinn_feynman"
OUTPUT = Path("docs") / "code_map.md"
#: directories never listed (generated or cached)
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".vscode", "pinn_feynman.egg-info", ".ipynb_checkpoints"}
#: top-level directories never listed (dataset cache and training outputs, not versioned)
SKIP_TOP = {"data", "runs"}
SKIP_FILES = {".gitattributes"}
#: optional file at the repository root with more names to skip, one per line (directories end with "/")
LOCAL_SKIP = ".codemap-skip"
#: top-level directories listed as one row, not file by file: name -> description
FOLDED_TOP = {"configs": "YAML configurations of the trainings and the queues that use them, one folder per problem "
                         "(see its README)"}
LISTED_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".csv", ".txt"}
#: method names too generic to attribute usages reliably
GENERIC_NAMES = {"to", "update", "epoch", "part", "state_dict", "load_state_dict", "forward", "append",
                 "column", "tensors", "matrices", "validate", "fit", "losses", "as_dict", "from_config"}


@dataclass
class Symbol:
    name: str
    kind: str
    module: str
    path: Path
    line: int
    signature: str = ""
    doc: str = ""
    usages: list = field(default_factory=list)


def first_line(text):
    return (text or "").strip().splitlines()[0].strip() if (text or "").strip() else ""


def module_name(path, src_root):
    parts = list(path.relative_to(src_root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def python_files(root):
    return sorted(p for p in root.rglob("*.py") if not SKIP_DIRS & set(p.parts))


def _skipped(rel):
    return bool(SKIP_DIRS & set(rel.parts)) or rel.parts[0] in SKIP_TOP


def parse(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _signature(node):
    return f"({ast.unparse(node.args)})"


def public_symbols(module, path, tree):
    """Top-level public functions, classes (with public methods) and UPPERCASE constants."""
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            out.append(Symbol(node.name, "function", module, path, node.lineno, _signature(node),
                              first_line(ast.get_docstring(node))))
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            out.append(Symbol(node.name, "class", module, path, node.lineno, "",
                              first_line(ast.get_docstring(node))))
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and (not item.name.startswith("_") or item.name == "__call__"):
                    out.append(Symbol(f"{node.name}.{item.name}", "method", module, path, item.lineno,
                                      _signature(item), first_line(ast.get_docstring(item))))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    out.append(Symbol(target.id, "constant", module, path, node.lineno))
    return out


def identifier_occurrences(path, tree):
    """{identifier: [line, ...]} for names, attributes and imported names in a file."""
    occ = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            key = node.id
        elif isinstance(node, ast.Attribute):
            key = node.attr
        elif isinstance(node, ast.alias):
            key = node.name.split(".")[-1]
        else:
            continue
        line = getattr(node, "lineno", None)
        if line is None and isinstance(node, ast.alias):
            continue
        occ.setdefault(key, []).append(line)
    return occ


def resolve_import(module, node):
    """Absolute module named by an ImportFrom inside `module` (relative imports resolved)."""
    if node.level == 0:
        return node.module
    base = module.split(".")[: -node.level]         # `module` of a package __init__ ends in ".__init__"
    return ".".join(base + ([node.module] if node.module else []))


def import_edges(module, path, tree):
    edges = set()
    pkg_module = module if path.name != "__init__.py" else module + ".__init__"
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            target = resolve_import(pkg_module, node)
            if node.module is None:                     # from . import a, b  -> the submodules
                edges.update(f"{target}.{alias.name}" for alias in node.names)
            elif target and target.startswith(PACKAGE):
                edges.add(target)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(PACKAGE):
                    edges.add(alias.name)
    return sorted(edges - {module})


@dataclass
class CodeMap:
    root: Path
    modules: dict            # module -> (path, docstring line)
    symbols: list
    imports: dict            # module -> [modules]
    entry_points: list       # modules with a __main__ block
    files: list              # (relative path, description)


def build(root=None):
    root = Path(root or project_root())
    src_root = root / "src"
    modules, symbols, imports, entry_points = {}, [], {}, []
    trees = {}
    for path in python_files(src_root / PACKAGE):
        module = module_name(path, src_root)
        tree = parse(path)
        trees[path] = tree
        modules[module] = (path, first_line(ast.get_docstring(tree)))
        symbols += public_symbols(module, path, tree)
        imports[module] = import_edges(module, path, tree)
        if any(isinstance(n, ast.If) and "__main__" in ast.unparse(n.test) for n in tree.body):
            entry_points.append(module)
    for path in python_files(root / "tests"):
        trees[path] = parse(path)
    occurrences = {path: identifier_occurrences(path, tree) for path, tree in trees.items()}
    for sym in symbols:
        key = sym.name.split(".")[-1]
        for path, occ in occurrences.items():
            lines = sorted(set(l for l in occ.get(key, []) if not (path == sym.path and l == sym.line)))
            if lines:
                sym.usages.append((path, lines))
    return CodeMap(root, modules, symbols, imports, entry_points, list_files(root))


def describe(path):
    """One-line description of a file (docstring, first heading, first comment)."""
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return ""
    if path.suffix == ".py":
        try:
            return first_line(ast.get_docstring(ast.parse(text)))
        except SyntaxError:
            return ""
    for line in text.splitlines():
        line = line.strip()
        if path.suffix == ".md" and line.startswith("#"):
            return line.lstrip("#").strip()
        if path.suffix in (".yaml", ".yml", ".toml") and line.startswith("#"):
            return line.lstrip("#").strip()
    return ""


def local_skips(root):
    """(directory names, file names) from the optional LOCAL_SKIP file."""
    path = root / LOCAL_SKIP
    names = [n.strip() for n in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    names = [n for n in names if n and not n.startswith("#")]
    return {n.rstrip("/") for n in names if n.endswith("/")}, {n for n in names if not n.endswith("/")}


def list_files(root):
    skip_dirs, skip_files = local_skips(root)
    out, folded = [], set()
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if _skipped(rel) or skip_dirs & set(rel.parts) or path.name in SKIP_FILES | skip_files or not path.is_file():
            continue
        if path.suffix not in LISTED_SUFFIXES or rel.parts[0] == "docs" and path.name == OUTPUT.name:
            continue
        top = rel.parts[0]
        if top in FOLDED_TOP and len(rel.parts) > 1:
            if top not in folded:
                folded.add(top)
                out.append((f"{top}/", FOLDED_TOP[top]))
            continue
        out.append((rel.as_posix(), describe(path)))
    return out


def _link(root, path, line=None):
    rel = Path("..") / path.relative_to(root)
    anchor = f"#L{line}" if line else ""
    return f"[`{path.relative_to(root).as_posix()}`]({rel.as_posix()}{anchor})"


def _usage_cell(root, sym):
    if not sym.usages:
        return "—"
    parts = []
    for path, lines in sym.usages:
        rel = path.relative_to(root).as_posix()
        parts.append(f"`{rel}` ({', '.join(str(l) for l in lines[:8])}{', …' if len(lines) > 8 else ''})")
    note = " ⚠️ generic name" if sym.name.split(".")[-1] in GENERIC_NAMES else ""
    return "<br>".join(parts) + note


def render(cmap):
    root = cmap.root
    has_tests = (root / "tests").is_dir()
    lines = [
        "# Code map",
        "",
        "> **Generated file** by `python -m pinn_feynman.devtools.codemap`: do not edit by hand. "
        "`python -m pinn_feynman.devtools.codemap --check` exits with 1 if it is out of date.",
        "",
        "How to use it: to know **what to touch when you change a function**, look it up in section 2 and read "
        f"the *used in* column (files and lines{', tests included' if has_tests else ''}). "
        "Usages are found by name: for generic "
        "names, marked with ⚠️, different objects with the same name may also appear.",
        "",
        "## 1. Repository files",
        "",
        "| File | Description |",
        "|---|---|",
    ]
    lines += [f"| `{rel}` | {desc} |" for rel, desc in cmap.files]
    lines += ["", "## 2. Package modules and public symbols", ""]
    by_module = {}
    for sym in cmap.symbols:
        by_module.setdefault(sym.module, []).append(sym)
    for module, (path, doc) in cmap.modules.items():
        lines += [f"### `{module}`", "", f"{_link(root, path)} — {doc}", ""]
        syms = by_module.get(module, [])
        if syms:
            lines += ["| Symbol | Kind | Signature | Line | Used in |", "|---|---|---|---|---|"]
            for sym in syms:
                sig = f"`{sym.signature}`" if sym.signature else ""
                lines.append(f"| `{sym.name}` | {sym.kind} | {sig} | {sym.line} | {_usage_cell(root, sym)} |")
            lines.append("")
    lines += ["## 3. Imports between package modules", "", "| Module | Imports |", "|---|---|"]
    for module, targets in cmap.imports.items():
        if targets:
            lines.append(f"| `{module}` | {', '.join(f'`{t}`' for t in targets)} |")
    lines += ["", "## 4. Entry point (`python -m ...`)", ""]
    lines += [f"- `python -m {m}`" for m in cmap.entry_points]
    if has_tests:
        lines += ["", "## 5. Public functions and classes not mentioned by any test", "",
                  "Mentioned = the name appears in a file under `tests/`. This does not measure how many lines run "
                  "(that needs `coverage`), but it flags the functions that no test calls directly.", ""]
        lines += [f"- `{sym.module}.{sym.name}`" for sym in untested_symbols(cmap)] or ["- none"]
    lines.append("")
    return "\n".join(lines)


def untested_symbols(cmap):
    """Public functions and classes whose name never appears in tests/."""
    tests_dir = cmap.root / "tests"
    return [sym for sym in cmap.symbols if sym.kind in ("function", "class")
            and not any(tests_dir in path.parents for path, _ in sym.usages)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="fail if the map is out of date")
    args = parser.parse_args(argv)
    root = project_root()
    text = render(build(root))
    target = root / OUTPUT
    if args.check:
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        if current.replace("\r\n", "\n") != text:
            print(f"{OUTPUT} is out of date: run python -m pinn_feynman.devtools.codemap", file=sys.stderr)
            return 1
        print(f"{OUTPUT} is up to date")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
