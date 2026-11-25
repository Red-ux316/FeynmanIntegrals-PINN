"""Generate (or find in the cache) the datasets of one or more configurations.

    python -m pinn_feynman.data.build configs/box_massless/baseline.yaml [more configs ...] [--workers N] [--only NAME ...]
    python -m pinn_feynman.data.build configs/box_one_mass/baseline.yaml

A dataset shared by several configurations (same points, same problem) is listed and built once.
"""

import argparse
import os
import time

from .. import progress as console
from ..config import load_config
from .cache import cache_path, get_dataset
from .dataset import DatasetSpec


def dataset_specs(config):
    """{name: DatasetSpec} from the `data` section of a configuration."""
    data = config["data"]
    return {name: DatasetSpec.from_config(data["problem"], data["domain"], entry)
            for name, entry in data["datasets"].items()}


def unique_specs(config_paths, only=None):
    """[(name, spec)] of all the configurations, each distinct dataset once, in order of appearance."""
    seen, out = set(), []
    for path in config_paths:
        for name, spec in dataset_specs(load_config(path)).items():
            if (only and name not in only) or spec in seen:
                continue
            seen.add(spec)
            out.append((name, spec))
    return out


def describe(spec):
    """`[-0.9, -0.1]  uniform  1000 pt  seed 1` (or `triangle (margin 0.1)  ...`), plus the Chebyshev
    nodes of an interpolated dataset."""
    text = f"{spec.domain.label()}  {spec.sampler:8s} {spec.n:>7d} pt"
    if spec.seed is not None:
        text += f"  seed {spec.seed}"
    if spec.method == "chebyshev":
        text += f"  from {spec.n_nodes} Chebyshev nodes"
    return text


def build_all(config_paths, workers=1, only=None):
    """Build the missing datasets of the configurations, with a progress bar each. Returns [(name, spec)]."""
    specs = unique_specs(config_paths, only)
    missing = [(name, spec) for name, spec in specs if not cache_path(spec).exists()]
    console.header(f"data: {len(config_paths)} configuration(s), {len(specs)} dataset(s), "
                   f"{len(missing)} to compute ({workers} worker(s))")
    for name, spec in specs:
        if (name, spec) not in missing:
            console.note(f"  {name:11s} {describe(spec)}  cached")
            continue
        console.note(f"  {name:11s} {describe(spec)}  computing")
        start = time.perf_counter()
        get_dataset(spec, workers=workers, progress=name)
        console.note(f"  {name:11s} done in {console.duration(time.perf_counter() - start)}")
    return specs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="+", help="YAML configuration file(s)")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="parallel processes")
    parser.add_argument("--only", nargs="*", help="dataset names to build (default: all)")
    args = parser.parse_args(argv)
    return build_all(args.config, args.workers, args.only)


if __name__ == "__main__":
    main()
