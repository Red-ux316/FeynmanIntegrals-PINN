"""Check a Chebyshev-interpolated dataset against an exactly computed one.

    python -m pinn_feynman.data.check_interpolation configs/box_massless/baseline.yaml [--interpolated test] [--exact validation]

The interpolant behind `--interpolated` is evaluated at the points of `--exact` and compared
with their exact values, for the real and imaginary parts.
"""

import argparse

import numpy as np

from ..config import load_config
from .build import dataset_specs
from .cache import get_dataset, interpolate_reference, nodes_spec


def interpolation_errors(nodes_dataset, exact_dataset):
    """{part: (max absolute error, max error relative to max |value|)} per part."""
    interpolated = interpolate_reference(nodes_dataset, exact_dataset.x)
    out = {}
    for part, pick in (("real", np.real), ("imag", np.imag)):
        diff = np.abs(pick(interpolated) - pick(exact_dataset.values))
        out[part] = (diff.max(), diff.max() / np.abs(pick(exact_dataset.values)).max())
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--interpolated", default="test")
    parser.add_argument("--exact", default="validation")
    args = parser.parse_args(argv)
    specs = dataset_specs(load_config(args.config))
    spec = specs[args.interpolated]
    if spec.method != "chebyshev":
        raise SystemExit(f"dataset {args.interpolated!r} is not interpolated")
    nodes = get_dataset(nodes_spec(spec))
    exact = get_dataset(specs[args.exact])
    print(f"interpolant: {spec.n_nodes} Chebyshev nodes; checked at {len(exact)} exact points ({args.exact})")
    for part, (abs_err, rel_err) in interpolation_errors(nodes, exact).items():
        print(f"  {part}: max abs error {abs_err:.3e}, relative to max |value| {rel_err:.3e}")


if __name__ == "__main__":
    main()
