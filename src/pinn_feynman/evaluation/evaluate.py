"""Evaluate trained runs on the test set and draw the figures of the thesis.

    python -m pinn_feynman.evaluation.evaluate <run_dir> [<run_dir> ...] [--dataset test] [--which best|final]

For each run: evaluation.json and figures in the run directory. With one real and one
imaginary run, also real_vs_imag.png (Fig. 6.4) in the parent directory of the first run.
Problems in one variable (massless box) and in several (one-mass box): the figures
along x become, with two variables, a map of the error on the domain and the error against
the distance from each edge.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ..data.build import dataset_specs
from ..data.cache import get_dataset
from ..physics import SYSTEMS
from . import plots
from .boni_reference import boni_part
from .metrics import absolute_error, relative_error, summarize
from .predict import load_run, predict


def evaluate_run(run, dataset):
    """Errors of a run on a dataset: (prediction, target, abs errors, rel errors, report dict)."""
    target = run.target(dataset)
    prediction = predict(run.model, dataset.x, run.input_scale)
    abs_err = absolute_error(prediction, target)
    rel_err = relative_error(prediction, target)
    report = {
        "run": str(run.directory), "part": run.part, "n_points": len(dataset),
        "absolute_error": summarize(abs_err, run.orders),
        "relative_error": summarize(rel_err, run.orders),
        "training": run.summary,
        "boni": boni_part(run.problem, run.part),
    }
    return prediction, target, abs_err, rel_err, report


def draw_figures(run, dataset, prediction, target, abs_err, rel_err, out_dir=None):
    """Figures of a run, in its directory or in `out_dir`."""
    d, part = (Path(out_dir) if out_dir else run.directory), run.part
    system = SYSTEMS[run.problem]
    boni = boni_part(run.problem, part)
    history = read_history(run.directory / "history.csv")
    if dataset.spec.domain.dim > 1:
        _draw_figures_multi(run, dataset, abs_err, rel_err, d, history, system, boni)
        return
    if history:
        plots.loss_curves(history, d / "loss.png", f"{system.TITLE}, {part} part: losses")
        plots.training_dynamics(history, d / "dynamics.png", f"{part} part: ReLoBRaLo weights, learning rate, residual statistics")
    # the massless box declares neither MASTER_NAMES nor FIGURE_LABELS: its calls keep the labels in x_B written in
    # `visualization`
    labels = _master_labels(run, system)
    axis = getattr(system, "FIGURE_LABELS", {})

    def given(**options):
        return {k: v for k, v in options.items() if v is not None}

    plots.component_heatmap(abs_err, run.orders, d / "components.png",
                            f"{part} part: mean absolute error per master integral and order",
                            *(() if labels is None else (labels,)))
    plots.error_histograms(abs_err, run.orders, d / "hist_abs.png", f"{part} part: absolute error",
                           "absolute error", boni.get("abs_error_mean"))
    plots.error_histograms(rel_err, run.orders, d / "hist_rel.png", f"{part} part: relative error",
                           "relative error", boni.get("rel_error_mean"))
    x_boni = run.input_scale * dataset.x
    # dashed lines at the singular points inside the plotted range (none for the massless box on Boni's domain)
    singular = _singular_in_range(run, x_boni) or None
    plots.solution_curves(x_boni, prediction, target, run.orders, d / "curves.png",
                          f"{part} part: network (dashed) and exact solution",
                          **given(xlabel=axis.get("variable"), master_labels=labels,
                                  shown_orders=getattr(system, "SHOWN_ORDERS", None), singular=singular))
    plots.error_vs_x(x_boni, abs_err, run.orders, d / "error_vs_x.png", f"{part} part: error along the domain",
                     **given(xlabel=axis.get("along"), singular=singular))
    plots.error_vs_distance(x_boni, abs_err, rel_err, run.orders, d / "error_vs_distance.png",
                            f"{part} part: error against the distance from the physical singularity",
                            **given(xlabel=axis.get("distance")))


def _singular_in_range(run, x):
    """The singular points of the problem (on the axis input_scale * x) inside the range of `x`, VV-02."""
    from ..visualization.singularities import singular_points
    lo, hi = float(np.min(x)), float(np.max(x))
    return [p for p in singular_points(run.problem, scale=float(run.input_scale)) if lo <= p[0] <= hi]


def _master_labels(run, system):
    """Labels of the masters the network learns, from the MASTER_NAMES of the system (None without them)."""
    names = getattr(system, "MASTER_NAMES", None)
    if not names:
        return None
    columns = range(len(names)) if run.columns is None else run.columns
    return [rf"$M_{i + 1}$ ({names[i].replace('_', ' ')})" for i in columns]


def _draw_figures_multi(run, dataset, abs_err, rel_err, d, history, system, boni):
    """Figures of a run in several variables: losses, dynamics, components, histograms, error map, error by edge."""
    part = run.part
    if history:
        plots.loss_curves(history, d / "loss.png", f"{system.TITLE}, {part} part: losses")
        plots.training_dynamics(history, d / "dynamics.png", f"{part} part: ReLoBRaLo weights, learning rate, residual statistics")
    labels = _master_labels(run, system)                                    # bottom-up: only the unknowns
    plots.component_heatmap(abs_err, run.orders, d / "components.png",
                            f"{part} part: mean absolute error per master integral and order", labels)
    plots.error_histograms(abs_err, run.orders, d / "hist_abs.png", f"{part} part: absolute error",
                           "absolute error", boni.get("abs_error_mean"))
    plots.error_histograms(rel_err, run.orders, d / "hist_rel.png", f"{part} part: relative error",
                           "relative error", boni.get("rel_error_mean"))
    domain = dataset.spec.domain
    outline = domain.vertices() / domain.scale if hasattr(domain, "vertices") else None
    axis_labels = getattr(system, "VARIABLE_LABELS", None)
    plots.error_map(dataset.x, abs_err, run.orders, d / "error_map.png",
                    f"{system.TITLE}, {part} part: mean absolute error (over the master integrals) on the domain",
                    outline=outline, axis_labels=axis_labels, singular_lines=_singular_lines(run))
    if hasattr(system, "edge_distances"):
        plots.error_vs_edge_distance(system.edge_distances(dataset.x), abs_err, rel_err, run.orders,
                                     d / "error_vs_edge_distance.png",
                                     f"{part} part: error against the distance from the nearest edge of the region",
                                     edge_labels=getattr(system, "EDGE_LABELS", None))


def _singular_lines(run):
    """The singular lines of a problem in two variables (VV-02); None without them (several variables, other cases)."""
    from ..visualization.singularities import singular_lines
    return singular_lines(run.problem) or None


def read_history(path):
    if not path.exists():
        return []
    import csv

    with open(path, encoding="utf-8") as stream:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--dataset", default="test")
    parser.add_argument("--which", choices=("best", "final"), default="best")
    args = parser.parse_args(argv)

    abs_by_part = {}
    for directory in args.runs:
        run = load_run(directory, args.which)
        dataset = get_dataset(dataset_specs(run.config)[args.dataset])
        prediction, target, abs_err, rel_err, report = evaluate_run(run, dataset)
        report.update({"dataset": args.dataset, "weights": args.which})
        (run.directory / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        draw_figures(run, dataset, prediction, target, abs_err, rel_err)
        abs_by_part[run.part] = abs_err
        a, r = report["absolute_error"]["overall"], report["relative_error"]["overall"]
        print(f"{run.directory.name} [{run.part}]: abs mean {a['mean']:.2e} median {a['median']:.2e} "
              f"max {a['max']:.2e} | rel mean {r['mean']:.2e} median {r['median']:.2e}", flush=True)
    if {"real", "imag"} <= set(abs_by_part):
        out = Path(args.runs[0]).parent / "real_vs_imag.png"
        plots.real_vs_imag(abs_by_part["real"], abs_by_part["imag"], out)
        print(f"figure: {out}")


if __name__ == "__main__":
    main()
