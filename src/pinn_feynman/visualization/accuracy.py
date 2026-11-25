"""Accuracy of the network against the exact solution (Boni's Fig. 6.3-6.7): curves, errors along x and by
component, histograms, comparisons of runs. The figures along x take `singular` ([(x, label, kind)], dashed vertical
lines, `singularities.mark`), the maps `singular_lines` (`singularities.mark_lines`)."""

import warnings

import numpy as np

from .singularities import mark, mark_lines
from .style import INK, INK_2, SERIES, STYLE, SURFACE, _save, order_label, plt


def _log_bins(values, n=50):
    v = values[np.isfinite(values) & (values > 0)]
    lo, hi = np.log10(v.min()), np.log10(v.max())
    return np.logspace(lo, hi, n)


def error_histograms(errors, orders, path, title, xlabel, reference_mean=None):
    """Fig. 6.3: distribution of the errors, one panel per order in eps (log bins, proportions)."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(orders), figsize=(2.3 * len(orders), 2.8), sharex=True, sharey=True)
        bins = _log_bins(errors.ravel())
        for k, (ax, order) in enumerate(zip(axes, orders)):
            v = errors[:, :, k].ravel()
            v = v[np.isfinite(v) & (v > 0)]
            ax.hist(v, bins=bins, weights=np.full(v.size, 1 / max(v.size, 1)), color=SERIES[0],
                    edgecolor=SURFACE, linewidth=0.5)
            with warnings.catch_warnings():          # no values (relative errors of an order that vanishes
                warnings.simplefilter("ignore", RuntimeWarning)      # identically): "mean nan", as always drawn
                ax.axvline(v.mean(), color=INK, linewidth=1)
                ax.text(0.03, 0.95, f"{order_label(order)}\nmean {v.mean():.1e}", transform=ax.transAxes,
                        va="top", fontsize=8)
            if reference_mean is not None:
                ax.axvline(reference_mean, color=INK_2, linewidth=1, linestyle="--")
            ax.set_xscale("log")
            ax.set_xlabel(xlabel)
        axes[0].set_ylabel("proportion")
        note = "solid: mean of this run" + ("; dashed: mean reported by Boni" if reference_mean else "")
        fig.suptitle(f"{title}   ({note})", x=0.01, ha="left", fontsize=9)
        _save(fig, path)


def real_vs_imag(abs_real, abs_imag, path):
    """Fig. 6.4: absolute-error distributions of the two trainings."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 3.4))
        both = np.concatenate([abs_real.ravel(), abs_imag.ravel()])
        bins = _log_bins(both)
        for values, color, label in ((abs_real, SERIES[0], "real part"), (abs_imag, SERIES[1], "imaginary part")):
            v = values.ravel()
            v = v[v > 0]
            ax.hist(v, bins=bins, weights=np.full(v.size, 1 / v.size), histtype="step", color=color,
                    linewidth=2, label=f"{label} (mean {v.mean():.1e})")
        ax.set_xscale("log")
        ax.set_xlabel("absolute error")
        ax.set_ylabel("proportion")
        ax.legend(loc="upper left")
        _save(fig, path)


def solution_curves(x_boni, prediction, target, orders, path, title, shown_orders=(-2, -1, 0), xlabel=None,
                    master_labels=None, singular=None):
    """Fig. 6.5-6.7: network against exact solution, per master integral and order, in x_B (or `xlabel`).
    `master_labels`: one panel title per master integral (default $M_1$, $M_2$, ...)."""
    order_index = [list(orders).index(o) for o in shown_orders]
    n_masters = target.shape[1]
    sort = np.argsort(x_boni)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(n_masters, len(shown_orders), figsize=(3.2 * len(shown_orders), 2.4 * n_masters),
                                 squeeze=False)
        for i in range(n_masters):
            for j, k in enumerate(order_index):
                ax = axes[i, j]
                ax.plot(x_boni[sort], target[sort, i, k], color=INK_2, linewidth=2, label="exact")
                ax.plot(x_boni[sort], prediction[sort, i, k], color=SERIES[0], linewidth=1.2, linestyle="--",
                        label="network")
                name = master_labels[i] if master_labels is not None else rf"$M_{i + 1}$"
                ax.set_title(f"{name}, {order_label(shown_orders[j])}", loc="left", fontsize=9)
                mark(ax, singular)
                if i == n_masters - 1:
                    ax.set_xlabel(r"$x_B = 10\,t/s$" if xlabel is None else xlabel)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", ncol=2)
        fig.suptitle(title, x=0.01, ha="left", fontsize=9)
        _save(fig, path)


def error_vs_x(x_boni, errors, orders, path, title, n_bins=40, xlabel=None, singular=None):
    """Mean absolute error per order in bins of x_B (not in the thesis: distance from x = 0); `xlabel` replaces the
    massless label."""
    edges = np.linspace(x_boni.min(), x_boni.max(), n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    which = np.clip(np.digitize(x_boni, edges) - 1, 0, n_bins - 1)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.8, 3.6))
        for k, order in enumerate(orders):
            per_point = errors[:, :, k].mean(axis=1)
            means = [per_point[which == b].mean() if np.any(which == b) else np.nan for b in range(n_bins)]
            ax.plot(centres, means, color=SERIES[k % len(SERIES)], label=order_label(order))
        ax.set_yscale("log")
        mark(ax, singular)
        ax.set_xlabel(r"$x_B = 10\,t/s$   (physical singularity at $x_B=0$, spurious at $x_B=-10$)"
                      if xlabel is None else xlabel)
        ax.set_ylabel("mean absolute error")
        ax.set_title(title, loc="left")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        _save(fig, path)


def component_heatmap(errors, orders, path, title, master_labels=None):
    """Mean absolute error of each (master integral, order): an n_masters x n_orders table in one hue (log scale).
    `master_labels`: one row label per master integral (default $M_1$, $M_2$, ...). The colour scale uses the values
    > 0 only: a component with error exactly 0 (an order masked to zero, D2b) is left white and written "0"."""
    from matplotlib.colors import LogNorm

    table = np.nanmean(errors, axis=0)                         # (n_masters, n_orders)
    zero = table == 0
    positive = table[np.isfinite(table) & (table > 0)]
    lo, hi = (positive.min(), positive.max()) if positive.size else (1.0, 10.0)
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(6.4, 2.6))
        shown = np.ma.masked_where(zero, table) if zero.any() else table
        im = ax.imshow(shown, cmap="Blues", norm=LogNorm(vmin=lo, vmax=hi))
        ax.set_xticks(range(len(orders)), [order_label(o) for o in orders])
        ax.set_yticks(range(table.shape[0]), master_labels if master_labels is not None
                      else [rf"$M_{i + 1}$" for i in range(table.shape[0])])
        for i in range(table.shape[0]):
            for k in range(table.shape[1]):
                if zero[i, k]:
                    ax.text(k, i, "0", ha="center", va="center", fontsize=8, color=INK)
                    continue
                dark = np.log(table[i, k] / lo) > 0.6 * np.log(hi / lo)
                ax.text(k, i, f"{table[i, k]:.1e}", ha="center", va="center", fontsize=8,
                        color=SURFACE if dark else INK)
        fig.colorbar(im, ax=ax, label="mean absolute error")
        ax.set_title(title, loc="left")
        _save(fig, path)


def error_vs_distance(x_boni, abs_err, rel_err, orders, path, title, n_bins=30, xlabel=None):
    """Mean absolute and relative error per order against the distance |x_B| from the physical
    singularity x_B = 0, in logarithmic bins (objective 2 of the project; two panels); `xlabel` replaces the
    massless label."""
    dist = np.abs(x_boni)
    edges = np.logspace(np.log10(dist.min()), np.log10(dist.max()), n_bins + 1)
    centres = np.sqrt(edges[:-1] * edges[1:])
    which = np.clip(np.digitize(dist, edges) - 1, 0, n_bins - 1)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for ax, errors, label in ((axes[0], abs_err, "mean absolute error"), (axes[1], rel_err, "mean relative error")):
            for k, order in enumerate(orders):
                per_point = np.nanmean(errors[:, :, k], axis=1)
                means = [np.nanmean(per_point[which == b]) if np.any(which == b) else np.nan for b in range(n_bins)]
                ax.plot(centres, means, color=SERIES[k % len(SERIES)], label=order_label(order))
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"distance from the physical singularity, $|x_B|$" if xlabel is None else xlabel)
            ax.set_ylabel(label)
        axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        fig.suptitle(title, x=0.01, ha="left", fontsize=9)
        _save(fig, path)


def error_map(points, errors, orders, path, title, outline=None, axis_labels=None, gridsize=30,
              colorbar_label="mean absolute error", singular_lines=None):
    """Two variables: mean absolute error (over the master integrals) per order on the plane of the points, one
    panel per order, hexagonal cells holding the mean of their points; one logarithmic colour scale shared by the
    panels. `points` (N, 2) in the physical variables, `errors` (N, n_masters, n_orders), `outline` the vertices
    (K, 2) of the domain, drawn as a closed polygon."""
    from matplotlib.colors import LogNorm

    points = np.asarray(points)
    per_point = np.nanmean(errors, axis=1)                    # (N, n_orders)
    positive = per_point[np.isfinite(per_point) & (per_point > 0)]
    norm = LogNorm(vmin=positive.min(), vmax=positive.max()) if positive.size else None
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, axes = plt.subplots(1, len(orders) + 1, figsize=(2.9 * len(orders) + 0.8, 3.0),
                                 gridspec_kw={"width_ratios": [1] * len(orders) + [0.07]})
        axes, colorbar_axis = axes[:-1], axes[-1]
        image = None
        for k, (ax, order) in enumerate(zip(axes, orders)):
            image = ax.hexbin(points[:, 0], points[:, 1], C=per_point[:, k], reduce_C_function=np.mean,
                              gridsize=gridsize, cmap="Blues", norm=norm, linewidths=0)
            if outline is not None:
                closed = np.vstack([outline, outline[:1]])
                ax.plot(closed[:, 0], closed[:, 1], color=INK_2, linewidth=1)
            mark_lines(ax, singular_lines)
            ax.set_title(order_label(order), loc="left", fontsize=9)
            ax.set_aspect("equal", adjustable="box")
            if axis_labels:
                ax.set_xlabel(axis_labels[0])
            if k:
                ax.set_yticklabels([])
        if axis_labels:
            axes[0].set_ylabel(axis_labels[1])
        if image is not None:
            fig.colorbar(image, cax=colorbar_axis, label=colorbar_label)
        else:
            colorbar_axis.set_visible(False)
        fig.suptitle(title, x=0.01, ha="left", fontsize=9)
        _save(fig, path)


def error_vs_edge_distance(distances, abs_err, rel_err, orders, path, title, edge_labels=None, n_bins=20):
    """Several boundaries: mean absolute (top row) and relative (bottom row) error per order against the distance
    from the nearest edge, one column per edge, each with the points nearest to it (log bins). `distances` (N, E):
    the distance of every point from each of the E edges."""
    distances = np.asarray(distances)
    nearest = np.argmin(distances, axis=1)
    n_edges = distances.shape[1]
    labels = edge_labels or [f"edge {j + 1}" for j in range(n_edges)]
    with plt.rc_context(STYLE), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN means: orders without relative errors
        fig, axes = plt.subplots(2, n_edges, figsize=(3.6 * n_edges + 1.2, 6.0), sharey="row", squeeze=False)
        for j in range(n_edges):
            mine = nearest == j
            dist = distances[mine, j]
            for row, (errors, label) in enumerate(((abs_err, "mean absolute error"), (rel_err, "mean relative error"))):
                ax = axes[row, j]
                if dist.size and dist.max() > dist.min() > 0:
                    edges = np.logspace(np.log10(dist.min()), np.log10(dist.max()), n_bins + 1)
                    centres = np.sqrt(edges[:-1] * edges[1:])
                    which = np.clip(np.digitize(dist, edges) - 1, 0, n_bins - 1)
                    for k, order in enumerate(orders):
                        per_point = np.nanmean(errors[mine][:, :, k], axis=1)
                        means = [np.nanmean(per_point[which == b]) if np.any(np.isfinite(per_point[which == b]))
                                 else np.nan for b in range(n_bins)]
                        ax.plot(centres, means, color=SERIES[k % len(SERIES)], label=order_label(order))
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                if row == 0:
                    ax.set_title(f"nearest edge {labels[j]} ({int(mine.sum())} points)", loc="left", fontsize=9)
                else:
                    ax.set_xlabel(f"distance from {labels[j]}")
                if j == 0:
                    ax.set_ylabel(label)
        handles, names = axes[0, 0].get_legend_handles_labels()
        if handles:
            axes[0, -1].legend(handles, names, loc="center left", bbox_to_anchor=(1.0, 0.5))
        fig.suptitle(title, x=0.01, ha="left", fontsize=9)
        _save(fig, path)


def compare_errors(rows, path):
    """Mean (dot) and median (ring) absolute error per run, on a log axis."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(rows) + 1.4))
        y = np.arange(len(rows))
        ax.scatter([r["abs_mean"] for r in rows], y, color=SERIES[0], s=40, label="mean", zorder=3)
        ax.scatter([r["abs_median"] for r in rows], y, facecolors="none", edgecolors=SERIES[1], s=40,
                   linewidths=2, label="median", zorder=3)
        ax.set_yticks(y, [r["label"] for r in rows])
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel("absolute error on the test set")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        _save(fig, path)


def compare_error_vs_x(curves, path, n_bins=40, singular=None):
    """Mean absolute error (over masters and orders) along x_B, one line per run (<= 5 runs)."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        for k, (label, x_boni, per_point) in enumerate(curves):
            edges = np.linspace(x_boni.min(), x_boni.max(), n_bins + 1)
            which = np.clip(np.digitize(x_boni, edges) - 1, 0, n_bins - 1)
            means = [per_point[which == b].mean() if np.any(which == b) else np.nan for b in range(n_bins)]
            ax.plot((edges[:-1] + edges[1:]) / 2, means, color=SERIES[k % len(SERIES)], label=label)
        ax.set_yscale("log")
        mark(ax, singular)
        ax.set_xlabel(r"$x_B = 10\,t/s$")
        ax.set_ylabel("mean absolute error")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        _save(fig, path)


def compare_residuals(curves, path, n_bins=80, singular=None):
    """Physics residual norm along x_B (binned maximum and mean), one line per run: where the
    solution has not propagated from the boundary data (Daw et al., sec. 3)."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        for k, (label, x, norms) in enumerate(curves):
            edges = np.linspace(x.min(), x.max(), n_bins + 1)
            which = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
            means = [norms[which == b].mean() if np.any(which == b) else np.nan for b in range(n_bins)]
            ax.plot((edges[:-1] + edges[1:]) / 2, means, color=SERIES[k % len(SERIES)], label=label)
        ax.set_yscale("log")
        mark(ax, singular)
        ax.set_xlabel(r"$x_B = 10\,t/s$")
        ax.set_ylabel("physics residual norm (rms over 15 components)")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        _save(fig, path)


def compare_error_vs_distance(curves, path, n_bins=30):
    """Mean absolute error against the distance |x_B| from the physical singularity (log-log),
    one line per run: objective 2 of the project."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        for k, (label, x_boni, per_point) in enumerate(curves):
            dist = np.abs(x_boni)
            edges = np.logspace(np.log10(dist.min()), np.log10(dist.max()), n_bins + 1)
            which = np.clip(np.digitize(dist, edges) - 1, 0, n_bins - 1)
            means = [per_point[which == b].mean() if np.any(which == b) else np.nan for b in range(n_bins)]
            ax.plot(np.sqrt(edges[:-1] * edges[1:]), means, color=SERIES[k % len(SERIES)], label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"distance from the physical singularity, $|x_B|$")
        ax.set_ylabel("mean absolute error")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        _save(fig, path)


def compare_error_maps(points, per_group, path, outline=None, axis_labels=None, gridsize=30,
                       colorbar_label="mean absolute error", singular_lines=None):
    """Several variables: one panel per group with the mean absolute error (geometric mean over the seeds of the group,
    the test points being the same) on the plane of the points, one logarithmic colour scale shared by the panels.
    `points` (N, 2) in the physical variables, `per_group` = [(label, per-point error (N,))]."""
    from matplotlib.colors import LogNorm

    points = np.asarray(points)
    every = np.concatenate([np.asarray(v)[np.isfinite(v) & (np.asarray(v) > 0)] for _, v in per_group])
    norm = LogNorm(vmin=every.min(), vmax=every.max()) if every.size else None
    n = len(per_group)
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, axes = plt.subplots(1, n + 1, figsize=(2.9 * n + 0.8, 3.0), gridspec_kw={"width_ratios": [1] * n + [0.07]},
                                 squeeze=False)
        axes, colorbar_axis = axes[0, :-1], axes[0, -1]
        image = None
        for k, (ax, (label, values)) in enumerate(zip(axes, per_group)):
            image = ax.hexbin(points[:, 0], points[:, 1], C=np.asarray(values), reduce_C_function=np.mean,
                              gridsize=gridsize, cmap="Blues", norm=norm, linewidths=0)
            if outline is not None:
                closed = np.vstack([outline, outline[:1]])
                ax.plot(closed[:, 0], closed[:, 1], color=INK_2, linewidth=1)
            mark_lines(ax, singular_lines)
            ax.set_title(label, loc="left", fontsize=9)
            ax.set_aspect("equal", adjustable="box")
            if axis_labels:
                ax.set_xlabel(axis_labels[0])
            if k:
                ax.set_yticklabels([])
        if axis_labels:
            axes[0].set_ylabel(axis_labels[1])
        if image is not None:
            fig.colorbar(image, cax=colorbar_axis, label=colorbar_label)
        else:
            colorbar_axis.set_visible(False)
        _save(fig, path)


def compare_error_vs_edge_distance(distances, per_group, path, edge_labels=None, n_bins=20, colours=None):
    """Several boundaries: mean absolute error against the distance from the nearest edge (log bins), one panel per
    edge (the points nearest to it), one line per group. `distances` (N, E); `per_group` = [(label, error)] with the
    error (N,), or (seeds, N): then thin lines per seed and a thick median. `colours` {label: colour} (default: the
    series colours in order)."""
    distances = np.asarray(distances)
    nearest = np.argmin(distances, axis=1)
    n_edges = distances.shape[1]
    labels = edge_labels or [f"edge {j + 1}" for j in range(n_edges)]
    with plt.rc_context(STYLE), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fig, axes = plt.subplots(1, n_edges, figsize=(3.6 * n_edges + 1.2, 3.4), sharey=True, squeeze=False)
        for j, ax in enumerate(axes[0]):
            mine = nearest == j
            dist = distances[mine, j]
            if dist.size and dist.max() > dist.min() > 0:
                edges = np.logspace(np.log10(dist.min()), np.log10(dist.max()), n_bins + 1)
                centres = np.sqrt(edges[:-1] * edges[1:])
                which = np.clip(np.digitize(dist, edges) - 1, 0, n_bins - 1)
                for k, (label, values) in enumerate(per_group):
                    colour = (colours or {}).get(label, SERIES[k % len(SERIES)])
                    rows = np.atleast_2d(np.asarray(values))[:, mine]
                    curves = [[np.nanmean(v[which == b]) if np.any(which == b) else np.nan for b in range(n_bins)]
                              for v in rows]
                    if len(curves) > 1:
                        for c in curves:
                            ax.plot(centres, c, color=colour, linewidth=0.6, alpha=0.4)
                        ax.plot(centres, np.nanmedian(curves, axis=0), color=colour, linewidth=1.6,
                                label=f"{label} (median of {len(curves)})")
                    else:
                        ax.plot(centres, curves[0], color=colour, label=label)
                ax.set_xscale("log")
                ax.set_yscale("log")
            ax.set_title(f"nearest edge {labels[j]} ({int(mine.sum())} points)", loc="left", fontsize=9)
            ax.set_xlabel(f"distance from {labels[j]}")
        axes[0, 0].set_ylabel("mean absolute error")
        handles, names = axes[0, 0].get_legend_handles_labels()
        if handles:
            axes[0, -1].legend(handles, names, loc="center left", bbox_to_anchor=(1.0, 0.5))
        _save(fig, path)
