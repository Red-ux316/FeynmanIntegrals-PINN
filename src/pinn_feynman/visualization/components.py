"""Figures per component M_i^(k) of the network output: the masters and the orders in epsilon separately.

Arrays are in the network layout (N, n_masters, n_orders). `groups` = {label: [array of seed 0, of seed 1, ...]}: thin
line per seed, thick median, the colour of the group (`methods.group_styles`); a `reference` array (the exact values)
is drawn in black. The x axis is |x|, linear by default (Boni's domain is within one order of magnitude).
`singular`: [(position on the x axis, label, kind)] from `singularities.singular_points`, dashed vertical lines.
"""

import numpy as np

from .singularities import mark
from .style import INK_2, domain_box, SERIES, STYLE, _save, plt

ORDER_LABELS = ("eps^-2", "eps^-1", "eps^0", "eps^1", "eps^2")


def _styles(labels):
    from .methods import group_styles
    return group_styles(list(labels))


def _family(ax, x, curves, colour, label, style="-", mean=False):
    curves = [np.asarray(c, float) for c in curves if c is not None]
    if not curves:
        return
    if len(curves) == 1:
        ax.plot(x, curves[0], color=colour, linestyle=style, label=label)
        return
    for c in curves:
        ax.plot(x, c, color=colour, linestyle=style, linewidth=0.6, alpha=0.4)
    average = np.nanmean if mean else np.nanmedian
    ax.plot(x, average(curves, axis=0), color=colour, linestyle=style, linewidth=1.6,
            label=f"{label} ({'mean' if mean else 'median'} of {len(curves)})")


def _seed_lines(ax, x, curves, colour, label, style="-", log=True):
    """Every seed as a thin faded line of the group colour and, on top, the mean over the seeds as a thick one
    (geometric if `log`, as on a log axis); the label says how many seeds. One seed: one plain line."""
    curves = [np.asarray(c, float) for c in curves if c is not None]
    if not curves:
        return
    if len(curves) == 1:
        ax.plot(x, curves[0], color=colour, linestyle=style, linewidth=1.2, alpha=0.95, label=f"{label} (1 seed)")
        return
    for c in curves:
        ax.plot(x, c, color=colour, linestyle=style, linewidth=0.5, alpha=0.4)
    with np.errstate(divide="ignore", invalid="ignore"):
        stack = np.array(curves)
        mean = np.exp(np.nanmean(np.log(np.where(stack > 0, stack, np.nan)), axis=0)) if log else np.nanmean(stack, axis=0)
    ax.plot(x, mean, color=colour, linestyle=style, linewidth=1.8, alpha=0.95, label=f"{label} ({len(curves)} seeds, mean)")


def _band_means(ax, edges, values, colour):
    """The mean of each band as a faded segment behind the curves; `values` (n_bands,), NaN skipped."""
    for b, v in enumerate(values):
        if np.isfinite(v):
            ax.plot(edges[b:b + 2], [v, v], color=colour, linewidth=4.0, alpha=0.25, solid_capstyle="butt", zorder=0)


def _axis(ax, xlog, ylog):
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.tick_params(labelsize=6)


def component_grid(x, groups, path, title, reference=None, ylog=False, xlog=False, ylabel="", max_points=1500,
                   order_labels=ORDER_LABELS, master_labels=None, singular=None, signed=False, mean=False, xlabel=None):
    """One panel per component: rows = masters, columns = orders. `x` (N,) physical points (plotted as |x|, or as x on a
    linear axis with `signed`; sorted and thinned to `max_points`; `mean`: the bold curve is the mean of the seeds, not
    the median); `groups` {label: [(N, n_masters, n_orders)]}; `reference` (N, n_masters, n_orders), a
    dict {name: such an array} (several exact curves, drawn black with different dashes) or None. `master_labels`: the
    name of each master row (default M1, M2, ...; with a bottom-up network the rows are the masters it learns).
    With `ylog` the absolute value is drawn. `xlabel` replaces the default "x" / "|x|" (e.g. s1 along a cut)."""
    references = reference if isinstance(reference, dict) else ({"exact": reference} if reference is not None else {})
    ax_x = np.asarray(x, float) if signed else np.abs(np.asarray(x, float))
    order = np.argsort(ax_x)
    keep = order[np.linspace(0, len(order) - 1, min(max_points, len(order))).astype(int)]
    xs = ax_x[keep]
    first = next(iter(groups.values()))[0] if groups else next(iter(references.values()))
    n_masters, n_orders = np.asarray(first).shape[1:]
    styles = _styles(groups)
    value = (lambda a: np.abs(a)) if ylog else (lambda a: a)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(n_masters, n_orders, figsize=(2.9 * n_orders, 2.2 * n_masters), squeeze=False,
                                 sharex=True)
        for i in range(n_masters):
            for k in range(n_orders):
                ax = axes[i][k]
                for j, (name, exact) in enumerate(references.items()):
                    ax.plot(xs, value(np.asarray(exact)[keep, i, k]), color="#0b0b0b", linewidth=1.0,
                            linestyle=("-", "--", ":")[j % 3], label=name, zorder=5)      # above the network
                for label, runs in groups.items():
                    _family(ax, xs, [value(np.asarray(r)[keep, i, k]) for r in runs], styles[label][0], label, mean=mean)
                _axis(ax, xlog, ylog)
                mark(ax, singular)
                domain_box(ax)
                if i == 0:
                    ax.set_title(order_labels[k] if k < len(order_labels) else f"order {k}", fontsize=8)
                if k == 0:
                    ax.set_ylabel(f"{master_labels[i] if master_labels else f'M{i + 1}'} {ylabel}".strip(), fontsize=7)
                if i == n_masters - 1:
                    ax.set_xlabel(xlabel or ("x" if signed else "|x|"), fontsize=7)
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", fontsize=7)
        if title:
            fig.suptitle(title, x=0.01, ha="left", fontsize=9)
        _save(fig, path)
    return True


def per_order_cells(centres, groups, path, title, ylabel, ylog=True, xlog=False, references=None,
                    order_labels=ORDER_LABELS, xlabel="|x|", singular=None):
    """One panel per order; in each, one colour per master. `centres` (C,) of the cells (`xlabel` says which coordinate); `groups`
    {label: [(C, n_masters, n_orders)]} (normally one group: the masters take the colours); `references`
    {name: (C,)} curves drawn dashed in grey in every panel (e.g. 1/|x|). With a single master and several groups (the
    two parts of a bottom-up report) the colours are the groups."""
    runs = next(iter(groups.values()))
    n_masters, n_orders = np.asarray(runs[0]).shape[1:]
    by_group = n_masters == 1 and len(groups) > 1
    styles = _styles(groups)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, n_orders, figsize=(3.3 * n_orders, 3.0), squeeze=False, sharey=True)
        for k in range(n_orders):
            ax = axes[0][k]
            if by_group:
                for label, group_runs in groups.items():
                    _family(ax, centres, [np.asarray(r)[:, 0, k] for r in group_runs], styles[label][0], label)
            else:
                for i in range(n_masters):
                    _family(ax, centres, [np.asarray(r)[:, i, k] for r in runs], SERIES[i % len(SERIES)], f"M{i + 1}")
            for name, curve in (references or {}).items():
                ax.plot(centres, curve, color=INK_2, linestyle="--", linewidth=0.8, label=name)
            _axis(ax, xlog, ylog)
            mark(ax, singular)
            domain_box(ax)
            ax.set_title(order_labels[k] if k < len(order_labels) else f"order {k}", fontsize=8)
            ax.set_xlabel(xlabel, fontsize=7)
        axes[0][0].set_ylabel(ylabel, fontsize=7)
        axes[0][0].legend(fontsize=6)
        if title:
            fig.suptitle(title, x=0.01, ha="left", fontsize=9)
        _save(fig, path)
    return True


def segment_grid(curves, path, master_labels, order_labels, singular=None, shade=None, xlabel="x", network=None, ylabel=""):
    """The true values over a window, one panel per component (rows = masters, columns = orders). `curves`
    {part: [(x (N,), f (N, n_masters, n_orders))]}: the pieces of the curve of each part; every piece is drawn on its own,
    so that two pieces are never joined (no interpolation across a gap); both parts solid, the colours of
    `methods.PART_COLOUR`; `singular` [(x, label, kind)] dashed lines; `shade` [(x0, x1, label)] grey span (the standard
    domain). `network` {part: [(x (N,), [f (N, n_masters, n_orders) per seed])]}: the network over the pieces (one per
    domain shown), thin line per seed and the mean bold, the colour of the part; the true curve is black and drawn below the network."""
    from .methods import PART_COLOUR
    first = next(f for pieces in curves.values() for _, f in pieces)
    n_masters, n_orders = np.asarray(first).shape[1:]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(n_masters, n_orders, figsize=(2.9 * n_orders, 2.2 * n_masters), squeeze=False, sharex=True)
        for i in range(n_masters):
            for k in range(n_orders):
                ax = axes[i][k]
                shade_spans(ax, shade if (i, k) == (0, 0) else [(a, b, "") for a, b, _ in (shade or [])])
                for part, pieces in (network or {}).items():
                    for number, (x, seeds) in enumerate(pieces):
                        colour = PART_COLOUR.get(part, SERIES[0])
                        for seed in seeds:
                            ax.plot(x, np.asarray(seed)[:, i, k], color=colour, linewidth=0.6, alpha=0.35, zorder=6)
                        ax.plot(x, np.mean([np.asarray(s)[:, i, k] for s in seeds], axis=0), color=colour, linewidth=1.3, zorder=7,
                                label=f"network {part}" if number == 0 else None)
                for part, pieces in curves.items():
                    for number, (x, f) in enumerate(pieces):
                        ax.plot(x, np.asarray(f)[:, i, k], color=("#0b0b0b" if part == "real" else "#5a5a5a") if network else PART_COLOUR.get(part, SERIES[0]),
                                linewidth=1.0, linestyle="-", zorder=2 if network else 5,
                                label=(f"true {part}" if network else part) if number == 0 else None)
                _axis(ax, False, False)
                mark(ax, singular)
                if i == 0:
                    ax.set_title(order_labels[k] if k < len(order_labels) else f"order {k}", fontsize=8)
                if k == 0:
                    ax.set_ylabel(f"{master_labels[i] if master_labels else f'M{i + 1}'} {ylabel}".strip(), fontsize=7)
                if i == n_masters - 1:
                    ax.set_xlabel(xlabel, fontsize=7)
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", fontsize=7)
        _save(fig, path)
    return True


def shade_spans(ax, shade):
    """Light grey spans behind the curves, `shade` = [(x0, x1, label)] (the zones crossed by a cut), the label on top."""
    for x0, x1, label in (shade or []):
        ax.axvspan(x0, x1, color=INK_2, alpha=0.08, zorder=0, linewidth=0)
        ax.text(0.5 * (x0 + x1), 1.0, label, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6,
                color=INK_2)


def block_curves(x, series, path, title, ylabel, ylog=True, references=None, xlabel="x", bands=None, singular=None,
                 shade=None):
    """A quantity along x in one panel. `x` (B,) block centres (linear axis); `series` {name: [(B,) per seed]}, one
    colour per name, every seed a line; `bands` = (edges (n_bands+1,), {name: (n_bands,) mean of the band}) drawn faded
    behind; `references` {name: (B,)} dashed grey; `shade` [(x0, x1, label)] light grey spans (e.g. the zones along a
    cut)."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        shade_spans(ax, shade)
        for j, (name, runs) in enumerate(series.items()):
            if bands is not None and name in bands[1]:
                _band_means(ax, bands[0], bands[1][name], SERIES[j % len(SERIES)])
            _seed_lines(ax, x, runs, SERIES[j % len(SERIES)], name, log=ylog)
        for j, (name, curve) in enumerate((references or {}).items()):
            ax.plot(x, curve, color=INK_2, linestyle=("--", ":", "-.")[j % 3], linewidth=0.9, label=name)
        _axis(ax, False, ylog)
        mark(ax, singular)
        domain_box(ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        if title:
            ax.set_title(title, loc="left", fontsize=9)
        _save(fig, path)
    return True


def per_order_blocks(x, groups, path, title, ylabel, ylog=True, order_labels=ORDER_LABELS, xlabel="x", bands=None,
                     singular=None):
    """One panel per order, one colour per master (or per group with a single master), every seed a line.
    `groups` {label: [(B, n_masters, n_orders)]}; `bands` = (edges, {label: [(n_bands, n_masters, n_orders) per seed]}),
    drawn as the mean over the seeds."""
    runs = next(iter(groups.values()))
    n_masters, n_orders = np.asarray(runs[0]).shape[1:]
    by_group = n_masters == 1 and len(groups) > 1
    styles = _styles(groups)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, n_orders, figsize=(3.3 * n_orders, 3.0), squeeze=False, sharey=True)
        for k in range(n_orders):
            ax = axes[0][k]
            entries = ([(label, styles[label][0], 0, group_runs) for label, group_runs in groups.items()] if by_group else
                       [(f"M{i + 1}", SERIES[i % len(SERIES)], i, runs) for i in range(n_masters)])
            for label, colour, i, group_runs in entries:
                if bands is not None:
                    key = label if by_group else next(iter(bands[1]))
                    mean = np.nanmean([np.asarray(b)[:, i, k] for b in bands[1][key]], axis=0)
                    _band_means(ax, bands[0], mean, colour)
                _seed_lines(ax, x, [np.asarray(r)[:, i, k] for r in group_runs], colour, label, log=ylog)
            _axis(ax, False, ylog)
            mark(ax, singular)
            domain_box(ax)
            ax.set_title(order_labels[k] if k < len(order_labels) else f"order {k}", fontsize=8)
            ax.set_xlabel(xlabel, fontsize=7)
        axes[0][0].set_ylabel(ylabel, fontsize=7)
        axes[0][0].legend(fontsize=6)
        if title:
            fig.suptitle(title, x=0.01, ha="left", fontsize=9)
        _save(fig, path)
    return True


def cell_curves(centres, series, path, title, ylabel, ylog=True, xlog=False, references=None, xlabel="|x|",
                singular=None):
    """Several quantities per cell in one panel: `series` {name: [(C,) per seed]} (one colour per name, thin seeds,
    thick median); `references` {name: (C,)} dashed grey."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        for j, (name, runs) in enumerate(series.items()):
            _family(ax, centres, runs, SERIES[j % len(SERIES)], name)
        for j, (name, curve) in enumerate((references or {}).items()):
            ax.plot(centres, curve, color=INK_2, linestyle=("--", ":", "-.")[j % 3], linewidth=0.9, label=name)
        _axis(ax, xlog, ylog)
        mark(ax, singular)
        domain_box(ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        if title:
            ax.set_title(title, loc="left", fontsize=9)
        _save(fig, path)
    return True


def component_errors(errors, orders, path, title, master_labels=None, ylabel="mean absolute error"):
    """Error of each component (master x order), per group: a dot per seed and a bar at the geometric mean, one colour
    per group (the groups are normally the real and imaginary parts, `methods.PART_COLOUR`). `errors` {label: [(n_masters,
    n_orders) per seed]}; one panel per master. Components that are identically zero are left to the caller."""
    styles = _styles(errors)
    first = np.asarray(next(iter(errors.values()))[0])
    n_masters = first.shape[0]
    n_groups = len(errors)
    width = 0.8 / n_groups
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, n_masters, figsize=(1.0 * len(orders) * n_groups + 2.5, 3.2), squeeze=False)
        for i in range(n_masters):
            ax = axes[0][i]
            for g, (label, runs) in enumerate(errors.items()):
                values = np.array([np.asarray(r)[i] for r in runs], float)          # (seeds, orders)
                centre = np.arange(len(orders)) - 0.4 + width * (g + 0.5)
                for v in values:
                    ax.scatter(centre, v, color=styles[label][0], s=14, zorder=3)
                mean = np.exp(np.nanmean(np.log(np.maximum(values, 1e-300)), axis=0))
                ax.hlines(mean, centre - width / 2, centre + width / 2, color=styles[label][0], linewidth=2, zorder=2,
                          label=label if i == 0 else None)
            ax.set_yscale("log")
            ax.set_xticks(range(len(orders)), [f"eps^{o}" for o in orders], fontsize=7)
            ax.set_title(master_labels[i] if master_labels else f"M{i + 1}", loc="left", fontsize=8)
            if i == 0:
                ax.set_ylabel(ylabel)
        axes[0][0].legend(fontsize=7)
        if title:
            fig.suptitle(title, x=0.01, ha="left", fontsize=9)
        _save(fig, path)
    return True
