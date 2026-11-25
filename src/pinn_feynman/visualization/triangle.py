"""Maps on a two-variable domain with a grid of panels: rows = groups (e.g. the real and imaginary parts), columns =
orders in epsilon (the one-mass box).

`accuracy.error_map` draws one group (one row, one panel per order, one colour scale) and `accuracy.compare_error_maps`
one quantity per group (one row, one panel per group). The reports of the bottom-up runs want the parts side by side
and the orders separate in one figure, each part with its own colour scale (the scales of the two parts differ by
orders of magnitude): `order_maps`.

On the fine cells of a triangle (equal small triangles, none cut by the edge): `cell_maps` (a grid of panels, sequential `Blues` on a log scale), `ratio_maps` (a ratio around
1, the only diverging map: log10 of the ratio with `RdBu_r` centred at 0) and `edge_curves` (a quantity against the
distance from the nearest edge, linear axis, thin seeds and a thick mean).
"""

import numpy as np

from .style import INK_2, SERIES, STYLE, _save, order_label, plt


def order_maps(points, rows, orders, path, outline=None, axis_labels=None, colorbar_label="", gridsize=30,
               overlays=None, empty_text="identically zero"):
    """One panel per (row, order): the mean of |value| of the points in each hexagonal cell, logarithmic colours,
    one colour scale and colour bar per row.

    `points` (N, 2) in the physical variables; `rows` {label: (N, n_orders)} (NaN = not drawn: a panel whose values
    are all NaN or zero shows `empty_text`); `outline` (K, 2) the vertices of the domain, drawn as a closed polygon;
    `overlays` [(K, 2)] further closed polygons drawn dashed (e.g. the zones read in the text). Returns False when
    there is nothing to draw."""
    from matplotlib.colors import LogNorm

    points = np.asarray(points, float)
    if not rows:
        return False
    n_rows, n_orders = len(rows), len(orders)
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, axes = plt.subplots(n_rows, n_orders + 1, figsize=(2.6 * n_orders + 0.9, 2.7 * n_rows), squeeze=False,
                                 gridspec_kw={"width_ratios": [1] * n_orders + [0.07]})
        for i, (label, values) in enumerate(rows.items()):
            values = np.abs(np.asarray(values, float)).reshape(len(points), n_orders)
            positive = values[np.isfinite(values) & (values > 0)]
            norm = LogNorm(vmin=positive.min(), vmax=positive.max()) if positive.size else None
            image = None
            for k, order in enumerate(orders):
                ax = axes[i, k]
                column = values[:, k]
                if np.any(np.isfinite(column) & (column > 0)):
                    keep = np.isfinite(column)
                    image = ax.hexbin(points[keep, 0], points[keep, 1], C=np.maximum(column[keep], positive.min()),
                                      reduce_C_function=np.mean, gridsize=gridsize, cmap="Blues", norm=norm,
                                      linewidths=0)
                else:
                    ax.text(0.5, 0.5, empty_text, transform=ax.transAxes, ha="center", va="center", fontsize=7,
                            color=INK_2)
                for polygon, style in ([(outline, "-")] if outline is not None else []) + \
                        [(p, "--") for p in (overlays or [])]:
                    closed = np.vstack([polygon, polygon[:1]])
                    ax.plot(closed[:, 0], closed[:, 1], color=INK_2, linewidth=0.9 if style == "-" else 0.6,
                            linestyle=style)
                ax.set_aspect("equal", adjustable="box")
                ax.tick_params(labelsize=6)
                if i == 0:
                    ax.set_title(order_label(order), loc="left", fontsize=9)
                if k == 0:
                    ax.set_ylabel(f"{label}\n{axis_labels[1]}" if axis_labels else label, fontsize=8)
                else:
                    ax.set_yticklabels([])
                if i == n_rows - 1 and axis_labels:
                    ax.set_xlabel(axis_labels[0], fontsize=8)
                elif i < n_rows - 1:
                    ax.set_xticklabels([])
            if image is not None:
                fig.colorbar(image, cax=axes[i, -1], label=colorbar_label)
            else:
                axes[i, -1].set_visible(False)
        _save(fig, path)
    return True


def _mean(curves, log):
    """Mean over the seeds (first axis), geometric with `log`; NaN where no seed has a value, without warnings."""
    import warnings
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        if log:
            return np.exp(np.nanmean(np.log(np.where(curves > 0, curves, np.nan)), axis=0))
        return np.nanmean(curves, axis=0)


def _outlines(ax, outline, overlays):
    for polygon, style in ([(outline, "-")] if outline is not None else []) + [(p, "--") for p in (overlays or [])]:
        polygon = np.asarray(polygon, float)
        closed = np.vstack([polygon, polygon[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color=INK_2, linewidth=0.9 if style == "-" else 0.6, linestyle=style)


def _lines(ax, lines):
    for segment in (lines or []):
        segment = np.asarray(segment, float)
        ax.plot(segment[:, 0], segment[:, 1], color=INK_2, linewidth=0.6, linestyle=":")


def _grid(rows, columns):
    n_rows, n_cols = len(rows), max(1, len(columns))
    fig, axes = plt.subplots(n_rows, n_cols + 1, figsize=(2.6 * n_cols + 0.9, 2.7 * n_rows), squeeze=False,
                             gridspec_kw={"width_ratios": [1] * n_cols + [0.07]})
    return fig, axes


def _frame(ax, i, k, n_rows, label, title, axis_labels):
    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale_view()
    ax.tick_params(labelsize=6)
    if i == 0 and title:
        ax.set_title(title, loc="left", fontsize=9)
    if k == 0:
        ax.set_ylabel(f"{label}\n{axis_labels[1]}" if axis_labels else label, fontsize=8)
    else:
        ax.set_yticklabels([])
    if i == n_rows - 1 and axis_labels:
        ax.set_xlabel(axis_labels[0], fontsize=8)
    elif i < n_rows - 1:
        ax.set_xticklabels([])


def cell_maps(polygons, rows, columns, path, outline=None, overlays=None, lines=None, axis_labels=None,
              colorbar_label="", shared=False, empty_text="identically zero"):
    """A grid of maps on the fine cells: one row per label of `rows` {label: [(n_cells,) per column]}, one column per
    title of `columns`. Sequential colours (`Blues`) on a log scale, one scale per row (all the panels with `shared`).
    `polygons` (n_cells, 3, 2) the cells in (s1, s2); a NaN or non-positive cell is left blank; a panel without any
    positive value shows `empty_text`. `outline`: the triangle; `overlays`: polygons drawn dashed (the zones); `lines`:
    segments drawn dotted (the cuts). Returns False when there is nothing to draw."""
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LogNorm

    if not rows:
        return False
    polygons = np.asarray(polygons, float)
    columns = list(columns)
    values = {label: [np.asarray(v, float) for v in panels] for label, panels in rows.items()}

    def positive(arrays):
        every = np.concatenate([a[np.isfinite(a) & (a > 0)] for a in arrays]) if arrays else np.array([])
        return every

    every = positive([a for panels in values.values() for a in panels])
    n_rows = len(values)
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, axes = _grid(values, columns)
        for i, (label, panels) in enumerate(values.items()):
            mine = every if shared else positive(panels)
            norm = LogNorm(vmin=mine.min(), vmax=mine.max()) if mine.size and mine.max() > mine.min() else (
                LogNorm(vmin=mine.min() / 2, vmax=mine.min() * 2) if mine.size else None)
            image = None
            for k in range(len(columns)):
                ax = axes[i, k]
                v = panels[k] if k < len(panels) else np.full(len(polygons), np.nan)
                good = np.isfinite(v) & (v > 0)
                if good.any() and norm is not None:
                    image = PolyCollection(polygons[good], array=v[good], cmap="Blues", norm=norm, linewidths=0)
                    ax.add_collection(image)
                else:
                    ax.text(0.5, 0.5, empty_text, transform=ax.transAxes, ha="center", va="center", fontsize=7,
                            color=INK_2)
                ax.update_datalim(polygons.reshape(-1, 2))
                _outlines(ax, outline, overlays)
                _lines(ax, lines)
                _frame(ax, i, k, n_rows, label, columns[k], axis_labels)
            if image is not None:
                fig.colorbar(image, cax=axes[i, -1], label=colorbar_label)
            else:
                axes[i, -1].set_visible(False)
        _save(fig, path)
    return True


def ratio_maps(polygons, panels, path, outline=None, overlays=None, lines=None, axis_labels=None, colorbar_label=""):
    """Maps of a ratio around 1 on the fine cells, one panel per (title, ratio (n_cells,)) of `panels`: log10 of the ratio
    with the diverging `RdBu_r` centred at 0 (1 = white, blue below, red above), one symmetric scale for all the panels.
    Returns False when no panel has a finite positive ratio."""
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import TwoSlopeNorm

    polygons = np.asarray(polygons, float)
    logs = []
    for _, ratio in panels:
        r = np.asarray(ratio, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            logs.append(np.where(np.isfinite(r) & (r > 0), np.log10(np.where(r > 0, r, 1.0)), np.nan))
    finite = np.concatenate([v[np.isfinite(v)] for v in logs]) if logs else np.array([])
    if not finite.size:
        return False
    top = max(float(np.max(np.abs(finite))), 1e-3)
    norm = TwoSlopeNorm(vmin=-top, vcenter=0.0, vmax=top)
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, axes = _grid({"": None}, [t for t, _ in panels])
        image = None
        for k, ((title, _), v) in enumerate(zip(panels, logs)):
            ax = axes[0, k]
            good = np.isfinite(v)
            image = PolyCollection(polygons[good], array=v[good], cmap="RdBu_r", norm=norm, linewidths=0)
            ax.add_collection(image)
            ax.update_datalim(polygons.reshape(-1, 2))
            _outlines(ax, outline, overlays)
            _lines(ax, lines)
            _frame(ax, 0, k, 1, "", title, axis_labels)
        fig.colorbar(image, cax=axes[0, -1], label=colorbar_label)
        _save(fig, path)
    return True


def edge_curves(centres, groups, path, edge_labels, ylabel, ylog=True, references=None, colours=None):
    """A quantity against the distance from the nearest edge, one panel per edge (the points nearest to it), linear x
    axis. `groups` {label: [(n_edges, n_bins) per seed]}: thin line per seed, thick mean (geometric with `ylog`), one
    colour per group (`colours` {label: colour}, default the series colours); `references` {name: value} horizontal
    dashed lines (e.g. 1 for a density)."""
    centres = np.asarray(centres, float)
    n_edges = len(edge_labels)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, n_edges, figsize=(3.6 * n_edges + 1.2, 3.4), sharey=True, squeeze=False)
        for j, ax in enumerate(axes[0]):
            for g, (label, runs) in enumerate(groups.items()):
                colour = (colours or {}).get(label, SERIES[g % len(SERIES)])
                curves = np.array([np.asarray(r, float)[j] for r in runs])
                if len(curves) > 1:
                    for c in curves:
                        ax.plot(centres, c, color=colour, linewidth=0.5, alpha=0.4)
                mean = _mean(curves, ylog)
                ax.plot(centres, mean, color=colour, linewidth=1.8 if len(curves) > 1 else 1.2,
                        label=f"{label} ({len(curves)} seed{'s' if len(curves) > 1 else ''}"
                              f"{', mean' if len(curves) > 1 else ''})")
            for k, (name, value) in enumerate((references or {}).items()):
                ax.axhline(value, color=INK_2, linestyle=("--", ":", "-.")[k % 3], linewidth=0.9, label=name)
            if ylog:
                ax.set_yscale("log")
            ax.tick_params(labelsize=6)
            ax.set_title(f"nearest edge {edge_labels[j]}", loc="left", fontsize=9)
            ax.set_xlabel(f"distance from {edge_labels[j]}")
        axes[0, 0].set_ylabel(ylabel)
        handles, names = axes[0, 0].get_legend_handles_labels()
        if handles:
            axes[0, -1].legend(handles, names, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)
        _save(fig, path)
    return True


def cut_curves(cuts, groups, path, ylabel, ylog=True, references=None, shade=None, reference=None, xlabel="s1"):
    """A quantity along the cuts of the triangle, one panel per cut, linear x axis. `cuts` {cut name: (B,) block centres
    in s1}; `groups` {label: {cut name: [(B,) per seed]}}: thin line per seed, thick mean (geometric with `ylog`), one
    colour per group (`methods.group_styles`, the group `reference` dark grey, "real"/"imag" their part colours);
    `references` {name: value} horizontal lines; `shade` {cut name: [(x0, x1, label)]} grey spans (the zones crossed)."""
    from .components import shade_spans
    from .methods import group_styles
    styles = group_styles(list(groups), reference=reference)
    names = list(cuts)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names) + 1.2, 3.4), sharey=True, squeeze=False)
        for j, name in enumerate(names):
            ax = axes[0, j]
            shade_spans(ax, (shade or {}).get(name))
            x = np.asarray(cuts[name], float)
            for label, per_cut in groups.items():
                colour, style = styles[label]
                curves = np.array([np.asarray(c, float) for c in per_cut.get(name, [])])
                if not len(curves) or not len(x):
                    continue
                if len(curves) > 1:
                    for c in curves:
                        ax.plot(x, c, color=colour, linestyle=style, linewidth=0.5, alpha=0.4)
                mean = _mean(curves, ylog)
                ax.plot(x, mean, color=colour, linestyle=style, linewidth=1.8 if len(curves) > 1 else 1.2, alpha=0.95,
                        label=f"{label} ({len(curves)} seed{'s' if len(curves) > 1 else ''}"
                              f"{', mean' if len(curves) > 1 else ''})")
            for k, (ref, value) in enumerate((references or {}).items()):
                ax.axhline(value, color=INK_2, linestyle=("--", ":", "-.")[k % 3], linewidth=0.9, label=ref)
            if ylog:
                ax.set_yscale("log")
            ax.tick_params(labelsize=6)
            ax.set_title(name, loc="left", fontsize=9, pad=12)
            ax.set_xlabel(xlabel)
        axes[0, 0].set_ylabel(ylabel)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            axes[0, -1].legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)
        _save(fig, path)
    return True
