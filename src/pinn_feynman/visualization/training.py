"""Figures along the epochs: losses, training dynamics, adaptive-method columns."""

from importlib import import_module
from importlib.util import find_spec

import numpy as np

from .style import SERIES, STYLE, _save, plt


def loss_curves(history_rows, path, title):
    """Fig. 6.2: training and validation loss per epoch."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 3.6))
        epochs = [r["epoch"] for r in history_rows]
        ax.plot(epochs, [r["train_loss"] for r in history_rows], color=SERIES[0], label="train loss (weighted)")
        ax.plot(epochs, [r["val_loss"] for r in history_rows], color=SERIES[1], label="validation loss (5.16)")
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(title, loc="left")
        ax.legend()
        _save(fig, path)


def training_dynamics(history_rows, path, title):
    """ReLoBRaLo weights and learning rate per epoch; with the residual statistics in the history (VD-01),
    a third panel with the skewness and kurtosis of the residual norms (one y axis per panel)."""
    epochs = [r["epoch"] for r in history_rows]
    stats = [r for r in history_rows if r.get("residual_skewness") not in (None, "")]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3 if stats else 2, figsize=(15 if stats else 10, 3.4))
        ax1, ax2 = axes[0], axes[1]
        if "weight_0" in history_rows[0]:
            ax1.plot(epochs, [r["weight_0"] for r in history_rows], color=SERIES[0], label=r"$\lambda_{data}$")
            ax1.plot(epochs, [r["weight_1"] for r in history_rows], color=SERIES[1], label=r"$\lambda_{DE}$")
            ax1.legend()
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("ReLoBRaLo weight (end of epoch)")
        ax2.plot(epochs, [r["lr"] for r in history_rows], color=SERIES[0])
        ax2.set_yscale("log")
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("learning rate")
        if stats:
            ax3 = axes[2]
            ax3.plot([r["epoch"] for r in stats], [float(r["residual_skewness"]) for r in stats], color=SERIES[0],
                     label="skewness")
            ax3.plot([r["epoch"] for r in stats], [float(r["residual_kurtosis"]) for r in stats], color=SERIES[1],
                     label="kurtosis")
            ax3.set_yscale("symlog")
            ax3.set_xlabel("epoch")
            ax3.set_ylabel("residual statistics (App. H)")
            ax3.legend()
        fig.suptitle(title, x=0.01, ha="left", fontsize=9)
        _save(fig, path)


def compare_val_loss(histories, path):
    """Validation loss per epoch, one line per run."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        for k, (label, rows) in enumerate(histories):
            ax.plot([r["epoch"] for r in rows], [r["val_loss"] for r in rows], color=SERIES[k % len(SERIES)],
                    label=label, linewidth=1.5)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation loss (5.16)")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        _save(fig, path)


ADAPTIVE_COLUMNS = (("r3_retained", "points retained by R3"),)
_EXTRA = f"{__package__}.extra"
if find_spec(_EXTRA) is not None:                 # the columns of further methods, in their order
    ADAPTIVE_COLUMNS = import_module(_EXTRA).ADAPTIVE_COLUMNS


def adaptive_columns(history, path, title):
    """One panel per adaptive-method column present in the history; False if there is none."""
    present = [(key, label) for key, label in ADAPTIVE_COLUMNS if key in history[0]]
    if not present:
        return False
    epochs = [r["epoch"] for r in history]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(present), figsize=(3.6 * len(present), 3.2), squeeze=False)
        for ax, (key, label) in zip(axes[0], present):
            ax.plot(epochs, [r.get(key, np.nan) for r in history], color=SERIES[0])
            ax.set_xlabel("epoch")
            ax.set_ylabel(label)
        fig.suptitle(title, x=0.01, ha="left", fontsize=9)
        _save(fig, path)
    return True


# --- the losses and the ReLoBRaLo coefficients of one or more runs ------------------------------------------------------
# `groups` = {label: [history rows of seed 0, of seed 1, ...]}; one run = {label: [rows]}. With several seeds each seed
# is a thin line and the median over the seeds a thick one; the colour is the method's (`methods.group_styles`).

def _column(rows, key):
    """(epochs, values) of a history column; None if the column is absent or empty."""
    pairs = [(r["epoch"], r[key]) for r in rows if r.get(key) not in (None, "") and np.isfinite(r[key])]
    if not pairs:
        return None
    e, v = zip(*pairs)
    return np.asarray(e), np.asarray(v, dtype=float)


def _styles(labels):
    from .methods import group_styles
    return group_styles(list(labels))


def moving_average(epochs, values, window, log=True):
    """Centred moving average over `window` epochs (shorter at the ends); on a log axis the mean of log10, i.e. the
    geometric mean, so that a spike up does not lift the curve. window <= 1: the values unchanged."""
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) < 2:
        return epochs, values
    v = np.log10(np.where(values > 0, values, np.nan)) if log else values
    half = window // 2
    out = np.array([np.nanmean(v[max(0, i - half):i + half + 1]) for i in range(len(v))])
    return epochs, 10.0 ** out if log else out


MAX_GROUPS_WITH_SEEDS = 2            # up to this many groups in a panel the seeds are drawn too (p.34 of the 30/10 request)


def _draw(ax, groups, value, styles=None, log=True, smooth=1):
    """One family of curves per group: thin lines per seed and the median over seeds. `value(rows)` -> (epochs,
    values) or None; `smooth` > 1: each curve is first averaged over that many neighbouring epochs
    (`moving_average`). Returns True if something was drawn."""
    styles = styles or _styles(groups)
    drawn = False
    for label, runs in groups.items():
        colour, style = styles[label]
        curves = [c for c in (value(rows) for rows in runs) if c is not None]
        curves = [moving_average(e, v, smooth, log) for e, v in curves]
        if not curves:
            continue
        drawn = True
        if len(curves) == 1:
            ax.plot(*curves[0], color=colour, linestyle=style, label=label)
            continue
        if len(groups) <= MAX_GROUPS_WITH_SEEDS:        # with more groups only the medians, or the plot is unreadable
            for e, v in curves:
                ax.plot(e, v, color=colour, linestyle=style, linewidth=0.7, alpha=0.4)
        n = min(len(e) for e, _ in curves)
        ax.plot(curves[0][0][:n], np.median([v[:n] for _, v in curves], axis=0), color=colour, linestyle=style,
                linewidth=2, label=f"{label} (median of {len(curves)})")
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("epoch")
    return drawn


def _single(groups):
    return len(groups) == 1 and len(next(iter(groups.values()))) == 1


def _legend(ax):
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)


def _pde_column(rows):
    """L_PDE of the training: the unweighted one when it was logged (methods with point weights weight the other)."""
    return _column(rows, "physics_loss_unweighted") or _column(rows, "physics_loss")


TRAINING_LOSSES = (("L_data (boundary points)", lambda rows: _column(rows, "data_loss")),
                   ("L_PDE (collocation points)", _pde_column),
                   ("weighted sum (ReLoBRaLo)", lambda rows: _column(rows, "train_loss")))


def loss_training(groups, path, title, smooth=1):
    """The three losses of the training per epoch: L_data at the boundary points, L_PDE at the collocation
    points, their sum weighted by ReLoBRaLo. One run: one panel with the three curves; several runs: one panel per
    loss, one curve family per group. `smooth`: moving average over that many epochs (several runs only)."""
    with plt.rc_context(STYLE):
        if _single(groups):
            rows = next(iter(groups.values()))[0]
            fig, ax = plt.subplots(figsize=(6.4, 3.8))
            for k, (name, value) in enumerate(TRAINING_LOSSES):
                curve = value(rows)
                if curve is not None:
                    ax.plot(*curve, color=SERIES[k], label=name)
            ax.set_yscale("log")
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            _legend(ax)
            ax.set_title(title, loc="left")
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
            for ax, (name, value) in zip(axes, TRAINING_LOSSES):
                _draw(ax, groups, value, smooth=smooth)
                ax.set_ylabel(name)
            _legend(axes[0])
            fig.suptitle(title, x=0.01, ha="left", fontsize=9)
        _save(fig, path)
    return True


def loss_validation(groups, path, title, markers=None, smooth=1):
    """The two losses on the validation points per epoch: L_data (network against the exact values) and L_PDE
    (`val_physics_loss`, in the runs that have the column). `markers` = {label: {"best": value, "final": value}}:
    the validation L_PDE of the best and final network computed afterwards, drawn at the last epoch for the runs
    without the column."""
    styles = _styles(groups)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        _draw(axes[0], groups, lambda rows: _column(rows, "val_loss"), styles, smooth=smooth)
        axes[0].set_ylabel("L_data, validation (eq. 5.16)")
        drawn = _draw(axes[1], groups, lambda rows: _column(rows, "val_physics_loss"), styles, smooth=smooth)
        for label, values in (markers or {}).items():
            runs = groups.get(label) or []
            last = max((r["epoch"] for rows in runs for r in rows), default=0)
            for kind, marker in (("best", "o"), ("final", "s")):
                if values.get(kind) is not None:
                    axes[1].scatter([last], [values[kind]], color=styles[label][0], marker=marker, zorder=3,
                                    label=f"{label}: {kind} network (computed afterwards)")
                    drawn = True
        axes[1].set_yscale("log")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("L_PDE, validation")
        if not drawn:
            axes[1].text(0.5, 0.5, "no validation L_PDE in these runs",
                         transform=axes[1].transAxes, ha="center", va="center", fontsize=8)
        for ax in axes:
            _legend(ax)
        fig.suptitle(title, x=0.01, ha="left", fontsize=9)
        _save(fig, path)
    return True


def _ratio(rows, top, bottom):
    a, b = _column(rows, top), _column(rows, bottom)
    if a is None or b is None:
        return None
    common, ia, ib = np.intersect1d(a[0], b[0], return_indices=True)
    return common, a[1][ia] / b[1][ib]


def _pde_share(rows):
    """Share of the L_PDE term in the weighted sum, lambda_PDE L_PDE / (lambda_data L_data + lambda_PDE L_PDE).
    Approximate: the weights are those at the end of the epoch, the losses means over the epoch."""
    cols = [_column(rows, k) for k in ("weight_0", "weight_1", "data_loss", "physics_loss")]
    if any(c is None for c in cols):
        return None
    n = min(len(c[0]) for c in cols)
    w0, w1, ld, lp = (c[1][:n] for c in cols)
    return cols[0][0][:n], w1 * lp / (w0 * ld + w1 * lp)


def relobralo(groups, path, title, smooth=1):
    """The ReLoBRaLo coefficients per epoch: lambda_data (weight_0), lambda_PDE (weight_1), their ratio and the
    share of the L_PDE term in the weighted sum. False if the runs have no weights (another balancing)."""
    if not any(_column(rows, "weight_0") is not None for runs in groups.values() for rows in runs):
        return False
    panels = (("lambda_data (weight of L_data)", lambda rows: _column(rows, "weight_0"), False),
              ("lambda_PDE (weight of L_PDE)", lambda rows: _column(rows, "weight_1"), False),
              ("lambda_PDE / lambda_data", lambda rows: _ratio(rows, "weight_1", "weight_0"), True),
              ("share of L_PDE in the sum (approx.)", _pde_share, False))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
        for ax, (name, value, log) in zip(axes, panels):
            _draw(ax, groups, value, log=log, smooth=smooth)
            ax.set_ylabel(name)
        axes[3].set_ylim(0, 1)
        _legend(axes[0])
        fig.suptitle(title + "  (weights at the end of each epoch; the share uses the epoch-mean losses)",
                     x=0.01, ha="left", fontsize=9)
        _save(fig, path)
    return True


def learning_rate(groups, path, title):
    """The learning rate per epoch (with the exponential scheduler every run has the same path: a check)."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.4, 2.6))
        _draw(ax, groups, lambda rows: _column(rows, "lr"))
        ax.set_ylabel("learning rate")
        _legend(ax)
        ax.set_title(title, loc="left")
        _save(fig, path)
    return True


def r3_retained(groups, sizes, path, title):
    """Share of the R3 population kept at each refresh (`r3_retained` / population size). `sizes` = {label:
    population size}. False if no run has the column."""
    present = {label: runs for label, runs in groups.items()
               if label in sizes and any(_column(rows, "r3_retained") is not None for rows in runs)}
    if not present:
        return False
    styles = _styles(groups)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        for label, runs in present.items():
            size = sizes[label]

            def share(rows, size=size):
                c = _column(rows, "r3_retained")
                return None if c is None else (c[0], c[1] / size)
            _draw(ax, {label: runs}, share, styles={label: styles[label]}, log=False)
        ax.set_ylim(0, 1)
        ax.set_ylabel("share of the population retained")
        _legend(ax)
        ax.set_title(title, loc="left")
        _save(fig, path)
    return True


def pde_sampling_bias(groups, path, title):
    """L_PDE on the training points over L_PDE on the (uniform) validation points, per epoch: about 1 when the
    training points are uniform, above 1 when the sampling chases the residual (R3). Only runs with
    `val_physics_loss`; False if there is none."""
    def value(rows):
        return _ratio(rows, "physics_loss", "val_physics_loss")
    if not any(value(rows) is not None for runs in groups.values() for rows in runs):
        return False
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        _draw(ax, groups, value)
        ax.axhline(1.0, color="#888888", linewidth=0.8, linestyle="--")
        ax.set_ylabel("L_PDE training / L_PDE validation")
        _legend(ax)
        ax.set_title(title, loc="left")
        _save(fig, path)
    return True


def gradients(groups, path, title, smooth=1):
    """How hard each loss term pushes the network, from the recorder's gradient columns of history.csv
    (median over the sampled steps of each epoch): norm of the gradient of the weighted L_data and of the weighted
    L_PDE, their ratio, the ratio of the steps Adam would take with each term alone, and the cosine between the two
    gradients. False if no run has the columns (`training.diagnostics.gradients` off)."""
    if not any(_column(rows, "grad_norm_data") is not None for runs in groups.values() for rows in runs):
        return False
    panels = (("|grad| of the weighted L_data", lambda rows: _column(rows, "grad_norm_data"), True),
              ("|grad| of the weighted L_PDE", lambda rows: _column(rows, "grad_norm_pde"), True),
              ("|grad L_data| / |grad L_PDE|", lambda rows: _ratio(rows, "grad_norm_data", "grad_norm_pde"), True),
              ("Adam step from L_data / from L_PDE", lambda rows: _ratio(rows, "adam_step_data", "adam_step_pde"),
               True),
              ("cosine between the two gradients", lambda rows: _column(rows, "grad_cosine"), False))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 5, figsize=(21, 3.8))
        for ax, (name, value, log) in zip(axes, panels):
            _draw(ax, groups, value, log=log, smooth=smooth)
            ax.set_ylabel(name)
        axes[4].axhline(0, color="#888888", linewidth=0.6)
        axes[4].set_ylim(-1, 1)
        _legend(axes[0])
        fig.suptitle(title + "  (median over the steps sampled in each epoch, one every 64; the Adam step uses the "
                     "second moment before the step)", x=0.01, ha="left", fontsize=9)
        _save(fig, path)
    return True
