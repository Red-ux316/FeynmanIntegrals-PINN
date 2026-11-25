"""Figures of a tuning study (log scale: values span many decades)."""

import numpy as np
import optuna

from .style import INK_2, SERIES, STYLE, _save, plt


def tuning_history(study, path, reference=None):
    """Value of every trial (completed: dots; pruned: rings, last reported value) and the
    running best, on a log axis. `reference` = {label: value} drawn as horizontal lines."""
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED and t.intermediate_values]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        ax.scatter([t.number for t in done], [t.value for t in done], color=SERIES[0], s=30, label="completed", zorder=3)
        ax.scatter([t.number for t in pruned], [t.intermediate_values[max(t.intermediate_values)] for t in pruned],
                   facecolors="none", edgecolors=SERIES[1], s=30, linewidths=1.5, label="pruned (last value)", zorder=3)
        if done:
            order = sorted(done, key=lambda t: t.number)
            best = np.minimum.accumulate([t.value for t in order])
            ax.step([t.number for t in order], best, where="post", color=INK_2, linewidth=1.5, label="best so far")
        for k, (label, value) in enumerate((reference or {}).items()):
            ax.axhline(value, color=SERIES[2 + k], linewidth=1.5, linestyle="--", label=label)
        ax.set_yscale("log")
        ax.set_xlabel("trial")
        ax.set_ylabel("best validation loss of the trial")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        _save(fig, path)


def importance_bars(importances, path, title):
    names = list(importances)[::-1]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 0.5 * len(names) + 1.5))
        ax.barh(names, [importances[n] for n in names], color=SERIES[0])
        ax.set_xlabel("importance (PED-ANOVA)")
        ax.set_title(title, loc="left")
        _save(fig, path)


def _numeric(values):
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError):
        return None


def slices(trials, params, path, title):
    """Value of each trial against each parameter (completed: dots; pruned: rings)."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(params), figsize=(3.2 * len(params), 3.3), squeeze=False, sharey=True)
        for ax, name in zip(axes[0], params):
            for state, filled in (("COMPLETE", True), ("PRUNED", False)):
                sel = [t for t in trials if t["state"] == state and t["value"] is not None and name in t["params"]]
                raw = [t["params"][name] for t in sel]
                xs = _numeric(raw)
                if xs is None:
                    cats = sorted({str(v) for v in raw})
                    xs = [cats.index(str(v)) for v in raw]
                    ax.set_xticks(range(len(cats)))
                    ax.set_xticklabels(cats)
                ax.scatter(xs, [t["value"] for t in sel], s=28, color=SERIES[0] if filled else SERIES[1],
                           facecolors=SERIES[0] if filled else "none", linewidths=1.3)
            values = [t["params"][name] for t in trials if name in t["params"]]
            nums = _numeric(values)
            if nums and min(nums) > 0 and max(nums) / min(nums) >= 100:
                ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(name)
        axes[0][0].set_ylabel("trial value")
        fig.suptitle(title, x=0.01, ha="left", fontsize=9)
        _save(fig, path)


def parallel_coordinates(trials, params, path, title, top=None):
    """Each completed trial is a line through the normalised parameter axes, coloured by its value."""
    done = [t for t in trials if t["state"] == "COMPLETE" and t["value"] is not None and t["value"] > 0]
    if not done or len(params) < 2:
        return False
    columns = []
    for name in params:
        raw = [t["params"].get(name) for t in done]
        nums = _numeric(raw)
        if nums is None:
            cats = sorted({str(v) for v in raw})
            nums = [float(cats.index(str(v))) for v in raw]
        elif min(nums) > 0 and max(nums) / min(nums) >= 100:
            nums = list(np.log10(nums))
        lo, hi = min(nums), max(nums)
        columns.append([(v - lo) / (hi - lo) if hi > lo else 0.5 for v in nums])
    logv = np.log10([t["value"] for t in done])
    norm = (logv - logv.min()) / (logv.max() - logv.min() if logv.max() > logv.min() else 1)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(1.6 * len(params) + 2, 3.8))
        cmap = plt.get_cmap("viridis_r")
        for i in np.argsort(-norm):
            ax.plot(range(len(params)), [c[i] for c in columns], color=cmap(norm[i]), alpha=0.8, linewidth=1.4)
        ax.set_xticks(range(len(params)))
        ax.set_xticklabels(params, rotation=20, ha="right")
        ax.set_yticks([])
        ax.set_title(title + " (dark = low value)", loc="left")
        _save(fig, path)
    return True


def contour_pair(trials, pair, path, title):
    """Trial value on the plane of two parameters (dots coloured by value; numeric parameters only)."""
    a, b = pair
    done = [t for t in trials if t["state"] == "COMPLETE" and t["value"] and a in t["params"] and b in t["params"]]
    xs, ys = _numeric([t["params"][a] for t in done]), _numeric([t["params"][b] for t in done])
    if not done or xs is None or ys is None:
        return False
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5.4, 4))
        points = ax.scatter(xs, ys, c=np.log10([t["value"] for t in done]), cmap="viridis_r", s=50)
        fig.colorbar(points, ax=ax, label="log10 trial value")
        for name, setter, values in ((a, ax.set_xscale, xs), (b, ax.set_yscale, ys)):
            if min(values) > 0 and max(values) / min(values) >= 100:
                setter("log")
        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.set_title(title, loc="left")
        _save(fig, path)
    return True


def category_boxes(trials, name, path, title):
    """Distribution of the trial value for each choice of a categorical (or few-valued) parameter."""
    done = [t for t in trials if t["state"] == "COMPLETE" and t["value"] and name in t["params"]]
    choices = sorted({str(t["params"][name]) for t in done})
    if len(choices) < 2:
        return False
    data = [[t["value"] for t in done if str(t["params"][name]) == c] for c in choices]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(1.2 * len(choices) + 2.5, 3.4))
        ax.boxplot(data, tick_labels=choices, showfliers=True)
        for i, values in enumerate(data, start=1):
            ax.scatter([i] * len(values), values, color=SERIES[0], s=14, zorder=3)
        ax.set_yscale("log")
        ax.set_xlabel(name)
        ax.set_ylabel("trial value")
        ax.set_title(title, loc="left")
        _save(fig, path)
    return True
