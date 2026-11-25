"""Style shared by every figure of the project: colours, matplotlib settings, saving, warnings.

Colours: categorical slots in fixed order (reference palette of the dataviz guidelines); text in neutral inks, never
in series colours; one y axis per panel; static PNG on a light background.
"""

import warnings
from contextlib import contextmanager

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # slots 1-5, fixed order
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"

STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "font.size": 9, "legend.frameon": False,
}


def order_label(order):
    return rf"$\varepsilon^{{{order}}}$"


_collector = None
_domain = None                           # (lo, hi) of the domain the figures along x are about, or None


def set_domain(lo=None, hi=None):
    """The domain written in the box of the figures along x (`domain_box`); `set_domain()` clears it."""
    global _domain
    _domain = None if lo is None else (float(lo), float(hi))


def domain_box(ax):
    """A small box "domain: [a, b]" in the lower right corner, once per figure (the first axes asked), when a domain
    is set. The figures along x call it next to `mark`."""
    if _domain is None or getattr(ax.figure, "_domain_boxed", False):
        return
    ax.figure._domain_boxed = True
    ax.text(0.99, 0.02, f"domain: [{_domain[0]:g}, {_domain[1]:g}]", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.5, color=INK_2, bbox={"boxstyle": "round,pad=0.25", "fc": SURFACE, "ec": GRID, "lw": 0.6}, zorder=5)


def _save(fig, path):
    if _collector is not None:           # `collect()` active: the caller gets the figures instead of files
        _collector.append(fig)
        return
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


save = _save


@contextmanager
def collect():
    """Figures saved inside the block are returned in the list instead of being written."""
    global _collector
    previous, _collector = _collector, []
    try:
        yield _collector
    finally:
        _collector = previous


@contextmanager
def quiet():
    """Silence two known, harmless messages while a report is built: matplotlib's "Tight layout not
    applied" (legends outside the axes; the figure is saved correctly) and Optuna's "experimental"
    notice for PED-ANOVA, the importance measure chosen for the tuning reports."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Tight layout not applied")
        warnings.filterwarnings("ignore", message="PedAnovaImportanceEvaluator is experimental")
        yield
