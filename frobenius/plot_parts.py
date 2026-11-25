"""Massless box near x = 0: the true M_3^(k) (k = -1, 0, 1) and dM_3^(k)/dx split into the analytic part (the Frobenius
form with ln(-x) = 0) and the singular part (the terms with powers of ln(-x)); real part on top, imaginary part below.
Constants fixed at x0 = -0.3, 100 coefficients of U~, 60 points on [-0.6, -0.002]. Linear axes; the grey band is the
domain of the thesis.

Writes figures/parts_x0.pdf and figures/parts_x0_derivative.pdf. Run from the repository root (a few minutes):
    python frobenius/plot_parts.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

import mbox_x0 as fr

OUT = Path(__file__).resolve().parent / "figures"
TRUE_C, ANA_C, SING_C = "#1a1a19", "#2a78d6", "#d03b3b"
LO, HI = -0.9, -0.1


def clean(values, part):
    v = np.array([complex(t).real if part == "re" else complex(t).imag for t in values])
    v[np.abs(v) < 1e-10] = 0.0          # rounding noise drawn as zero
    return v


def ylim_from(ax, arrays, factor=1.1):
    v = np.concatenate(arrays)
    v = v[np.isfinite(v)]
    if v.size == 0 or np.max(np.abs(v)) == 0:
        ax.set_ylim(-1, 1)
        return
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    span = hi - lo if hi > lo else abs(hi) + 1e-12
    ax.set_ylim(lo - (factor - 1) * span, hi + (factor - 1) * span)


def fig_x0(rows, deriv, name):
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.2), sharex=True)
    xs = [r[0] for r in rows]
    off = 3 if deriv else 0
    sym = r"dM_3^{(%s)}/dx" if deriv else r"M_3^{(%s)}"
    for j, k in enumerate(fr.ORDERS):
        for i, part in enumerate(("re", "im")):
            ax = axes[i, j]
            T = clean([r[1 + off][j] for r in rows], part)
            A = clean([r[2 + off][j] for r in rows], part)
            S = clean([r[3 + off][j] for r in rows], part)
            first = i == 0 and j == 0
            ax.axvspan(LO, HI, facecolor="0.88", edgecolor="0.55", lw=0.6, zorder=0,
                       label="domain of the thesis" if first else None)
            ax.plot(xs, T, color=TRUE_C, lw=2.0, alpha=0.85, label="true value" if first else None)
            ax.plot(xs, A, color=ANA_C, lw=1.4, ls="--", label="analytic part" if first else None)
            ax.plot(xs, S, color=SING_C, lw=1.4, label="singular part" if first else None)
            ax.axvline(0, color="0.4", lw=0.8, ls=":")
            ylim_from(ax, [T, A, S])
            ax.set_xlim(-0.62, 0.01)
            ax.set_ylabel(("Re " if part == "re" else "Im ") + "$" + sym % k + "$")
            if i == 1:
                ax.set_xlabel(r"$x=t/s$")
            if i == 0:
                ax.set_title(rf"order $\varepsilon^{{{k}}}$", fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=7, loc="best")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"written {OUT / name}")


def main():
    xs = [-mp.mpf("0.002") - (mp.mpf("0.6") - mp.mpf("0.002")) * j / 59 for j in range(60)]
    rows = fr.parts(xs)
    for i, k in enumerate(fr.ORDERS):                # analytic + singular = true, to working precision
        w = max(abs(r[2][i] + r[3][i] - r[1][i]) / max(1, abs(r[1][i])) for r in rows)
        wd = max(abs(r[5][i] + r[6][i] - r[4][i]) / max(1, abs(r[4][i])) for r in rows)
        print(f"order eps^{k}: max |analytic + singular - true| / max(1,|true|): M_3 {mp.nstr(w, 3)}, "
              f"dM_3/dx {mp.nstr(wd, 3)}")
    fig_x0(rows, False, "parts_x0.pdf")
    fig_x0(rows, True, "parts_x0_derivative.pdf")


if __name__ == "__main__":
    main()
