"""Massless box near x = 0: difference between the Frobenius form and the exact box.

At eps = 0.3 the local form M = third row of H(x) x^Lambda d is evaluated with U~ truncated to its first N coefficients
(N = 1, 2, 5, 10, 20, 100) and compared with the exact M (closed form, 50 digits) on -1e-1 >= x >= -1e-9. The constants d
are those of the full series (100 coefficients), fixed at x = -0.1. Log axes; the plotted quantity is |M_N - M| / |M|.

Writes figures/series_error.pdf. Run from the repository root:
    python frobenius/plot_series_error.py [--dps 50]
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

import mbox_x0 as fr

OUT = Path(__file__).resolve().parent / "figures" / "series_error.pdf"
EPS = mp.mpf("0.3")
TRUNC = (1, 2, 5, 10, 20, 100)
COLORS = ("#d03b3b", "#eb6834", "#c9a227", "#3c9a5f", "#2a78d6", "#1a1a19")


def main():
    mp.mp.dps = 50
    if "--dps" in sys.argv:                         # working precision; the figure uses 50 digits
        mp.mp.dps = int(sys.argv[sys.argv.index("--dps") + 1])
    Lam, U = fr.frobenius_series(EPS, max(TRUNC))
    d = fr.fix_constants(Lam, U, mp.mpf("-0.1"), EPS)
    xs = [-mp.mpf(10) ** (-a) for a in np.linspace(1, 9, 81)]
    exact = [fr.exact_M(x, EPS)[2] for x in xs]
    err = {n: [float(abs(fr.local_m3(Lam, U[:n], d, x)[0] - m) / abs(m)) for x, m in zip(xs, exact)] for n in TRUNC}
    for n in TRUNC:
        print(f"N = {n:3d}: rel. diff. at x = -1e-1: {err[n][0]:.2e}, -1e-5: {err[n][40]:.2e}, -1e-9: {err[n][-1]:.2e}")

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax_x = [float(-x) for x in xs]
    for n, col in zip(TRUNC, COLORS):
        y = np.array(err[n])
        y[y == 0] = np.nan                # x = -0.1, where d is fixed: zero by construction
        ax.loglog(ax_x, y, color=col, lw=1.4, label=f"$N={n}$")
    ax.set_xlabel(r"$-x$")
    ax.set_ylabel(r"$|M_N-M|\,/\,|M|$")
    ax.invert_xaxis()
    ax.grid(True, which="major", lw=0.3, alpha=0.5)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT)
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
