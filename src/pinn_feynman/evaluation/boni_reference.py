"""Results reported in Boni's thesis: massless box (sec. 6.2.1.1, printed pp. 102-103), one-mass box
(sec. 6.2.2.1, printed pp. 107-110) and non-planar three-point function (sec. 6.3.3.1, printed pp. 124-126).

Values are as stated in the text; the thesis does not say which average "mean" is, and
the histograms (Fig. 6.3-6.4) use logarithmic bins.
"""

from ..problems import ProblemMapping

BONI_MASSLESS_BOX = {
    "epochs": 400,
    "train_loss": 1e-7,                 # "L_train ~ 1e-7", stable from ~300 epochs
    "val_loss": 1e-8,                   # "L_val ~ 1e-8"
    "real": {
        "abs_error_mean": 9e-5,         # "they peak close to their mean value at 9e-5"
        "abs_error_range": (1e-8, 1e-4),
        "rel_error_mean": 8.5e-5,       # "the mean value is of the same order ... (8.5e-5)"
        "worst_rel_order": -1,          # "the eps^-1 order (j = 1 in the plot) has the worst result"
    },
    "imag": {
        "abs_error_mean": 4e-5,         # "a mean absolute error of 4e-5"
    },
}

#: One-mass box (sec. 6.2.2.1, printed pp. 107-110). 300 epochs, L_train ~ 1e-7,
#: L_val ~ 1e-8 (Fig. 6.9). Fig. 6.10: E_abs^mean = 1e-4, E_rel^mean = 3e-4 (all orders in one panel); Fig. 6.11:
#: the imaginary part is wider, mean absolute error still 1e-4. That 1e-4 and 3e-4 are of the real part is a
#: reading (the text does not say); the thesis does not say which average "mean" is, nor whether the components
#: that vanish identically are counted (our reports show the median next to the mean).
BONI_ONE_MASS_BOX = {
    "epochs": 300,
    "train_loss": 1e-7,
    "val_loss": 1e-8,
    "real": {
        "abs_error_mean": 1e-4,         # Fig. 6.10, left
        "rel_error_mean": 3e-4,         # Fig. 6.10, right
    },
    "imag": {
        "abs_error_mean": 1e-4,         # Fig. 6.11
    },
}

#: Non-planar three-point function (sec. 6.3.3.1, printed pp. 124-126; checked against the thesis text). 350 epochs, L_train ~ 1e-5 (stops improving early), L_val ~ 1e-7 (Fig. 6.21). Fig. 6.22: E_abs^mean =
#: 3e-4 with maximum O(1e-3); E_rel^mean = 4e-4 for the only non-zero component (the eps^4 one); Fig. 6.23: the
#: imaginary part has mean absolute error 4e-4. Values as stated in the text.
BONI_NONPLANAR = {
    "epochs": 350,
    "train_loss": 1e-5,
    "val_loss": 1e-7,
    "real": {
        "abs_error_mean": 3e-4,         # Fig. 6.22, left
        "rel_error_mean": 4e-4,         # Fig. 6.22, right: eps^4 component only
    },
    "imag": {
        "abs_error_mean": 4e-4,         # Fig. 6.23
    },
}

#: problem -> results reported by Boni (the problems that declare them in `pinn_feynman.problems`)
BONI = ProblemMapping(lambda problem: problem.boni_results(), present=lambda problem: problem.boni is not None)


def boni_part(problem, part):
    """Boni's numbers for one part of a problem ({} if none)."""
    return BONI.get(problem, {}).get(part, {})
