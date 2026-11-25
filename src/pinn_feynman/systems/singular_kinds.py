"""Kind of every singular point (one variable) and singular line (two variables) of the systems: one place for it.

The kind is decided on the FULL system of the family (all its master integrals: 11 for the non-planar three-point,
3 for the sunrise and the massless box, 4 for the one-mass box), not on the bottom-up block the network learns:

- ``physical``: a pole of the full matrix where the master integrals are singular (not analytic: logarithm, root,
  divergent derivative), in at least one master;
- ``spurious``: a pole of the full matrix where all the master integrals are analytic (singular for the equation
  only).

A point can be regular for the bottom-up block and still be physical: the singularity then reaches the unknowns
through the known term (non-planar x = -4). The positions come from the system modules (`SINGULAR_POINTS`,
`FORCING_BRANCH_POINTS`, `DENOMINATOR_FACTORS`); this module only gives their kind and a short label. Used by
`visualization.singularities` (dashed lines on the figures); not by `weighting.distance`, which computes the poles of
A_UU itself. Keys: the name of the
problem of the data (the base of a bottom-up problem). Every point and line of a system module must have a kind.
"""

#: problem -> {point: (note, kind)}
KINDS = {
    # full system M1, M2 (bubbles), M3 (box): x = 0 (t = 0) physical (log of the t bubble and of the box); x = -1
    # (u = 0) spurious: the bubbles do not depend on u and the planar box is regular there (Henn p. 11; Boni p. 101)
    "box_massless": {0: ("t=0", "physical"), -1: ("u=0", "spurious")},
    # full system (A.10), tadpole^2 + two sunrise masters: s = 27 = 9 m^2 three-particle threshold; at s = 0 and at the
    # pseudo-threshold s = 3 = m^2 the matrix has poles but all three masters are analytic (F > 0 for s < 27;
    # Boni calls s = 0 physical)
    "sunrise_equal_mass": {0: ("", "spurious"), 3: ("pseudo-threshold", "spurious"), 27: ("threshold", "physical")},
    # s > 0: same system and same points; the domains lie between them
    "sunrise_equal_mass_positive": {0: ("", "spurious"), 3: ("pseudo-threshold", "spurious"),
                                    27: ("threshold", "physical")},
    # full 11-master system: poles at 0, -4, 4, 16. x = 0 physical (massless cut, Boni p. 124). x = -4 physical:
    # two-massive-particle threshold s = 4 m^2, pole of the sector-22 sunrise (m3, m4, root (x + 4)^(1/2)); regular for
    # the top block B + eps D, it reaches m10, m11 through the known term, and m11'' diverges there as (x + 4)^(-1/2)
    # (Boni calls it spurious). x = 16 spurious: pole of the top block only, the
    # physical solution is regular there (vMT 4.39-4.40). The pseudo-threshold x = 4 (pole of sector 15, Boni (6.59))
    # is not a branch point of the known term and is not drawn
    "nonplanar_three_point": {0: ("", "physical"), -4: ("threshold", "physical"), 16: ("", "spurious")},
}

#: problem -> {denominator factor (str, s0 = 1): (label, kind)}; full system: box M1 and the bubbles (functions of s, t,
#: m^2 only, singular only on t = 0 and m^2 = 0): t = 0 and m^2 = 0 physical (logarithms); u = 0
#: spurious (box regular; Boni calls it physical); m^2 = s and m^2 = t singular for the coefficients only
LINE_KINDS = {
    "box_one_mass": {"s1": ("t=0", "physical"), "s2": ("m^2=0", "physical"), "s1 - s2 + 1": ("u=0", "spurious"),
                     "1 - s2": ("m^2=s", "spurious"), "s1 - s2": ("m^2=t", "spurious")},
}


def point_kind(problem, point):
    """'physical' or 'spurious' for a singular point of a one-variable problem; KeyError if it has no kind."""
    try:
        return KINDS[problem][point][1]
    except KeyError:
        raise KeyError(f"{problem}: singular point {point:g} has no kind in systems/singular_kinds.py") from None


def line_kind(problem, factor):
    """'physical' or 'spurious' for a singular line (its denominator factor as a string, s0 = 1); KeyError if none."""
    try:
        return LINE_KINDS[problem][factor][1]
    except KeyError:
        raise KeyError(f"{problem}: singular line {factor} = 0 has no kind in systems/singular_kinds.py") from None
