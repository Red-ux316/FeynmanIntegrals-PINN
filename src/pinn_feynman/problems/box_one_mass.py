"""One-loop one-mass box (Boni sec. 6.2.2): two variables (s1, s2) at s0 = 1, true values by series in Nielsen polylogs."""

from ._registry import Problem

PROBLEM = Problem(
    name="box_one_mass",
    system="pinn_feynman.systems.box_one_mass",
    reference="pinn_feynman.reference.box_one_mass",
    boni="pinn_feynman.evaluation.boni_reference:BONI_ONE_MASS_BOX",
)
