"""One-loop massless box (Boni sec. 6.2.1): one variable x = t/s, true values in closed form."""

from ._registry import Problem

PROBLEM = Problem(
    name="box_massless",
    system="pinn_feynman.systems.box_massless",
    reference="pinn_feynman.reference.box_massless",
    boni="pinn_feynman.evaluation.boni_reference:BONI_MASSLESS_BOX",
)
