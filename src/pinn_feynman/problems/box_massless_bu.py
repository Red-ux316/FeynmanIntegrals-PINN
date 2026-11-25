"""Massless box, bottom-up: the network learns the box M3, the bubbles M1 and M2 come in closed form."""

from ._registry import Problem

PROBLEM = Problem(
    name="box_massless_bu",
    system="pinn_feynman.systems.box_massless",
    reference="pinn_feynman.reference.box_massless",
    base="box_massless",
    unknowns=(2,),
    provider="pinn_feynman.providers.box_massless:bubbles",
)
