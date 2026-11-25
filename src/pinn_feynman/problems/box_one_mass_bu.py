"""One-mass box, bottom-up: the network learns the box M1, the three bubbles come in closed form."""

from ._registry import Problem

PROBLEM = Problem(
    name="box_one_mass_bu",
    system="pinn_feynman.systems.box_one_mass",
    reference="pinn_feynman.reference.box_one_mass",
    base="box_one_mass",
    unknowns=(0,),
    provider="pinn_feynman.providers.box_one_mass:bubbles",
)
