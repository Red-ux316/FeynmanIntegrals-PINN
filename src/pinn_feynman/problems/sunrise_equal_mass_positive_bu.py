"""Equal-mass sunrise for s > 0, bottom-up: the network learns the coupled block (I_11100, I_21100), the
tadpole I_11000 (real) comes in closed form; above 27 the imaginary part satisfies the homogeneous equation."""

from ._registry import Problem

PROBLEM = Problem(
    name="sunrise_equal_mass_positive_bu",
    system="pinn_feynman.systems.sunrise_equal_mass",
    reference="pinn_feynman.reference.sunrise_equal_mass_positive",
    base="sunrise_equal_mass_positive",
    unknowns=(1, 2),
    provider="pinn_feynman.providers.sunrise_equal_mass:tadpole",
)
