"""Equal-mass sunrise for s > 0, m^2 = 3: domains between the singular points 0, 3, 27 of (A.10), true values by
integration of the equation along the real axis from one AMFlow point per interval
(reference/sunrise_equal_mass_positive.py). Real below 27, complex above (s + i0): both parts can be trained; below 27
only the real part makes sense."""

from ._registry import Problem

PROBLEM = Problem(
    name="sunrise_equal_mass_positive",
    system="pinn_feynman.systems.sunrise_equal_mass",
    reference="pinn_feynman.reference.sunrise_equal_mass_positive",
)
