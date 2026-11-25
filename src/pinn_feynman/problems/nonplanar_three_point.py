"""Non-planar two-loop three-point function (Boni sec. 6.3.3), the two masters of the top sector (m10, m11), x = -s/m^2.

The top sector is the whole basis of the system: the nine subtopologies enter only through the closed-form forcing
term N11^(4) (vMT arXiv:1701.05905 eq. 4.2), supplied by a forcing provider. Without it the equation would be the
homogeneous one, which the masters do not satisfy: this problem is therefore declared with `unknowns` and `provider`
already (bottom-up formulation), and its data are its own. Complex masters for x < 0: both parts are trained.
"""

from ._registry import Problem

PROBLEM = Problem(
    name="nonplanar_three_point",
    system="pinn_feynman.systems.nonplanar_three_point",
    reference="pinn_feynman.reference.nonplanar_three_point",
    unknowns=(0, 1),
    provider="pinn_feynman.providers.nonplanar_three_point:source",
    boni="pinn_feynman.evaluation.boni_reference:BONI_NONPLANAR",
)
