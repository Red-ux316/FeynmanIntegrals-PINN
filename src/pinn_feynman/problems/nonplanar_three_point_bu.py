"""Non-planar three-point, bottom-up name: the same problem as `nonplanar_three_point` (network on (m10, m11),
forcing term from the subtopologies), declared with `base` so that its configurations and queues follow the
`*_bu` naming of the other diagrams. Data and cache are those of the base."""

from ._registry import Problem

PROBLEM = Problem(
    name="nonplanar_three_point_bu",
    system="pinn_feynman.systems.nonplanar_three_point",
    reference="pinn_feynman.reference.nonplanar_three_point",
    base="nonplanar_three_point",
    unknowns=(0, 1),
    provider="pinn_feynman.providers.nonplanar_three_point:source",
)
