"""Per-point weights of the physics loss: the interface, and `NoWeighting` (Boni's plain mean).

A point weighting receives, at every optimisation step, the base-point indices of the batch, their network inputs z
(B, d) and the detached residual norms (B,), and returns per-point weights w (B,) or None. The physics loss becomes
mean_i( w_i^2 * mean_j r_ij^2 ), i.e. <(w r)^2>; None means Boni's plain mean.

A class is built by `factory.build_point_weighting` through its class method `build(params, n_base, domain, system,
input_scale)`, which takes from these arguments what the class needs.
"""


class NoWeighting:
    """Boni's physics loss: plain mean over points (returns None)."""

    per_point_state = False

    @classmethod
    def build(cls, params, n_base=None, domain=None, system=None, input_scale=1.0):
        return cls(**params)

    def __call__(self, idx, z, residual_norm):
        return None

    def end_epoch(self):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass
