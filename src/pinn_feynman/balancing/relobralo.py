"""ReLoBRaLo: Relative Loss Balancing with Random Lookback.

Bischof & Kraus, arXiv:2110.09813, eq. (11); Boni, thesis sec. 4.3, eqs. (4.9)-(4.14).

    lam_bal_i(t, t') = m * softmax_i( L_i(t) / (T L_i(t')) )
    lam_hist_i(t)    = rho * lam_i(t-1) + (1 - rho) * lam_bal_i(t, 0)      (paper)
    lam_i(t)         = alpha * lam_hist_i(t) + (1 - alpha) * lam_bal_i(t, t-1)

with rho ~ Bernoulli(expected_rho). Boni's eq. (4.13) has lam_bal_i(t, t-1) in place of
lam_i(t-1) ("boni_literal", kept for comparison).
"""

import torch

HISTORIES = ("paper", "boni_literal")


class ReLoBRaLo:
    """Adaptive weights for m loss terms; call with the list of losses at each step."""

    def __init__(self, n_losses, temperature=0.5, alpha=0.999, expected_rho=0.999,
                 history="paper", seed=None, scales=None, scale_epochs=None):
        if history not in HISTORIES:
            raise ValueError(f"history must be one of {HISTORIES}, got {history!r}")
        self.n_losses = n_losses
        self.temperature = temperature
        self.alpha = alpha
        self.expected_rho = expected_rho
        self.history = history
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)
        self.initial = None           # L_i(0)
        self.previous = None          # L_i(t-1)
        self.weights = None           # lam_i(t-1)
        self.last_rho = None
        self.scales = torch.ones(n_losses, dtype=torch.float64) if scales is None else             torch.tensor([float(c) for c in scales], dtype=torch.float64)
        if self.scales.numel() != n_losses or not bool((self.scales > 0).all()):
            raise ValueError(f"scales must be {n_losses} positive numbers, got {scales!r}")
        self.scaled = scales is not None and bool((self.scales != 1).any())
        # the scales act only in the epochs [first, last] (1-based, inclusive); None = always
        if scale_epochs is not None and (len(scale_epochs) != 2 or int(scale_epochs[0]) > int(scale_epochs[1])):
            raise ValueError(f"scale_epochs must be [first, last], got {scale_epochs!r}")
        self.scale_epochs = None if scale_epochs is None else (int(scale_epochs[0]), int(scale_epochs[1]))
        self.active = True

    def set_epoch(self, epoch):
        """Called by the trainer at the start of each epoch: switches the scales on/off (`scale_epochs`)."""
        self.active = self.scale_epochs is None or self.scale_epochs[0] <= int(epoch) <= self.scale_epochs[1]

    @property
    def _scaling(self):
        return self.scaled and self.active

    def _balanced(self, current, reference):
        """m * softmax(L(t) / (T L(t'))), eq. (4.12)."""
        tiny = torch.finfo(current.dtype).tiny
        return self.n_losses * torch.softmax(current / (self.temperature * reference.clamp_min(tiny)), dim=0)

    def _draw_rho(self):
        return float(torch.rand((), generator=self.generator) < self.expected_rho)

    def update(self, losses):
        """New weights (detached tensor of shape (m,)) from the current loss values."""
        current = torch.stack([loss.detach() for loss in losses]).reshape(-1)
        if current.numel() != self.n_losses:
            raise ValueError(f"expected {self.n_losses} losses, got {current.numel()}")
        if self.initial is None:
            self.initial, self.previous = current.clone(), current.clone()
            self.weights = torch.ones_like(current)
        bal_prev = self._balanced(current, self.previous)
        bal_init = self._balanced(current, self.initial)
        rho = self._draw_rho()
        memory = self.weights if self.history == "paper" else bal_prev
        hist = rho * memory + (1 - rho) * bal_init
        self.weights = self.alpha * hist + (1 - self.alpha) * bal_prev
        self.previous = current.clone()
        self.last_rho = rho
        return self.weights

    def state_dict(self):
        """Everything needed to continue exactly (checkpoint/resume)."""
        return {"initial": self.initial, "previous": self.previous, "weights": self.weights,
                "last_rho": self.last_rho, "generator": self.generator.get_state()}

    def load_state_dict(self, state):
        self.initial, self.previous, self.weights = state["initial"], state["previous"], state["weights"]
        self.last_rho = state["last_rho"]
        self.generator.set_state(state["generator"])

    def coefficients(self):
        """lam_i c_i of the last update: the factor of each term in the total (the weights when unscaled)."""
        return self.weights if not self._scaling else self.weights * self.scales.to(self.weights)

    def __call__(self, losses):
        """Weighted total loss sum_i lam_i c_i L_i, with the weights treated as constants."""
        weights = self.update(losses)
        if not self._scaling:                                # without scales: the plain weighted sum
            return sum(w * loss for w, loss in zip(weights, losses))
        return sum(w * c.to(loss) * loss for w, c, loss in zip(weights, self.scales, losses))
