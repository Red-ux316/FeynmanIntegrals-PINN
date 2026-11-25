"""Constant weights for the loss terms (reference for comparison with ReLoBRaLo)."""

import torch


class FixedWeights:
    """Weighted total loss sum_i w_i L_i with fixed w_i (default: all 1)."""

    def __init__(self, n_losses, weights=None):
        self.n_losses = n_losses
        self.weights = torch.tensor(weights if weights is not None else [1.0] * n_losses, dtype=torch.float64)
        if self.weights.numel() != n_losses:
            raise ValueError(f"expected {n_losses} weights, got {self.weights.numel()}")

    def update(self, losses):
        return self.weights

    def state_dict(self):
        return {"weights": self.weights}

    def load_state_dict(self, state):
        self.weights = state["weights"]

    def __call__(self, losses):
        return sum(w.to(loss.device) * loss for w, loss in zip(self.weights, losses))
