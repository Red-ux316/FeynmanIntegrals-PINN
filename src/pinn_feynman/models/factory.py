"""Model construction from the `model` section of a configuration."""

import torch
from torch import nn

from .mlp import MLP
from .order_mask import mask_orders

MODELS = {"mlp": MLP}
ACTIVATIONS = {"gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU}


def _lookup(registry, name, kind):
    try:
        return registry[name]
    except KeyError:
        raise ValueError(f"unknown {kind} {name!r}; available: {sorted(registry)}") from None


def active_positions(active_orders, orders):
    """Positions on the order axis of the labels `active_orders` (a subset of `orders`, distinct and increasing)."""
    if orders is None:
        raise ValueError("model.active_orders needs the order labels of the problem (build_model(..., orders=...))")
    orders, active = [int(o) for o in orders], [int(a) for a in active_orders]
    unknown = [a for a in active if a not in orders]
    if unknown:
        raise ValueError(f"model.active_orders {active}: {unknown} not among the orders {orders}")
    if not active or len(set(active)) != len(active) or sorted(active) != active:
        raise ValueError(f"model.active_orders {active} must be non-empty, distinct and increasing")
    return [orders.index(a) for a in active]


def build_model(model_config, n_masters, n_orders, dtype=torch.float64, seed=None, orders=None):
    """Network for n_masters x n_orders outputs; `seed` fixes the initial weights.

    With `model_config["active_orders"]` (labels among `orders`, the order labels of the problem) the network computes
    only those orders and the others are exact zeros (models.order_mask.ActiveOrders). The full network is built
    first with the same random draws, then its last layer is sliced: the masked network starts from the same weights,
    and the global generator ends in the same state, as the unmasked one with the same seed.
    """
    if seed is not None:
        torch.manual_seed(seed)
    model_cls = _lookup(MODELS, model_config["name"], "model")
    activation = _lookup(ACTIVATIONS, model_config["activation"], "activation")
    model = model_cls(in_features=model_config.get("in_features", 1), hidden=list(model_config["hidden"]),
                      n_masters=n_masters, n_orders=n_orders, activation=activation)
    model = model.to(dtype)
    active_orders = model_config.get("active_orders")
    if active_orders is None:
        return model
    if orders is not None and len(orders) != n_orders:
        raise ValueError(f"{len(orders)} order labels given for {n_orders} orders")
    positions = active_positions(active_orders, orders)
    if len(positions) == n_orders:
        return model
    return mask_orders(model, positions)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
