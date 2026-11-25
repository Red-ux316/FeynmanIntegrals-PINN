"""Assemble all the pieces of an experiment from a configuration."""

from dataclasses import dataclass

import torch

from ..balancing import build_balancer
from ..collocation import build_collocation_sampler, sampler_class
from ..config import torch_dtype
from ..data.build import dataset_specs
from ..data.cache import get_dataset
from ..data.domains import as_domain
from ..models import build_model
from ..physics import build_system, system_variables
from ..problems import available_problems, get_problem
from ..weighting import NoWeighting, build_point_weighting, point_weighting_class
from .optim import build_optimizer, build_scheduler
from .recorder import build_recorder
from .trainer import PointSet, Trainer

#: seed offsets, so that one configuration seed drives independent random streams
SEED_OFFSETS = {"model": 0, "collocation": 1, "balancer": 2}


@dataclass
class Experiment:
    config: dict
    part: str
    trainer: Trainer
    datasets: dict


def resolve_device(name):
    """'cpu', 'cuda' or 'auto' (cuda if available, else cpu)."""
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but this torch build sees no CUDA GPU "
                           f"(torch {torch.__version__})")
    if name not in ("cpu", "cuda"):
        raise ValueError(f"device must be 'cpu', 'cuda' or 'auto', got {name!r}")
    return name


def to_point_set(dataset, part, input_scale, dtype, columns=None):
    """Dataset in x = t/s -> PointSet in the network variable z = input_scale * x.

    `columns`: the master integrals the network learns (bottom-up); None = all of them.
    """
    x, y = dataset.tensors(part, dtype=dtype)
    if columns is not None:
        y = y[:, list(columns)]
    return PointSet(z=input_scale * x, y=y)


def load_datasets(config, names=("boundary", "validation")):
    specs = dataset_specs(config)
    return {name: get_dataset(specs[name]) for name in names}


def check_collocation(config):
    """A point weighting with one state per base point needs fixed base points."""
    weighting = point_weighting_class(config.get("point_weighting"))
    sampler = sampler_class(config["collocation"])
    if weighting.per_point_state and not sampler.fixed_points:
        raise ValueError(f"point weighting {config['point_weighting']['name']!r} keeps one state per base point: "
                         f"it needs fixed base points, not collocation {config['collocation']['name']!r}")


def check_variables(config):
    """Number of variables of the system, checked against the domain and the network input.

    The configuration is not changed: `model.in_features` (default 1) must be written in it, so that the saved
    `config.yaml` rebuilds the network (evaluation) and the massless configurations keep their keys.
    """
    name = config["physics"]["system"]
    data_problem = get_problem(name).data_problem() if name in available_problems() else name
    if config["data"]["problem"] != data_problem:
        raise ValueError(f"data.problem is {config['data']['problem']!r} but physics.system is {name!r}: "
                         f"the true values and the equations must be those of the same problem ({data_problem!r})")
    variables = system_variables(name)
    d = len(variables)
    domain = as_domain(config["data"]["domain"])
    if domain.dim != d:
        raise ValueError(f"system {name!r} has {d} variables {variables} but the domain {domain.label()} has "
                         f"dimension {domain.dim}")
    in_features = int(config["model"].get("in_features", 1))
    if in_features != d:
        raise ValueError(f"model.in_features = {in_features} but system {name!r} has {d} variables {variables}: "
                         f"write `in_features: {d}` in the model section")
    return d


def check_part(config, part):
    """The part must be one the problem can train: the imaginary part of a problem whose values are real (sunrise at
    s < 0) is identically zero and is refused here, before any data are read."""
    name = config["physics"]["system"]
    if name not in available_problems():
        return
    parts = get_problem(name).parts
    if part not in parts:
        raise ValueError(f"{name!r} trains only the parts {parts}: the {part!r} part of its true values vanishes "
                         f"identically in the domain")


def check_active_orders(config):
    """`model.active_orders` may leave out only orders the physics sets to zero: those not in the
    `NONZERO_ORDERS` of the system module. A system that declares none has every order non-zero, so the mask must
    keep all of them. Returns the order labels of the problem when the key is set, else None."""
    active = config["model"].get("active_orders")
    if active is None:
        return None
    from ..evaluation.predict import problem_shape
    from ..models.factory import active_positions
    from ..physics import SYSTEMS
    name = config["physics"]["system"]
    orders = problem_shape(name)[1]
    active_positions(active, orders)                       # labels among the orders, distinct, increasing
    nonzero = getattr(SYSTEMS[name], "NONZERO_ORDERS", None)
    required = orders if nonzero is None else nonzero
    missing = [o for o in required if o not in [int(a) for a in active]]
    if missing:
        raise ValueError(f"model.active_orders {list(active)} leaves out the orders {missing}, which {name!r} does not "
                         f"set to zero (NONZERO_ORDERS = {None if nonzero is None else tuple(nonzero)})")
    return orders


def check_provider_cost(config):
    """A provider of the knowns that is not cheap needs fixed collocation points and no population gradients: the
    coefficients (forcing included) are recomputed at every change of points."""
    problem = get_problem(config["physics"]["system"])
    if problem.provider is None or problem.build_provider().cheap:
        return
    collocation = config["collocation"]["name"]
    if not sampler_class(config["collocation"]).fixed_points:
        raise ValueError(f"collocation {collocation!r} changes the points during the training but the provider of "
                         f"{problem.name!r} is not cheap: use fixed base points (base_points, dirichlet, r3)")
    if (config["training"].get("diagnostics") or {}).get("population_gradient"):
        raise ValueError(f"training.diagnostics.population_gradient evaluates the physics loss on new points but the "
                         f"provider of {problem.name!r} is not cheap")


def check_multi_variable(config):
    """With more than one variable only the components written for any domain are allowed; the others stop here,
    before their one-dimensional code is built."""
    collocation = config["collocation"]["name"]
    if not getattr(sampler_class(config["collocation"]), "multi_variable", False):
        raise NotImplementedError(f"collocation {collocation!r} works on one variable only: with a multi-variable "
                                  "system use base_points, dirichlet or resample (multi-variable version: future work)")
    weighting = point_weighting_class(config.get("point_weighting"))
    if getattr(weighting, "multi_variable_refusal", None):
        raise NotImplementedError(weighting.multi_variable_refusal)
    if weighting is not NoWeighting and not getattr(weighting, "multi_variable", False):
        raise NotImplementedError(f"point weighting {config['point_weighting']['name']!r} works on one variable "
                                  "only (its gate is a distance on an interval; multi-variable version: future work)")
    if config["training"].get("diagnostics"):
        raise NotImplementedError("the training recorder (training.diagnostics) works on one variable only "
                                  "(multi-variable version: future work)")


def build_experiment(config, part, datasets=None, seed=None):
    """Trainer ready to `fit`, for the real or imaginary part (Boni's two trainings)."""
    if check_variables(config) > 1:
        check_multi_variable(config)
    check_collocation(config)
    check_part(config, part)
    check_provider_cost(config)
    orders = check_active_orders(config)                  # None without model.active_orders: build_model as before
    seed = config["training"]["seed"] if seed is None else seed
    dtype = torch_dtype(config["dtype"])
    scale = float(config["physics"]["input_scale"])
    datasets = load_datasets(config) if datasets is None else datasets
    columns = get_problem(config["physics"]["system"]).columns()
    boundary = to_point_set(datasets["boundary"], part, scale, dtype, columns)
    validation = to_point_set(datasets["validation"], part, scale, dtype, columns)
    n_masters, n_orders = boundary.y.shape[1:]

    model = build_model(config["model"], n_masters, n_orders, dtype=dtype, seed=seed + SEED_OFFSETS["model"],
                        orders=orders)
    system = build_system(config["physics"]["system"], input_scale=scale, part=part)
    domain = as_domain(config["data"]["domain"]).scaled(scale)     # network variable z = scale * v
    point_weighting = build_point_weighting(config.get("point_weighting"), config["collocation"].get("n_base"), domain,
                                            system=config["physics"]["system"], input_scale=scale)
    gate = point_weighting if getattr(point_weighting, "shared_gate", False) else None    # given to samplers with `needs_gate`
    sampler = build_collocation_sampler(config["collocation"], domain, seed=seed + SEED_OFFSETS["collocation"],
                                        dtype=dtype, gate=gate)
    balancer = build_balancer(config["balancing"], n_losses=2, seed=seed + SEED_OFFSETS["balancer"])
    optimizer = build_optimizer(config["training"]["optimizer"], model.parameters())
    scheduler = build_scheduler(config["training"]["scheduler"], optimizer)
    trainer = Trainer(model, system, boundary, validation, sampler, balancer, optimizer, scheduler,
                      log_every=config["training"].get("log_every", 0),
                      device=resolve_device(config["training"].get("device", "cpu")),
                      point_weighting=point_weighting,
                      residual_stats_points=config["training"].get("residual_stats_points", 1024),
                      validation_physics=config["training"].get("validation_physics", True),
                      recorder=build_recorder(config["training"].get("diagnostics"), config["data"]["domain"], scale,
                                              config["training"]["epochs"]))
    return Experiment(config=config, part=part, trainer=trainer, datasets=datasets)
