"""On-disk cache of reference datasets (.npz files in paths.data_dir())."""

import hashlib
import json
import os
import uuid

import numpy as np

from ..paths import data_dir
from .dataset import DatasetSpec, ReferenceDataset
from .generation import compute_reference, generator_info
from .interpolation import barycentric_interpolate
from .sampling import sample


def _metadata(spec):
    _, params = generator_info(spec.problem)
    return json.dumps({"spec": spec.as_dict(), "generator": params}, sort_keys=True)


def cache_key(spec):
    """Hash of the specification and of the generator parameters (12 hex digits)."""
    return hashlib.sha1(_metadata(spec).encode()).hexdigest()[:12]


def cache_path(spec, directory=None):
    """File of the dataset in the cache directory (default: paths.data_dir())."""
    directory = data_dir() if directory is None else directory
    return directory / f"{spec.problem}_{spec.sampler}_n{spec.n}_seed{spec.seed}_{spec.method}_{cache_key(spec)}.npz"


def save_dataset(dataset, path):
    """Write to a uniquely named temporary file, then rename: readers (also other sessions on a
    shared folder) never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    with open(tmp, "wb") as stream:
        np.savez(stream, x=dataset.x, values=dataset.values, meta=np.array(_metadata(dataset.spec)))
    os.replace(tmp, path)


def load_dataset(path):
    with np.load(path) as f:
        spec = json.loads(str(f["meta"]))["spec"]
        spec = DatasetSpec(**spec)            # the domain is converted from its configuration form
        return ReferenceDataset(spec=spec, x=f["x"], values=f["values"])


def nodes_spec(spec):
    """Exact dataset on the Chebyshev nodes behind a 'chebyshev' dataset."""
    return DatasetSpec(problem=spec.problem, sampler="chebyshev", domain=spec.domain, n=spec.n_nodes)


def interpolate_reference(nodes_dataset, xs):
    """Values at xs interpolated from an exact dataset on Chebyshev nodes."""
    return barycentric_interpolate(nodes_dataset.x, nodes_dataset.values, xs)


def build_dataset(spec, workers=1, directory=None, progress=False):
    """Sample the points and compute their reference values.

    Method 'exact' evaluates every point; method 'chebyshev' interpolates from the
    (cached) exact values at the Chebyshev nodes. `progress`: False, True, or the label of the bar.
    """
    x = sample(spec.sampler, spec.domain, spec.n, spec.seed)
    if spec.method == "chebyshev":
        label = f"{progress} nodes" if isinstance(progress, str) else progress
        nodes = get_dataset(nodes_spec(spec), workers=workers, directory=directory, progress=label)
        values = interpolate_reference(nodes, x)
    else:
        values = compute_reference(spec.problem, x, workers=workers, progress=progress)
    return ReferenceDataset(spec=spec, x=x, values=values)


def get_dataset(spec, workers=1, directory=None, progress=False):
    """Load the dataset from the cache, generating and saving it if missing."""
    path = cache_path(spec, directory)
    if path.exists():
        return load_dataset(path)
    dataset = build_dataset(spec, workers=workers, directory=directory, progress=progress)
    save_dataset(dataset, path)
    return dataset
