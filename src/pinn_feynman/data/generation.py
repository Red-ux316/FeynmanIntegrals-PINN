"""Exact reference values on a grid of points, computed in parallel.

Each problem registers a generator `xs -> complex array (len(xs), N_masters, N_orders)`
and the parameters that identify its output (they enter the cache key). The points xs have
the shape of the problem's domain: (N,) for one variable, (N, d) for d variables.
How the points are split into chunks and processes is a per-problem option (COMPUTE_OPTIONS):
it never changes the values.
"""

from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np

from .. import progress as console
from ..problems import ProblemMapping, available_problems

#: problem name -> (generator, parameters that identify its output), read from `pinn_feynman.problems`
GENERATORS = ProblemMapping(lambda problem: (problem.generator(), problem.cache_params()),
                            present=lambda problem: problem.base is None)   # a bottom-up problem has no data of its own

#: problem -> how compute_reference splits the points: chunk size and maximum number of processes (None = as
#: requested); declared by the `reference` module of each problem (COMPUTE_OPTIONS)
COMPUTE_OPTIONS = ProblemMapping(lambda problem: problem.compute_options())
DEFAULT_COMPUTE_OPTIONS = {"chunk_size": 10, "max_workers": None}


def generator_info(problem):
    """(generator, identifying parameters) registered for `problem`."""
    try:
        return GENERATORS[problem]
    except KeyError:
        raise ValueError(f"unknown problem {problem!r}; available: {available_problems()}") from None


def _evaluate_chunk(problem, xs):
    generator, _ = generator_info(problem)
    return generator(xs)


def compute_reference(problem, xs, workers=1, chunk_size=None, progress=False):
    """Reference values at the points xs; `workers > 1` spreads chunks over processes.

    The result does not depend on `workers` or `chunk_size`: each point is computed
    independently. `chunk_size` None and the cap on `workers` come from COMPUTE_OPTIONS.
    `progress`: False, True, or the label of the progress bar.
    """
    options = COMPUTE_OPTIONS.get(problem, DEFAULT_COMPUTE_OPTIONS)
    chunk_size = options["chunk_size"] if chunk_size is None else chunk_size
    if options["max_workers"] is not None:
        workers = min(workers, options["max_workers"])
    xs = np.asarray(xs, dtype=float)
    chunks = [xs[i:i + chunk_size] for i in range(0, len(xs), chunk_size)]
    task = partial(_evaluate_chunk, problem)
    results = []
    label = progress if isinstance(progress, str) else "points"
    with console.bar(len(xs), label, unit="pt", disable=not progress) as bar:
        if workers <= 1:
            for chunk in chunks:
                results.append(task(chunk))
                bar.update(len(chunk))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for chunk, values in zip(chunks, pool.map(task, chunks)):
                    results.append(values)
                    bar.update(len(chunk))
    return np.concatenate(results, axis=0)

