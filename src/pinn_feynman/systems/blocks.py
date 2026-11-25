"""Blocks of a system dM/dz_k = A_k M: dependency graph, bottom-up order, split into unknowns U and knowns K.

Bottom-up formulation: the network learns only the unknowns U, the knowns K (sub-topologies, supplied
by a provider) enter as a forcing term. With M = (M_U, M_K) this needs A_KU = 0 (triangularity) and gives
    dM_U/dz_k = A_UU,k M_U + A_UK,k M_K.

The tools here work on sympy matrices, with A_k = sum_j eps^j A_{j,k}, and never use floating point to decide
whether an entry vanishes: `is_zero` is `sp.cancel` plus a cross-check at exact rational points.

* `epsilon_orders(matrix)`: the coefficients A_j of a matrix polynomial in eps (error if eps is in a denominator);
* `partition(orders_by_variable)`: strongly connected components ("blocks": they depend on the basis) in bottom-up order;
* `split(orders_by_variable, unknowns)`: checks A_KU = 0 and extracts A_UU and A_UK.

`orders_by_variable` is a list with one entry per variable k, each a tuple (A_0, ..., A_p) of sympy Matrices.
"""

import random
from dataclasses import dataclass

import sympy as sp

EPS = sp.Symbol("epsilon")


def epsilon_orders(matrix, eps=EPS):
    """(A_0, ..., A_p) with matrix = sum_j eps^j A_j, at least two orders (p >= 1).

    Every entry must be a polynomial in eps with coefficients free of eps; eps in a denominator (a basis whose
    cascade in eps is not triangular) is refused: a change of basis is needed first.
    """
    entries = []
    degree = 1
    for entry in matrix:
        numerator, denominator = sp.fraction(sp.cancel(sp.sympify(entry)))
        if denominator.has(eps):
            raise ValueError(f"eps in the denominator of entry {entry}: change basis")
        coefficients = sp.Poly(sp.expand(numerator), eps).all_coeffs()[::-1] if numerator.has(eps) else [numerator]
        entries.append([sp.cancel(c / denominator) for c in coefficients])
        degree = max(degree, len(coefficients) - 1)
    orders = []
    for j in range(degree + 1):
        orders.append(sp.Matrix(matrix.rows, matrix.cols,
                                [e[j] if j < len(e) else sp.Integer(0) for e in entries]))
    return tuple(orders)


def is_zero(entry, n_checks=3, seed=0):
    """True if the rational function `entry` vanishes identically: `sp.cancel` first, then exact rational points.

    Entries that are not rational in their symbols are an error; a disagreement between the two tests is a
    RuntimeError (never silently resolved).
    """
    entry = sp.sympify(entry)
    symbolic = sp.cancel(entry) == 0
    symbols = sorted(entry.free_symbols, key=str)
    if not symbols:
        return symbolic
    rng = random.Random(seed)
    values_at_zero = 0
    done = 0
    attempts = 0
    while done < n_checks and attempts < 50 * n_checks:
        attempts += 1
        point = {s: sp.Rational(rng.randint(-40, 40), rng.randint(1, 13)) for s in symbols}
        try:
            value = entry.subs(point)
        except ZeroDivisionError:
            continue
        if value.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
            continue
        if not value.is_number or not value.is_rational:
            raise ValueError(f"entry {entry} is not a rational function of {symbols}")
        done += 1
        values_at_zero += value == 0
    if done == 0:
        raise ValueError(f"no admissible rational point found for {entry}")
    if symbolic and values_at_zero != done:
        raise RuntimeError(f"sp.cancel says {entry} vanishes but a rational point disagrees")
    if not symbolic and values_at_zero == done and done >= n_checks:
        raise RuntimeError(f"sp.cancel says {entry} does not vanish but it vanishes at {done} rational points")
    return symbolic


@dataclass(frozen=True)
class Partition:
    """Blocks of a system, in bottom-up order (a block depends only on earlier blocks)."""

    n: int
    blocks: tuple[tuple[int, ...], ...]
    block_of: tuple[int, ...]
    depends_on: tuple[frozenset, ...]

    def top_blocks(self):
        """Blocks on which no other block depends (the recommended default for U)."""
        used = set().union(*self.depends_on) if self.depends_on else set()
        return tuple(b for b in range(len(self.blocks)) if b not in used)

    def closure_below(self, indices):
        """Indices of all blocks that the blocks of `indices` depend on, directly or not (without `indices`)."""
        seen = set()
        stack = [self.block_of[i] for i in indices]
        while stack:
            for b in self.depends_on[stack.pop()]:
                if b not in seen:
                    seen.add(b)
                    stack.append(b)
        inside = {self.block_of[i] for i in indices}
        return tuple(sorted(i for b in seen - inside for i in self.blocks[b]))


def _nonzero_pattern(orders_by_variable):
    """Set of (i, j), i != j, with A_{j',k}[i, j] != 0 for some order j' and variable k (exact test)."""
    edges = set()
    for orders in orders_by_variable:
        for matrix in orders:
            for i in range(matrix.rows):
                for j in range(matrix.cols):
                    if i != j and (i, j) not in edges and not is_zero(matrix[i, j]):
                        edges.add((i, j))
    return edges


def _strongly_connected(n, successors):
    """Tarjan's strongly connected components, iterative; list of sorted tuples."""
    index, low, on_stack, stack, components = {}, {}, set(), [], []
    counter = 0
    for root in range(n):
        if root in index:
            continue
        work = [(root, iter(successors[root]))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, it = work[-1]
            advanced = False
            for nxt in it:
                if nxt not in index:
                    index[nxt] = low[nxt] = counter
                    counter += 1
                    stack.append(nxt)
                    on_stack.add(nxt)
                    work.append((nxt, iter(successors[nxt])))
                    advanced = True
                    break
                if nxt in on_stack:
                    low[node] = min(low[node], index[nxt])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                components.append(tuple(sorted(component)))
    return components


def partition(orders_by_variable):
    """Blocks of the system and their bottom-up order.

    Arc i -> j (M_i depends on M_j) when A[i, j] != 0 for some order in eps and some variable: a basis with
    A_0 = 0 would otherwise give an empty graph. Blocks are the strongly connected components, ordered by Kahn's
    algorithm with ties broken by the smallest index (deterministic): a block depends only on earlier ones.
    """
    n = orders_by_variable[0][0].rows
    edges = _nonzero_pattern(orders_by_variable)
    successors = [sorted(j for (i, j) in edges if i == row) for row in range(n)]
    components = _strongly_connected(n, successors)
    block_of_index = {i: b for b, comp in enumerate(components) for i in comp}
    deps = [set() for _ in components]
    for i, j in edges:
        if block_of_index[i] != block_of_index[j]:
            deps[block_of_index[i]].add(block_of_index[j])
    remaining = set(range(len(components)))
    order = []
    while remaining:
        ready = [b for b in remaining if not (deps[b] & remaining)]
        if not ready:                                     # cannot happen for components of a condensation
            raise RuntimeError("cycle among blocks")
        b = min(ready, key=lambda c: components[c][0])
        order.append(b)
        remaining.discard(b)
    position = {b: p for p, b in enumerate(order)}
    blocks = tuple(components[b] for b in order)
    return Partition(
        n=n,
        blocks=blocks,
        block_of=tuple(position[block_of_index[i]] for i in range(n)),
        depends_on=tuple(frozenset(position[d] for d in deps[b]) for b in order),
    )


@dataclass(frozen=True)
class Split:
    """Result of `split`: A_UU and A_UK for each variable and each order in eps (same number of orders)."""

    unknowns: tuple[int, ...]
    knowns: tuple[int, ...]
    uu: list
    uk: list | None
    partition: Partition

    def report(self):
        lines = [f"blocks (bottom-up): {[list(b) for b in self.partition.blocks]}",
                 f"unknowns U = {list(self.unknowns)}, knowns K = {list(self.knowns)}",
                 f"orders in eps: {len(self.uu[0])} (degree {len(self.uu[0]) - 1})"]
        for b, deps in enumerate(self.partition.depends_on):
            lines.append(f"block {b} {list(self.partition.blocks[b])} depends on blocks {sorted(deps)}")
        return "\n".join(lines)


def split(orders_by_variable, unknowns):
    """Split into unknowns U (indices, increasing) and knowns K (the complement); checks A_KU = 0 exactly.

    Raises ValueError listing the first non-vanishing entries of A_KU (variable, order in eps, row, column).
    With K empty, `uk` is None.
    """
    n = orders_by_variable[0][0].rows
    unknowns = tuple(unknowns)
    if list(unknowns) != sorted(set(unknowns)) or not all(0 <= u < n for u in unknowns) or not unknowns:
        raise ValueError(f"unknowns {unknowns} must be distinct, increasing indices in 0..{n - 1}")
    knowns = tuple(i for i in range(n) if i not in unknowns)
    degree = max(len(orders) for orders in orders_by_variable)
    padded = [tuple(orders) + (sp.zeros(n, n),) * (degree - len(orders)) for orders in orders_by_variable]
    bad = [(k, j, i, u, padded[k][j][i, u]) for k in range(len(padded)) for j in range(degree)
           for i in knowns for u in unknowns if not is_zero(padded[k][j][i, u])]
    if bad:
        raise ValueError("A_KU != 0: the unknowns are not closed upward; (variable, order, row, column, entry) = "
                         f"{bad[:5]}")
    uu = [tuple(m.extract(list(unknowns), list(unknowns)) for m in orders) for orders in padded]
    uk = [tuple(m.extract(list(unknowns), list(knowns)) for m in orders) for orders in padded] if knowns else None
    return Split(unknowns, knowns, uu, uk, partition(padded))
