# pinn_feynman.systems

Systems of differential equations $d\vec M/dx=A(x,\varepsilon)\,\vec M$ for the Master Integrals. Each module is a diagram.

Every module of this folder exposes `VARIABLES` (the names of the variables, `("x",)` for the massless box, `("s1", "s2")` for the one-mass box, `("s",)` for the sunrise), `connection_matrix_orders(variable)` and `N_MASTERS`, so that `physics.TorchSystem.from_module` can build the torch system of any of them.

## `box_massless.py` — one-loop massless box

Boni's basis (thesis, eq. 6.26): $\vec M=(t^\varepsilon I_{0,1,0,1},\ t^\varepsilon I_{1,0,1,0},\ t^{2+\varepsilon}I_{1,1,1,1})$, box integrals in the $(s,t)$ channels. Variable $x=t/s$; the matrix is the thesis's (A.2), linear in $\varepsilon$.

| Name | Type | What it does |
|---|---|---|
| `EPS`, `X` | sympy symbols | $\varepsilon$ and $x$ |
| `C_X_BONI` | `int` | $c_x=10$: Boni trains on $x_B=10\,t/s$ |
| `BONI_DOMAIN` | `tuple` | Boni's domain in $x=t/s$: $(-0.9,-0.1)$, i.e. $x_B\in(-9,-1)$ |
| `SINGULAR_POINTS` | `tuple` | $x=0$ (physical, $t=0$) and $x=-1$ (spurious, $u=0$) |
| `connection_matrix()` | → `sympy.Matrix` | $A(x,\varepsilon)$ of (A.2) |
| `connection_matrix_orders()` | → `(A0, A1)` | $A=A^{(0)}+\varepsilon A^{(1)}$; error if $A$ is not linear in $\varepsilon$ |
| `numeric_orders(modules)` | → `(f0, f1)` | numeric versions of $A^{(0)}(x)$, $A^{(1)}(x)$ (`sympy.lambdify`) |
| `to_boni_x(x)`, `from_boni_x(xB)` | → `float` | conversion between $x=t/s$ and $x_B=10\,t/s$ |

With the Laurent coefficients $\vec M=\sum_l\varepsilon^l\vec M^{(l)}$ the system becomes $\partial_x\vec M^{(l)}=A^{(0)}\vec M^{(l)}+A^{(1)}\vec M^{(l-1)}$ (Boni eq. 6.5): this is the PINN's physics loss.

## `analysis.py` — local analysis at the singular points (Frobenius theory)

`python -m pinn_feynman.systems.analysis box_massless` prints a table with, for each singular point, including $x=\infty$ via $t=1/x$:
- the **pole order**: Fuchsian if $\le1$ (Henn eq. 4.5–4.6);
- the **residue matrix** $A_0=\lim_{x\to x_0}(x-x_0)A$;
- the **exponents**, i.e. the eigenvalues of $A_0$ with multiplicity;
- the **resonances**, pairs of exponents differing by a nonzero integer;
- the **Jordan blocks** of $A_0$.

| Name | What it does |
|---|---|
| `local_matrix(a, x, point)` | shifts the point to $t=0$; for infinity uses $-A(1/t)/t^2$ |
| `pole_order(a, t)` | maximum pole order at $t=0$ |
| `residue(a, t)` | $\lim_{t\to0} t\,a$ |
| `resonances(exponents)`, `jordan_blocks(m)` | resonances and Jordan structure |
| `analyse_point(a, x, point)` → `PointAnalysis` | everything for one point |
| `analyse(name)`, `report(analyses)` | all points of a one-variable system of this package (`SYSTEM_MODULES`: every module with `X`, `EPS`, `connection_matrix`, found by looking), Markdown table |

**Result for the massless box**, matching the hand calculations on the thesis's matrix (A.2):

| point | residue | exponents | resonances | Jordan |
|---|---|---|---|---|
| $x=0$ | $\mathrm{diag}(0,\varepsilon,1)$ | $0,\varepsilon,1$ | $1-0$ | no blocks |
| $x=-1$ | last row $(2-4\varepsilon,\,2-4\varepsilon,\,\varepsilon)$ | $0$ (double), $\varepsilon$ | none | no blocks |
| $x=\infty$ | — | — | — | — |

**At $x=\infty$ Boni's basis is not Fuchsian**: $A_{32}=2(2\varepsilon-1)x/(1+x)$ tends to a constant, and with $t=1/x$ gives a double pole. This is the type of spurious singularity at infinity described by Henn (eq. 4.7–4.8), removable with a change of basis. The PINN's domain ($s\neq0$) is unaffected.

## `box_massless_canonical.py` — the same box in canonical basis

CMZ basis (2.9), $\tilde F=(st\,I_{1111},\ s\,I_{1110},\ t\,I_{1101})=T\,\vec M$, with the matrix $T$ in `gauge_from_boni()`. System in *dlog* form:
$$\partial_x\tilde F=\varepsilon\Big[\frac{a}{x}+\frac{b}{1+x}\Big]\tilde F .$$
Used to **compare the local structure** of the two bases. The test redoes the transformation $(TA+\partial_xT)T^{-1}$ and obtains exactly this matrix.

| point | Boni's basis | canonical basis |
|---|---|---|
| $x=0$ | exponents $0,\varepsilon,1$, **resonance** | exponents $-\varepsilon,-\varepsilon,0$, **Jordan block** of size 2 |
| $x=-1$ | $0,0,\varepsilon$ | $0,0,\varepsilon$ |
| $x=\infty$ | **not Fuchsian** | Fuchsian, exponents $0,0,\varepsilon$, one Jordan block |

The exponents at $x=0$ differ by an integer between the two bases ($1$ vs. $0$): basis transformations can shift them by integers, as the theory of Fuchsian systems states (cf. Henn arXiv:1412.2296 §3.3 and §4). The **type** of logarithm changes: from resonance to Jordan, i.e. from the "hard" mechanism to the "easy" one.

Not a problem (no file in `problems/`), only in `analysis.SYSTEM_MODULES`: it has no true values or training.

## `box_one_mass.py` — one-loop one-mass box (two variables)

The differential system of the four master integrals in the variables $(s_1,s_2)$ with $s_0=1$, in the order (box, bubble$_t$, bubble$_{m^2}$, bubble$_s$). Derived from BDK eq. (4.33); it agrees with the matrices of Boni's App. A.4–A.5 up to typographical corrections (two entries of A.4, one of A.5, the missing fourth row of A.4). The matrices are real, linear in $\varepsilon$, and their denominators are exactly the singular set of the thesis (6.37). It is **not** registered in `analysis.SYSTEM_MODULES` (that analysis works in one variable).

| Name | Type | What it does |
|---|---|---|
| `EPS`, `S0`, `S1`, `S2` | sympy symbols | $\varepsilon$ and the three invariants |
| `INVARIANTS`, `VARIABLES`, `S0_VALUE`, `N_MASTERS`, `MASTER_NAMES` | constants | `("s0","s1","s2")`, `("s1","s2")`, 1, 4 and the names of the integrals |
| `DENOMINATOR_FACTORS` | `tuple` | the factors of the denominators (thesis 6.37) |
| `connection_matrix(variable)` | → `sympy.Matrix` | $A_{s_0}$, $A_{s_1}$ or $A_{s_2}$; $\sum_k s_kA_{s_k}=0$ |
| `connection_matrix_orders(variable, s0=1)` | → `(A0, A1)` | $A=A^{(0)}+\varepsilon A^{(1)}$; `s0=None` keeps $s_0$ symbolic; error if $A$ is not linear in $\varepsilon$ |
| `numeric_orders(modules="numpy")` | → `{"s1": (f0, f1), "s2": (f0, f1)}` | callables of $(s_1,s_2)$ with $s_0=1$ |
| `DEFAULT_MARGIN`, `MARGIN_TOLERANCE` | constants | 0.1 and $10^{-12}$ (the vertices of the triangle with margin are inside up to rounding) |
| `edge_distance(points)` | → array | $\min(-s_1,\ s_2,\ 1+s_1-s_2)$: positive inside, equal to the margin of the smallest triangle $T_{
m margin}$ containing the point |
| `in_domain(points, margin=0.0)` | → bool array | open region (`margin=0`) or closed triangle $T_{\rm margin}$ |
| `vertices(margin=0.1)` | → `(3, 2)` array | the three vertices of the triangle with that margin (the boundary points) |

$\max|A|$ grows like the inverse square of the margin: 128 at 0.1, about $2\cdot10^{4}$ at 0.01, $2\cdot10^{6}$ at 0.001. The sides of the region are $t=0$, $m^2=0$, $u=0$; the last one is where the thesis puts a "physical" singularity and the box is analytic.

**Checks**: integrability $\partial_2A_1-\partial_1A_2+[A_1,A_2]=0$ (symbolic), $\sum_ks_kA_{s_k}=0$, linearity in $\varepsilon$, denominators equal to (6.37), structure of the rows, `numeric_orders` equal to the symbolic matrices and finite inside the region, error for an unknown variable, domain and vertices. The matrices have also been checked against the closed formulas at finite $\varepsilon$.

## `sunrise_equal_mass.py` — two-loop equal-mass sunrise, $m^2=3$

Boni's basis (thesis eq. 6.50) $\vec M=(I_{11000},\ I_{11100},\ I_{21100})$ (product of two tadpoles, sunrise, sunrise with a squared propagator), literal propagators $q^2-m^2$, measure $d^dk/(i\pi^{d/2})$ per loop, no $e^{\varepsilon\gamma_E}$. Variable $s=p^2$; the matrix is the thesis's (A.10), **quadratic in $\varepsilon$**. Row 1 vanishes (constant tadpole), row 2 is the homogeneity relation.

| Name | Type | What it does |
|---|---|---|
| `EPS`, `S` (= `X`) | sympy symbols | $\varepsilon$ and $s$; `X` makes the module visible to `analysis` |
| `MASS_SQUARED`, `N_MASTERS`, `MASTER_NAMES`, `VARIABLES` | constants | 3, 3, `("tadpole_squared", "sunrise", "sunrise_dotted")`, `("s",)` |
| `SINGULAR_POINTS` | `tuple` | $s=0$, the pseudo-threshold $m^2=3$, the threshold $9m^2=27$ ($s=9$, cited by thesis eq. 6.52, is not a singularity of (A.10)) |
| `TITLE`, `VARIABLE_LABELS`, `FIGURE_LABELS` | strings | figure titles and axis labels (`FIGURE_LABELS` replaces the $x_B$ labels in the one-variable figures of `evaluation.evaluate`) |
| `connection_matrix()` | → `sympy.Matrix` | $A(s,\varepsilon)$ of (A.10) |
| `connection_matrix_orders(variable="s")` | → `(A0, A1, A2)` | $A=A^{(0)}+\varepsilon A^{(1)}+\varepsilon^2A^{(2)}$; error if not quadratic |
| `numeric_orders(modules)` | → `(f0, f1, f2)` | numeric versions (`sympy.lambdify`) |

**Local analysis** (`python -m pinn_feynman.systems.analysis sunrise_equal_mass`): Fuchsian at $0,3,27,\infty$; exponents of the $2\times2$ block $\{0,\varepsilon-1\}$ at $s=0$, $\{0,1-2\varepsilon\}$ at $3$ and $27$, $\{\varepsilon,2\varepsilon-1\}$ (plus $0$ of the tadpole) at infinity; no resonances, no Jordan blocks.

**Blocks:** $\{0\}$ below $\{1,2\}$: the bottom-up problem learns $U=\{1,2\}$. The shared code handles degree 2 in $\varepsilon$ (`TorchSystem.coefficient_orders`, `cascade_residual(..., *higher)`, the window $l_{\min}-2\ldots l_{\max}$ of `BottomUpSystem`).

## `blocks.py` — blocks, bottom-up order, split into unknowns and knowns

Works on sympy matrices $A_k=\sum_j\varepsilon^jA_{j,k}$ (a list per variable of tuples $(A_0,\dots,A_p)$), never decides a zero in floating point.

| Name | What it does |
|---|---|
| `epsilon_orders(matrix)` | coefficients $A_j$ (at least two); error if $\varepsilon$ is in a denominator (basis whose cascade is not triangular: change basis) |
| `is_zero(entry)` | `sp.cancel` plus a cross-check at exact rational points; `RuntimeError` if they disagree |
| `partition(orders_by_variable)` → `Partition` | arcs $i\to j$ if $A[i,j]
e0$ for some order and variable; blocks = strongly connected components (they depend on the basis), in bottom-up order; `top_blocks()`, `closure_below(indices)` |
| `split(orders_by_variable, unknowns)` → `Split` | checks $A_{KU}=0$ for every order and variable (error listing the first entries) and extracts $A_{UU}$, $A_{UK}$; `report()` |

Massless box: blocks $\{0\},\{1\},\{2\}$, $U=\{2\}$; one-mass: bubbles first, box last, $U=\{0\}$.

## `singular_kinds.py` — physical or spurious, for every singular point and line

`KINDS` ({problem: {point: (note, kind)}}) and `LINE_KINDS` ({problem: {denominator factor at $s_0=1$: (label, kind)}}), keyed by the problem of the data (the base of a bottom-up problem); `point_kind(problem, point)`, `line_kind(problem, factor)`. **physical**: the integral is singular there; **spurious**: singular for the equation only. Registry for the dashed lines of the figures (`visualization.singularities`); the distance weight DW (`weighting.distance`) does not use it: it computes the poles of $A_{UU}$ from the system.

## `nonplanar_three_point.py` — non-planar two-loop three-point function, top sector

The two masters of the top sector (Boni eq. 6.57 = von Manteuffel–Tancredi arXiv:1701.05905 eq. 2.3, "vMT"), $m_{10}=\varepsilon^4s^2I_{1111110}$, $m_{11}=\varepsilon^4s^2(s+16m^2)/(2(1+2\varepsilon))\,I_{1211110}$, variable $x=-s/m^2$. The system is $dm/dx=[B(x)+\varepsilon D(x)]\,m+N$ (vMT 2.7–2.8, Boni 6.58 with A.12): the module holds the $2\times2$ matrix only; the inhomogeneous term $N=(0,N_{11})$, which carries the nine subtopologies, is a **forcing provider** (`providers/nonplanar_three_point.py`). The matrix is vMT's: Boni's (A.12) differs in the $\varepsilon$ term of the entry (2,1) ($\varepsilon/(2x)$ against $2\varepsilon/x$), which acts only on the orders $\ge\varepsilon^5$ since $m^{(n)}=0$ for $n<4$.

| Name | Type | What it does |
|---|---|---|
| `EPS`, `X` | sympy symbols | $\varepsilon$ and $x$ |
| `N_MASTERS`, `MASTER_NAMES`, `VARIABLES` | constants | 2, `("m10", "m11")`, `("x",)` |
| `SINGULAR_POINTS`, `FORCING_BRANCH_POINTS` | `tuple` | $x=0,16$ (matrix); $x=-4$ (forcing: square-root threshold; $x=4$ is not a branch point of $N_{11}^{(4)}$, and at $x=0$ the forcing has a logarithm at a pole of the matrix) |
| `DEFAULT_DOMAIN` | `tuple` | $[-20,0.1]$, Boni (6.60) read literally |
| `NONZERO_ORDERS` | `tuple` | (4,): the only order the physics does not set to zero; `evaluation.cell_analysis.kappa_u` uses it |
| `SHOWN_ORDERS`, `TITLE`, `VARIABLE_LABELS`, `FIGURE_LABELS` | constants | orders drawn in the solution curves of `evaluation.evaluate` ($\varepsilon^4$ only), figure titles and labels |
| `homogeneous_matrices()`, `connection_matrix()` | → sympy | $(B,D)$ and $A=B+\varepsilon D$ |
| `connection_matrix_orders(variable="x")`, `numeric_orders(modules)` | → `(B, D)` | as the other systems |

**Local analysis** (`python -m pinn_feynman.systems.analysis nonplanar_three_point`, at finite $\varepsilon$): Fuchsian at $0,16,\infty$; exponents $\{1/2,\,1/2-\varepsilon\}$ at $x=0$, $\{0,-\varepsilon\}$ at 16, $\{2\varepsilon,-1\}$ at infinity; no resonances. At $\varepsilon=0$ (the only order that matters, $\varepsilon^4$ is solved with $B$ alone) the residue at $x=0$ is $\begin{pmatrix}0&-1/2\\1/2&1\end{pmatrix}$: double exponent $1/2$ with a Jordan block (vMT 4.29: $\sqrt x$, $\sqrt x\ln x$); the physical solution has no such branch (vMT 4.40). One block $\{0,1\}$: both masters are unknowns, $K$ is empty.
