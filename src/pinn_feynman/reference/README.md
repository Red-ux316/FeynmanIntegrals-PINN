# pinn_feynman.reference

True values of the Master Integrals, from closed formulas in the literature: they are the PINN's boundary and test data. Boni produced them with AMFlow in Mathematica (thesis §6.1.1); for the massless box the closed formulas give the same numbers (checked against his plots and in the tests) without needing Mathematica.

## `box_massless.py` — one-loop massless box

**Formulas** (measure $d^Dk/(i\pi^{D/2})$, propagators $[-(k+q)^2]$, Henn eq. 3.6):
- bubble, exact in $\varepsilon$ (Henn eq. 3.17 with $a_1=a_2=1$):
  $G_{\rm bub}(p^2)=(-p^2)^{-\varepsilon}\,\Gamma(\varepsilon)\Gamma(1-\varepsilon)^2/\Gamma(2-2\varepsilon)$;
- box, exact in $\varepsilon$, derived from Bern–Dixon–Kosower (hep-ph/9306240) eq. (4.18) with (4.12), (2.15), (2.11):
  $I_{\rm box}=\dfrac{2r_\Gamma}{\varepsilon^2st}\Big[(-s)^{-\varepsilon}\,{}_2F_1(1,-\varepsilon;1-\varepsilon;1+\tfrac{s}{t})+(-t)^{-\varepsilon}\,{}_2F_1(1,-\varepsilon;1-\varepsilon;1+\tfrac{t}{s})\Big]$,
  $r_\Gamma=\Gamma(1+\varepsilon)\Gamma(1-\varepsilon)^2/\Gamma(1-2\varepsilon)$;
- continuation to the physical region (BDK eq. 4.24): $\ln(-s)\to\ln|s|-i\pi$ for $s>0$.

**Boni's conventions** (fixed against his plots, Fig. 6.5–6.7): $\vec M=e^{\varepsilon\gamma_E}(t^\varepsilon G_{0101},\ t^\varepsilon G_{1010},\ t^{2+\varepsilon}G_{1111})$ with $t^\varepsilon=e^{\varepsilon\log t}$, $\log t=\ln|t|+i\pi$ for $t<0$.

| Name | What it does |
|---|---|
| `ORDERS` | Boni's Laurent orders: $\varepsilon^{-2}\ldots\varepsilon^{2}$ |
| `log_minus(v)` | $\log(-v-i0)$ with the Feynman prescription |
| `r_gamma(eps)` | $r_\Gamma$ |
| `bubble(p2, eps)` | massless bubble, finite $\varepsilon$ (also complex) |
| `box(s, t, eps)` | massless box, finite $\varepsilon$ (also complex) |
| `boni_basis(x, eps, s=1)` | $(M_1,M_2,M_3)$ in $x=t/s$ for finite $\varepsilon$ |
| `laurent_coefficients(x, orders)` | complex Laurent coefficients, shape `(3, len(orders))`, via Cauchy's formula (64 nodes, radius 0.2, 30 digits) |
| `reference_values(xs, orders)` | the same on a grid: shape `(len(xs), 3, len(orders))`. Real and imaginary part = Boni's two trainings. **Cost: about 0.2 s per point** |

### How the Laurent coefficients are computed

There is no expansion in $\varepsilon$ by hand, and no automatic mpmath command. The coefficients are extracted numerically from the **exact** formula, in three steps.

1. **The exact function at complex $\varepsilon$.** `boni_basis(x, eps)` evaluates $\vec M$ at any finite $\varepsilon$, also complex. No expansion is done there.

2. **Cauchy's formula.** If $f(\varepsilon)=\sum_l c_l\,\varepsilon^l$, then
   $$c_l=\frac{1}{2\pi i}\oint\frac{f(\varepsilon)}{\varepsilon^{l+1}}\,d\varepsilon .$$
   On the circle $\varepsilon=r\,e^{i\theta}$ this is an average over the angle. With $N$ equally spaced nodes (trapezoidal rule):
   $$c_l=\frac{1}{2\pi}\int_0^{2\pi}f(re^{i\theta})\,(re^{i\theta})^{-l}\,d\theta\;\approx\;\frac1N\sum_{j=0}^{N-1}f(\varepsilon_j)\,\varepsilon_j^{-l},\qquad\varepsilon_j=r\,e^{2\pi ij/N} .$$
   The code uses $r=0.2$, $N=64$, 30 digits. One set of 64 evaluations gives all five orders for all three master integrals.

3. **Why it is accurate.** For a function analytic on the circle, the trapezoidal rule converges exponentially. The only error is *aliasing*: the sum returns $c_l+c_{l+N}\,r^{N}+c_{l-N}\,r^{-N}+\dots$, where the $c_{l-N}$ vanish because the most singular order is $\varepsilon^{-2}$. The nearest singularities in $\varepsilon$ (other than the pole at $0$) are at $|\varepsilon|=1$ (poles of the Gamma functions and of ${}_2F_1$). By the Cauchy estimate on a circle of radius $R<1$, $|c_k|\le M_R\,R^{-k-2}$, with $M_R$ the maximum of $|\varepsilon^2 f|$ there. So the aliasing term is of relative size $(r/R)^N$ up to factors of order one, e.g. $(0.2/0.9)^{64}\approx10^{-42}$. This is below the 30 working digits, which are the real limit.

**Why not expand by hand.** Up to $\varepsilon^2$ the box needs the expansion of ${}_2F_1$ in polylogarithms up to weight 4, with the right branch for every piece. Cauchy's formula avoids both: it only uses the exact function, whose branches are fixed once by `log_minus` (BDK eq. 4.24).

**Checked** against an independent route, which computes the coefficients by series algebra in $\varepsilon$ (zeta values and polylogarithms instead of Gamma and ${}_2F_1$ on the circle). The package values agree with it within $10^{-16}$ (relative, i.e. the final rounding to double precision) down to $x=-10^{-8}$. Two other contours (128 nodes, radius 0.1; 96 nodes, radius 0.35) agree with it within $1.6\cdot10^{-37}$: the contour formula has converged.

**Checks**:
1. the first three orders of the box match BDK (I.11) within $10^{-20}$;
2. the coefficients satisfy the ODE (A.2) at all five orders;
3. box and bubble match the **direct numerical integration** of their Feynman-parameter representation (BDK eq. 2.2), at finite $\varepsilon$;
4. $M_1$ constant, $\mathrm{Re}\,M_2^{(0)}=2+\ln(-x)$;
5. reproduction of the values read from Boni's plots.

## `box_one_mass.py` — one-loop one-mass box

True values of the four master integrals of the one-mass box in **Boni's physical region** (thesis eq. 6.38), as functions of two variables $(s_1,s_2)$ with $s_0=1$ ($s_0=s$, $s_1=t$, $s_2=m^2$). The formulas are in the module docstring; the checks below are in the tests.

**Region.** $s_1<0$, $s_2>0$, $s_0+s_1-s_2>0$, i.e. $s>0$, $m^2>0$, $t<0$, $u=m^2-s-t<0$. It is the open triangle with vertices $(0,0)$, $(-1,0)$, $(0,1)$ in $(s_1,s_2)$. The module accepts every point of the open triangle (no margin): the training domain is variable and is chosen by the systems module (`vertices(margin)`).

**Basis.** $\vec M=(\text{box},\ \text{bubble}_t,\ \text{bubble}_{m^2},\ \text{bubble}_s)$, the order of the matrices of the thesis (App. A.4–A.5, with typographical corrections, see `systems/README.md`), not the order of its eq. 6.34. The normalisation is the one of the massless box ($e^{\varepsilon\gamma_E}$, $\mu=1$), the same prefactors as thesis eq. 6.34. $M_3$ (the $m^2$ bubble) is constant in the region; $\mathrm{Im}\,M_2=0$ at all orders.

**Formulas.**
- bubbles: Henn eq. 3.17 in the three channels;
- box: BDK eq. (4.36) in closed form, continued to the region with an explicit rule: $1-y\to1-y+i0$, i.e. the phase $+i\pi$ is written by hand on the logarithm of the two negative bases. It is **never** taken from `log` of a negative number (principal branches give a wrong value by 25–85 %; signed zero is a trap: `np.log(complex(-1,-0.0)) = -i\pi`).

**Two methods for the Laurent coefficients** $\varepsilon^{-2}\dots\varepsilon^{2}$ (shape `(N, 4, 5)`, complex; real and imaginary part are Boni's two trainings, split downstream as for the massless box):

| Name | What it does | Cost |
|---|---|---|
| `reference_values(points, orders=ORDERS)` | **method B**, analytic expansion in $\varepsilon$ with polylogarithms and Nielsen functions, numpy, in blocks of `BLOCK_SIZE` points. This is what produces the training data | about 140 µs per point, $10^5$ points in about 14 s |
| `laurent_coefficients(point, orders=ORDERS, s0=1)` | **method A1**, Cauchy's formula on a circle in $\varepsilon$ (64 nodes, radius 0.2, 30 digits) as in `box_massless.py`. The slow, accurate judge of the tests | about 0.3–0.7 s per point |
| `boni_basis(s1, s2, eps, s0=1)` | exact $\vec M$ at finite $\varepsilon$ (also complex) | — |
| `box(s0, s1, s2, eps)` | $G_{1111}$ at finite $\varepsilon$ | — |
| `hypergeometric_series(z)`, `nielsen_polylogs(z)` | the expansion of ${}_2F_1(-\varepsilon,-\varepsilon;1-\varepsilon;z)$ and the functions it needs | — |

`reference_values` raises `TypeError` for complex input and `ValueError` for a wrong shape, NaN, points outside the open region (also on its sides) or orders it cannot compute.

**Constants for the data cache key:** `METHOD = "series"`, `ORDERS`, `TS_STEP`, `TS_TMAX`. `BLOCK_SIZE` does not change the result (tested). The mpmath constants are computed lazily, not at import.

**Accuracy.** Method B agrees with A1 within about $10^{-15}$ (mixed absolute/relative) on the vertices of the triangle with margin 0.1, near a side and at margin $10^{-3}$. It is **not verified beyond margin $10^{-3}$**: recheck the quadrature before going closer to the sides.

**Checks**:
1. B against A1 on 10 points up to margin $10^{-3}$;
2. poles and finite part of the box against the literature, BDK (I.12) continued with (4.24); the wrong side of the cut differs by $4\pi i\ln(1-x)$ and is excluded by this test alone;
3. the differential equations $\partial M=AM$ at finite complex $\varepsilon$ on A1, and order by order on the output of B (finite differences);
4. the expansion of ${}_2F_1$ against `mpmath.hyp2f1`;
5. scale invariance ($\lambda=2.5$); $M_3$ constant, $\mathrm{Im}\,M_2=0$;
6. the closed box against direct numerical integration of the Feynman representation (a 1D reduction copied into the test file), real and complex $\varepsilon$;
7. regression on a stored table (same formula: not independent), and acceptance of the same points as the systems module.

**Registration:** the problem is declared in `problems/box_one_mass.py`; this module gives `CACHE_PARAMS` (what identifies its output, enters the cache key) and `COMPUTE_OPTIONS` (blocks of `BLOCK_SIZE` points in one process: the chunk of 10 and the process pool of `compute_reference` are counter-productive with method B). The same two constants are in `box_massless.py`; every reference module of a problem must define them with `reference_values` and `ORDERS`.

## `sunrise_equal_mass.py` — two-loop equal-mass sunrise, $m^2=3$

True values of $(I_{11000},I_{11100},I_{21100})$ at the orders $\varepsilon^{-2}\ldots\varepsilon^0$ for $s\in[-100,-0.01]$ (the range checked against pySecDec), conventions of `systems.sunrise_equal_mass`. There is no closed form: the values come from **integrating the (A.10)**.

- $I_{11000}=3^{2-2\varepsilon}\Gamma(\varepsilon-1)^2$, expanded with mpmath (`tadpole_series`);
- pole parts of $I_{11100}$, $I_{21100}$ in closed form (`pole_parts`; checked against pySecDec at four points): $9/(2\varepsilon^2)-[9(\gamma_E-\tfrac32+\ln3)+s/4]/\varepsilon$ and $1/(2\varepsilon^2)-[\gamma_E-\tfrac12+\ln3]/\varepsilon$;
- finite parts at $s=-1$ from pySecDec (requested `epsrel` $10^{-9}$, declared errors $4\cdot10^{-13}$ and $9\cdot10^{-14}$), read from `starting_values.csv`;
- the six unknowns are integrated in $t=\ln(-s)$ from $s=-1$ to both ends of the range (DOP853, `rtol` $10^{-13}$), **once per process**, and read from the dense output: $10^5$ points in about 0.1 s after a first call of a few seconds. One global integration means that the values never depend on how the points are split (`COMPUTE_OPTIONS`: one chunk, one process).

| Name | What it does |
|---|---|
| `ORDERS`, `N_MASTERS`, `START`, `START_FINITE`, `START_FINITE_SIGMA`, `S_RANGE`, `CACHE_PARAMS`, `COMPUTE_OPTIONS` | constants (orders $-2..0$; the starting point, its data and their errors; the range; what enters the cache key) |
| `read_starts(problem)` | the starting values of a problem, from `starting_values.csv` (also used for $s>0$) |
| `tadpole_series()` | the three Laurent coefficients of $I_{11000}$ |
| `pole_parts(s)` | `(N, 2, 2)`: orders $\varepsilon^{-2},\varepsilon^{-1}$ of $I_{11100}$, $I_{21100}$ |
| `reference_values(s, orders=ORDERS)` | complex `(N, 3, 3)` with zero imaginary part; `TypeError` for complex input, `ValueError` for NaN, points outside `S_RANGE` or other orders |

**Limits.** The precision is that of the starting data propagated by the equation: against the other three pySecDec points ($s=-0.01,-10,-100$) the mixed error is $\le3.4\cdot10^{-13}$ (in $[-10,-1]$: $\le10^{-13}$). The equation integrated here is **the same (A.10) the network solves**: these values do not check the matrix independently. Only $s<0$ (no imaginary part, no threshold).

**Checks**: tadpole against the sympy series of the closed form and against pySecDec; reproduction of the four pySecDec points within $10^{-12}$ (mixed); pole parts from the integration equal to the closed form within $10^{-12}$; independence from the split; the system $dM/ds=AM$ on fourth-order finite differences ($\le10^{-8}$ relative).

**Registration:** `problems/sunrise_equal_mass.py` (with `parts = ("real",)`) and the bottom-up `problems/sunrise_equal_mass_bu.py`.

## `sunrise_equal_mass_positive.py` — the same sunrise for $s>0$

Same basis, conventions, tadpole and pole parts as `sunrise_equal_mass.py`. The physical integral is analytic in the plane cut along $[27,\infty)$: $F>0$ on the simplex for every $s<27=(3m)^2$, so it is real and regular at $s=0$ and $s=3$ (singular points of the equation only), and above 27 its value is the limit $s+i0$.

- one starting point in each interval between the singular points, $s=1.5,\ 15,\ 45$: finite parts of $I_{11100}$, $I_{21100}$ from AMFlow + Kira (12, 12 and 10 digits), read from `starting_values.csv`;
- from there the six unknowns are integrated along the (A.10) on the real axis (DOP853, `rtol` $10^{-13}$) to both ends of the interval and read from the dense output; the points $0,3,27$ are never crossed;
- the imaginary part is set to zero below 27, and at $\varepsilon^{-2},\varepsilon^{-1}$ everywhere (the pole parts are polynomials in $s$).

| Name | Content |
|---|---|
| `S_RANGES` | covered intervals $[0.001,2.999]$, $[3.001,26.999]$, $[27.001,100]$; any other point is refused |
| `STARTS` | the starting points and their AMFlow finite parts (from `starting_values.csv`) |
| `start_values(index)` | the six unknowns at the starting point of `S_RANGES[index]` |
| `reference_values(s, orders)` | complex array $(N,3,3)$: real below 27, $s+i0$ above |
| `CACHE_PARAMS`, `COMPUTE_OPTIONS` | own cache key (problem `sunrise_equal_mass_positive`), one chunk and one process |

**Limits.** The precision is that of the AMFlow starting data propagated by the equation, which is the same (A.10) the network solves.

**Checks**: the continuation from $s=-1$ (pySecDec start, a path through the upper half plane) gives the same values within $10^{-10}$; the starting points return the AMFlow values; reality below 27; imaginary part at $\varepsilon^0$ above 27 against an independent Feynman-parameter integral (12 digits) within $10^{-10}$ relative; tadpole column equal to the closed form; refusal of points outside the intervals.

**Registration:** `problems/sunrise_equal_mass_positive.py` (real and imaginary part) and the bottom-up `problems/sunrise_equal_mass_positive_bu.py`.

## `starting_values.csv` — the starting values of the sunrise

The sunrise has no closed form: its values come from integrating the (A.10) from a few points where the finite parts were computed numerically. Those numbers are in this file, one row per master and point, not in the code:

| Column | Content |
|---|---|
| `problem` | `sunrise_equal_mass` ($s<0$) or `sunrise_equal_mass_positive` ($s>0$) |
| `s`, `master`, `order` | the point, `I_11100` or `I_21100`, the order of $\varepsilon$ (0: only finite parts) |
| `real`, `imag` | the value (all the printed digits are kept) |
| `error`, `digits` | declared absolute error (pySecDec) or guaranteed digits (AMFlow) |
| `source` | how the value was obtained |

`sunrise_equal_mass.read_starts(problem)` reads it. The values enter the cache key of the datasets (`CACHE_PARAMS`): changing one gives new data.

## `nonplanar_three_point.py` — non-planar two-loop three-point function

True values of the two top masters $(m_{10},m_{11})$ (basis of von Manteuffel–Tancredi arXiv:1701.05905, "vMT", = Boni 6.57) at the orders $\varepsilon^0\ldots\varepsilon^4$, for $x=-s/m^2\in[-30,1]$, $s\to s+i0$ (i.e. $x\to x-i0$). The orders $\varepsilon^0\ldots\varepsilon^3$ vanish identically (vMT §4.1); at $\varepsilon^4$ the masters satisfy $dm/dx=B(x)\,m+(0,N_{11}^{(4)})$.

- `forcing(x)`: $N_{11}^{(4)}$ on the real axis, vMT (4.34) for $0<x<4$, (4.35) for $-4\le x<0$, (4.36) for $x<-4$ (numpy; logarithms of the letters as `asinh`, arcsec as `atan2`, Clausen function by its series `clausen2`). Equal to the continuation of vMT (4.2) to $x-i0$ within $10^{-13}$ (tests);
- starting values at $x=1,-2,-10$ by **variation of constants** with the homogeneous solutions of vMT (4.18), (4.22) (complete elliptic integrals) continued to $x-i0$ and the constants of vMT (4.40) (no $\sqrt x$ branch at $x=0$): `variation_of_constants`, mpmath at 20 digits, about 2 s for the three points, once per process;
- five pieces integrated with DOP853 (`rtol` $10^{-13}$) in variables where the forcing is smooth: $t=\ln|x|$ towards $x=0$ (from 1 and from $-2$), $u=\sqrt{|x+4|}$ towards the threshold (from $-2$ and from $-10$), $t=\ln(-x)$ from $-10$ to $-30$; read from the dense outputs. About 3.5 s for the integration, then $10^5$ points in about 0.1 s.

| Name | What it does |
|---|---|
| `ORDERS`, `N_MASTERS`, `LEADING_ORDER`, `X_RANGE`, `X_SMALLEST`, `STARTS`, `CACHE_PARAMS`, `COMPUTE_OPTIONS` | constants: orders $0..4$, non-zero order 4, range $[-30,1]$, $0<|x|<10^{-12}$ refused ($x=0$ gives 0), starting points |
| `forcing(x)`, `clausen2(theta)` | $N_{11}^{(4)}(x-i0)$ for real $x<4$; $\mathrm{Cl}_2$ for $|\theta|\le\pi$ |
| `variation_of_constants(x, digits=20)` | $(m_{10}^{(4)},m_{11}^{(4)})$ at one point, mpmath (vMT 4.40 continued) |
| `start_values()`, `leading_values(x)` | the three starting values; `(N, 2)` complex at $\varepsilon^4$ |
| `reference_values(x, orders=ORDERS)` | complex `(N, 2, 5)`, only $\varepsilon^4$ non-zero; `TypeError` for complex input, `ValueError` for NaN, points outside the range or other orders |

**Limits.** The equation integrated here is the one the network solves (same $B$, same forcing): these values do not check them independently. The independent checks were: the forcing against vMT (4.2), the solution against the 50-digit value of vMT (4.44) at $x=-5$ ($1.7\cdot10^{-31}$ at 30 digits) and against vMT Fig. 1. Here the error against (4.44) is $9\cdot10^{-15}$, against `variation_of_constants` at other points $\le10^{-12}$.

**Checks**: (4.44); the two sides of $x=-4$ (independent starting points) agree within $10^{-11}$; variation of constants at $x=-0.5,-15$; orders below 4 zero, real for $x\ge0$; independence from the split; zero residual of the bottom-up system on fourth-order finite differences (real and imaginary parts), and non-zero without the forcing.

**Registration:** `problems/nonplanar_three_point.py` (data problem, already with the forcing provider) and `problems/nonplanar_three_point_bu.py` (same problem, bottom-up name).
