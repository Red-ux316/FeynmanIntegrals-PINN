"""pinn_feynman: PINNs for the differential equations of Feynman master integrals."""

import os

# OpenMP clash on Windows: the pip torch wheel and the conda MKL numpy ship
# two copies of libiomp5md.dll; if torch initialises its copy first, the first threaded
# numpy BLAS call aborts the interpreter (OMP Error #15). With MKL's sequential layer numpy
# never loads OpenMP. It must be set before numpy is imported; numpy here only multiplies
# 3x3 matrices, so one thread is enough.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
