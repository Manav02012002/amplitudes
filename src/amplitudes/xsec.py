from __future__ import annotations
import itertools
import numpy as np
from .phasespace import rambo_massless
from .spinor import SpinorPoint
from .me import matrix_element_squared_gluons_exact_SU_N
from .me_quark import matrix_element_squared_qqbar_ng_exact_SU_N
from .vegas import vegas_integrate

def _helicity_configs(n: int):
    for hs in itertools.product([-1, +1], repeat=n):
        yield tuple(hs)

def xsec_gg_to_ng(
    ng_final: int,
    Ecm: float,
    Nc: int = 3,
    g_s: float = 1.0,
    neval: int = 20000,
    niter: int = 5,
    seed: int = 1,
    sum_final_helicities: bool = True,
) -> tuple[float, float]:
    """σ for gg -> ng_final gluons (massless) in COM. Uses exact color sums."""
    rng = np.random.default_rng(seed)
    s = Ecm*Ecm
    nlegs = ng_final  # final-state multiplicity for RAMBO

    # initial state averages
    avg_hel = 1.0 / 4.0
    avg_col = 1.0 / ((Nc*Nc - 1.0)**2)

    # We'll treat amplitude with all legs outgoing: 1,2 are initial; we embed them into me2 by
    # providing a set of 'n' outgoing momenta. Here we do a pragmatic approach: we compute only
    # final-state |M|^2 with fixed initial helicities by crossing not implemented.
    # For production, you'd implement crossing; here we provide a fully functional pipeline for
    # 2->n kinematics by using the *same* tree objects in an all-outgoing convention and treating
    # initial legs as outgoing with negative energy. This is standard for numerical amplitude codes.

    p1 = np.array([Ecm/2, 0.0, 0.0, Ecm/2], dtype=np.complex128)
    p2 = np.array([Ecm/2, 0.0, 0.0, -Ecm/2], dtype=np.complex128)

    def integrand(_u):
        rr = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
        pf, w = rambo_massless(nlegs, Ecm, rr)
        # build all-outgoing momenta list: initial are crossed to outgoing with negative sign
        p = np.vstack([-p1, -p2, pf])
        # helicities: include initial + finals (will sum/avg)
        me2 = 0.0
        init_configs = list(itertools.product([-1,+1], repeat=2))
        final_configs = list(_helicity_configs(nlegs)) if sum_final_helicities else [(-1,)*nlegs]
        for hi1, hi2 in init_configs:
            for hf in final_configs:
                hels = (hi1, hi2) + tuple(hf)
                sp = SpinorPoint.from_momenta(p)
                me2 += matrix_element_squared_gluons_exact_SU_N(sp, hels, Nc=Nc, g_s=g_s)
        me2 *= avg_hel * avg_col
        flux = 2.0 * s
        return float((w * me2) / flux)

    I, err = vegas_integrate(integrand, ndim=6, neval=neval, niter=niter, rng=rng)
    return I, err

def xsec_qqbar_to_ng(
    ng_final: int,
    Ecm: float,
    Nc: int = 3,
    g_s: float = 1.0,
    neval: int = 20000,
    niter: int = 5,
    seed: int = 2,
    sum_final_helicities: bool = True,
) -> tuple[float, float]:
    """σ for q qbar -> ng_final gluons (massless) in COM. Exact color sums for qqbar+ng primitives."""
    rng = np.random.default_rng(seed)
    s = Ecm*Ecm
    nlegs = ng_final

    avg_hel = 1.0 / 4.0
    avg_col = 1.0 / (Nc*Nc)

    p1 = np.array([Ecm/2, 0.0, 0.0, Ecm/2], dtype=np.complex128)
    p2 = np.array([Ecm/2, 0.0, 0.0, -Ecm/2], dtype=np.complex128)

    def integrand(_u):
        rr = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
        pf, w = rambo_massless(nlegs, Ecm, rr)

        # all-outgoing: initial crossed
        # ordering for qqbar+ng amplitude in our me_quark expects [q, g..., qb]
        # For crossing, we use q (outgoing) = -p1, qb(outgoing) = -p2
        p = np.vstack([-p1, pf, -p2])

        init_configs = list(itertools.product([-1,+1], repeat=2))
        final_configs = list(_helicity_configs(nlegs)) if sum_final_helicities else [(-1,)*nlegs]
        me2 = 0.0
        for hq, hqb in init_configs:
            for hf in final_configs:
                hels = (hq,) + tuple(hf) + (hqb,)
                sp = SpinorPoint.from_momenta(p)
                me2 += matrix_element_squared_qqbar_ng_exact_SU_N(sp, hels, Nc=Nc, g_s=g_s)
        me2 *= avg_hel * avg_col
        flux = 2.0 * s
        return float((w * me2) / flux)

    I, err = vegas_integrate(integrand, ndim=6, neval=neval, niter=niter, rng=rng)
    return I, err
