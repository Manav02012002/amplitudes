from __future__ import annotations
import numpy as np
from .crossing import External
from .particles import Particle
from .kinematics import com_incoming_momenta, flux_factor
from .phasespace import rambo_massless
from .phasespace_massive import rambo_massive_equal_m
from .process import Process, matrix_element_squared
from .vegas import vegas_integrate

def xsec_2_to_n(
    initial_particles: tuple[Particle, Particle],
    final_particles: list[Particle],
    Ecm: float,
    Nc: int = 3,
    masses_final: list[float] | None = None,
    neval: int = 20000,
    niter: int = 5,
    seed: int = 1,
    sum_helicities: bool = True,
    average_initial: bool = True,
    include_ew: bool = False,
    quark_flavor: str = "u",
) -> tuple[float,float]:
    """
    General-purpose 2->n cross section engine:
      - crossing-safe (uses explicit initial/final externals)
      - supports massive final states (equal-mass mode implemented; general masses can be added later)

    Final-state generation:
      - if masses_final is None or all 0: RAMBO massless
      - if all masses are equal: uses massive equal-mass RAMBO with Jacobian
    """
    rng = np.random.default_rng(seed)
    n = len(final_particles)
    masses_final = masses_final or [0.0]*n
    if len(masses_final) != n:
        raise ValueError("masses_final length mismatch")

    p1, p2 = com_incoming_momenta(Ecm, m1=0.0, m2=0.0)
    flux = flux_factor(Ecm, m1=0.0, m2=0.0)

    def integrand(_u):
        rr = np.random.default_rng(int(rng.integers(0, 2**32-1)))
        if all(abs(m) < 1e-14 for m in masses_final):
            pf, w = rambo_massless(n, Ecm, rr)
        else:
            # equal-mass full Jacobian mode
            m0 = masses_final[0]
            if any(abs(m - m0) > 1e-12 for m in masses_final):
                raise ValueError("General unequal masses not implemented yet; use equal masses or all-zero.")
            pf, w = rambo_massive_equal_m(n, Ecm, m0, rr)

        init = [
            External(initial_particles[0], p1, incoming=True),
            External(initial_particles[1], p2, incoming=True),
        ]
        final = [External(final_particles[i], pf[i], incoming=False) for i in range(n)]
        proc = Process(initial=init, final=final, Nc=Nc, include_ew=include_ew, quark_flavor=quark_flavor)
        me2 = matrix_element_squared(proc, sum_helicities=sum_helicities, average_initial=average_initial)
        return float((w * me2) / flux)

    I, err = vegas_integrate(integrand, ndim=6, neval=neval, niter=niter, rng=rng)
    return I, err
