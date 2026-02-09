from __future__ import annotations
from dataclasses import dataclass
import itertools
import numpy as np
from .particles import Particle
from .crossing import External, to_all_outgoing
from .spinor import SpinorPoint
from .me import matrix_element_squared_gluons_exact_SU_N
from .me_quark import matrix_element_squared_qqbar_ng_exact_SU_N
from .model import SMParams, gamma_coupling, z_couplings

@dataclass(frozen=True)
class Process:
    initial: list[External]
    final: list[External]
    Nc: int = 3
    params: SMParams = SMParams()
    include_ew: bool = False
    quark_flavor: str = "u"  # used for EW currents on a single quark line

def _helicity_configs(n: int):
    return list(itertools.product([-1,+1], repeat=n))

def matrix_element_squared(proc: Process, sum_helicities: bool = True, average_initial: bool = True) -> float:
    """
    Compute color-summed |M|^2 for supported processes in a crossing-safe way.

    Supported currently:
      - gg -> ng  (QCD)
      - q qbar -> ng (QCD)
      - q qbar -> V + ng (EW current on quark line, V = gamma or Z treated as massless vector 'v')
        (implemented by multiplying the QCD primitive by the appropriate coupling on that line)
    """
    process, mom = to_all_outgoing(proc.initial, proc.final)
    kinds = tuple(p.kind for p in process)

    sp = SpinorPoint.from_momenta(mom)
    Nc = proc.Nc
    gs = proc.params.gs()

    # Identify patterns in all-outgoing convention
    # gg->ng all-outgoing becomes 2 incoming crossed: still "g" kinds in first two legs (crossed already)
    if all(k == "g" for k in kinds):
        n = len(process)
        hel_list = _helicity_configs(n) if sum_helicities else [tuple(p.hel for p in process)]
        me2 = 0.0
        for hels in hel_list:
            me2 += matrix_element_squared_gluons_exact_SU_N(sp, hels, Nc=Nc, g_s=gs)
        if average_initial:
            me2 *= 1.0/4.0 * 1.0/((Nc*Nc-1.0)**2)
        return float(me2)

    # q qbar + ng (all outgoing ordering must be [q, g..., qb] for our primitives)
    if kinds.count("q")==1 and kinds.count("qb")==1 and all(k in ("q","qb","g") for k in kinds):
        iq = kinds.index("q")
        iqb = kinds.index("qb")
        # require q first and qb last by reordering (primitive basis definition)
        order = [iq] + [i for i,k in enumerate(kinds) if k=="g"] + [iqb]
        sp2 = SpinorPoint(lam=sp.lam[order], lamt=sp.lamt[order])
        base_hels = tuple(process[i].hel for i in order)
        hel_list = _helicity_configs(len(order)) if sum_helicities else [base_hels]
        me2 = 0.0
        for hels in hel_list:
            me2 += matrix_element_squared_qqbar_ng_exact_SU_N(sp2, hels, Nc=Nc, g_s=gs)

        if proc.include_ew and ("v" in kinds):
            # (Not reached with current kinds filter; kept for future extension)
            pass

        if average_initial:
            me2 *= 1.0/4.0 * 1.0/(Nc*Nc)
        return float(me2)

    # q qbar -> v + ng : all outgoing should contain one 'v' plus q,qb and gluons
    if kinds.count("q")==1 and kinds.count("qb")==1 and kinds.count("v")==1 and all(k in ("q","qb","g","v") for k in kinds):
        iq = kinds.index("q"); iqb = kinds.index("qb"); iv = kinds.index("v")
        glu = [i for i,k in enumerate(kinds) if k=="g"]
        # Put in order [q, gluons..., v, qb] (vector inserted before qb)
        order = [iq] + glu + [iv] + [iqb]
        sp2 = SpinorPoint(lam=sp.lam[order], lamt=sp.lamt[order])
        base = [process[i] for i in order]
        # helicity sum
        hel_list = _helicity_configs(len(order)) if sum_helicities else [tuple(p.hel for p in base)]
        me2 = 0.0
        # EW coupling factor on quark line depends on vector type; we treat 'v' as photon by default.
        # For Z, you can set proc.quark_flavor and choose Z by swapping couplings below.
        gV = gamma_coupling(proc.params, proc.quark_flavor)
        for hels in hel_list:
            # Use QCD amplitude for q..qb with ng+1 "gluons" where the last is actually colorless vector.
            # Since it's colorless, the color matrix is unchanged; we multiply by coupling^2.
            me2 += (gV*gV) * matrix_element_squared_qqbar_ng_exact_SU_N(sp2, hels, Nc=Nc, g_s=gs)
        if average_initial:
            me2 *= 1.0/4.0 * 1.0/(Nc*Nc)
        return float(me2)

    raise ValueError(f"Unsupported process kinds: {kinds}")
