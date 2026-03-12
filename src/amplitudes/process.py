from __future__ import annotations
from dataclasses import dataclass
import itertools
from typing import Sequence
import numpy as np
from .particles import Particle
from .crossing import External, to_all_outgoing
from .spinor import SpinorPoint
from .me import matrix_element_squared_gluons_exact_SU_N
from .me_quark import matrix_element_squared_qqbar_ng_exact_SU_N
from .sm import SMParams, gamma_coupling

@dataclass(frozen=True)
class ProcessSpec:
    initial: tuple[External, ...]
    final: tuple[External, ...]
    Nc: int = 3
    params: SMParams = SMParams()
    include_ew: bool = False
    quark_flavor: str = "u"  # used for EW currents on a single quark line

    def __post_init__(self) -> None:
        object.__setattr__(self, "initial", tuple(self.initial))
        object.__setattr__(self, "final", tuple(self.final))

    def all_outgoing_particles(self) -> tuple[Particle, ...]:
        process, _ = to_all_outgoing(list(self.initial), list(self.final))
        return tuple(process)

    def all_outgoing_momenta(self) -> np.ndarray:
        _, mom = to_all_outgoing(list(self.initial), list(self.final))
        return mom

    def all_outgoing_kinds(self) -> tuple[str, ...]:
        return tuple(p.kind for p in self.all_outgoing_particles())

    def helicity_tuple_all_outgoing(self) -> tuple[int, ...]:
        return tuple(p.hel for p in self.all_outgoing_particles())

    def validate_supported_tree_process(self) -> str:
        return _tree_process_signature(self.all_outgoing_kinds())


@dataclass(frozen=True, init=False)
class Process(ProcessSpec):
    def __init__(
        self,
        initial: Sequence[External],
        final: Sequence[External],
        Nc: int = 3,
        params: SMParams = SMParams(),
        include_ew: bool = False,
        quark_flavor: str = "u",
    ) -> None:
        super().__init__(
            initial=tuple(initial),
            final=tuple(final),
            Nc=Nc,
            params=params,
            include_ew=include_ew,
            quark_flavor=quark_flavor,
        )

def _helicity_configs(n: int):
    return list(itertools.product([-1,+1], repeat=n))


def _tree_process_signature(kinds: Sequence[str]) -> str:
    if all(k == "g" for k in kinds):
        return "gluons"
    if kinds.count("q") == 1 and kinds.count("qb") == 1 and all(k in ("q", "qb", "g") for k in kinds):
        return "qqbar_ng"
    if kinds.count("q") == 1 and kinds.count("qb") == 1 and kinds.count("v") == 1 and all(
        k in ("q", "qb", "g", "v") for k in kinds
    ):
        return "qqbar_v_ng"
    raise ValueError(f"Unsupported process kinds: {tuple(kinds)}")


def _canonical_tree_order(kinds: Sequence[str]) -> tuple[int, ...]:
    signature = _tree_process_signature(kinds)
    if signature == "gluons":
        return tuple(range(len(kinds)))
    iq = kinds.index("q")
    iqb = kinds.index("qb")
    gluons = [i for i, kind in enumerate(kinds) if kind == "g"]
    if signature == "qqbar_ng":
        return (iq, *gluons, iqb)
    iv = kinds.index("v")
    return (iq, *gluons, iv, iqb)


def matrix_element_squared(proc: ProcessSpec, sum_helicities: bool = True, average_initial: bool = True) -> float:
    """
    Compute color-summed |M|^2 for supported processes in a crossing-safe way.

    Supported currently:
      - gg -> ng  (QCD)
      - q qbar -> ng (QCD)
      - q qbar -> V + ng (EW current on quark line, V = gamma or Z treated as massless vector 'v')
        (implemented by multiplying the QCD primitive by the appropriate coupling on that line)
    """
    process = proc.all_outgoing_particles()
    mom = proc.all_outgoing_momenta()
    kinds = proc.all_outgoing_kinds()
    signature = proc.validate_supported_tree_process()

    sp = SpinorPoint.from_momenta(mom)
    Nc = proc.Nc
    gs = proc.params.gs()

    # Identify patterns in all-outgoing convention
    # gg->ng all-outgoing becomes 2 incoming crossed: still "g" kinds in first two legs (crossed already)
    if signature == "gluons":
        n = len(process)
        hel_list = _helicity_configs(n) if sum_helicities else [tuple(p.hel for p in process)]
        me2 = 0.0
        for hels in hel_list:
            me2 += matrix_element_squared_gluons_exact_SU_N(sp, hels, Nc=Nc, g_s=gs)
        if average_initial:
            me2 *= 1.0/4.0 * 1.0/((Nc*Nc-1.0)**2)
        return float(me2)

    # q qbar + ng (all outgoing ordering must be [q, g..., qb] for our primitives)
    if signature == "qqbar_ng":
        order = list(_canonical_tree_order(kinds))
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
    if signature == "qqbar_v_ng":
        order = list(_canonical_tree_order(kinds))
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
