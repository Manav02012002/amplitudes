from __future__ import annotations
from dataclasses import dataclass
import itertools
from .crossing import External, to_all_outgoing
from .spinor import SpinorPoint
from .me import matrix_element_squared_gluons_exact_SU_N
from .me_quark import matrix_element_squared_qqbar_ng_exact_SU_N
from .model import SMParams

@dataclass(frozen=True)
class WidthScheme:
    scheme: str = "fixed"

@dataclass(frozen=True)
class TreeEngine:
    params: SMParams = SMParams()
    Nc: int = 3
    width_scheme: WidthScheme = WidthScheme()

    def me2(self, initial: list[External], final: list[External], sum_helicities: bool = True, average_initial: bool = True) -> float:
        proc, mom = to_all_outgoing(initial, final)
        kinds = tuple(p.kind for p in proc)
        sp = SpinorPoint.from_momenta(mom)
        gs = self.params.gs()
        Nc = self.Nc

        def hel_list(n):
            return list(itertools.product([-1,+1], repeat=n)) if sum_helicities else [tuple(p.hel for p in proc)]

        if all(k == "g" for k in kinds):
            me2 = 0.0
            for hels in hel_list(len(proc)):
                me2 += matrix_element_squared_gluons_exact_SU_N(sp, hels, Nc=Nc, g_s=gs)
            if average_initial:
                me2 *= 1.0/4.0 * 1.0/((Nc*Nc-1.0)**2)
            return float(me2)

        if kinds.count("q") == 1 and kinds.count("qb") == 1 and all(k in ("q","qb","g") for k in kinds):
            iq = kinds.index("q"); iqb = kinds.index("qb")
            glu = [i for i,k in enumerate(kinds) if k=="g"]
            order = [iq] + glu + [iqb]
            sp2 = SpinorPoint(lam=sp.lam[order], lamt=sp.lamt[order])
            me2 = 0.0
            for hels in hel_list(len(order)):
                me2 += matrix_element_squared_qqbar_ng_exact_SU_N(sp2, hels, Nc=Nc, g_s=gs)
            if average_initial:
                me2 *= 1.0/4.0 * 1.0/(Nc*Nc)
            return float(me2)

        raise ValueError(f"Unsupported process kinds: {kinds}")
