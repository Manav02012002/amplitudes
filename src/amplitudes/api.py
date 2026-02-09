from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .spinor import SpinorPoint
from .bcfw import bcfw_color_ordered_tree
from .me import matrix_element_squared_gluons_exact_SU_N
from .me_quark import matrix_element_squared_qqbar_ng_exact_SU_N
from .bcfw_quark import primitive_q_qb_gluons
from .particles import Particle, gluon, quark, antiquark, vector

@dataclass(frozen=True)
class SpinorHelicity:
    momenta: np.ndarray
    sp: SpinorPoint

    @staticmethod
    def from_momenta(momenta: np.ndarray) -> "SpinorHelicity":
        sp = SpinorPoint.from_momenta(momenta)
        return SpinorHelicity(momenta=np.asarray(momenta, dtype=np.complex128), sp=sp)

    def angle(self, i: int, j: int) -> complex:
        return self.sp.ang(i, j)

    def square(self, i: int, j: int) -> complex:
        return self.sp.sqr(i, j)

    def sandwich(self, i: int, P: np.ndarray, j: int) -> complex:
        return self.sp.sand(i, P, j)

class BCFW:
    @staticmethod
    def tree_amplitude(process: list[Particle], momenta: np.ndarray, order: tuple[int, ...] | None = None) -> complex:
        """
        Unified entry point:
          - all gluons: returns color-ordered amplitude in the given cyclic order
          - exactly one q and one qb with gluons: returns the primitive amplitude with ordering (q ... qb)
        Helicity convention: use Particle.hel = +/-1.
        """
        helicities = tuple(p.hel for p in process)
        kinds = tuple(p.kind for p in process)
        sh = SpinorHelicity.from_momenta(momenta)

        if order is not None:
            idx = list(order)
            lam = sh.sp.lam[idx]
            lamt = sh.sp.lamt[idx]
            sp = SpinorPoint(lam=lam, lamt=lamt)
            helicities = tuple(helicities[i] for i in idx)
            kinds = tuple(kinds[i] for i in idx)
        else:
            sp = sh.sp

        if kinds.count("q") == 1 and kinds.count("qb") == 1 and all(k in ("q","qb","g") for k in kinds):
            # require q at start and qb at end for the primitive wrapper
            if kinds[0] != "q" or kinds[-1] != "qb":
                raise ValueError("For quark primitives, provide cyclic order with q first and qb last.")
            return primitive_q_qb_gluons(sp, helicities)
        if all(k == "g" for k in kinds):
            return bcfw_color_ordered_tree(sp, helicities, i=0, j=1)
        raise ValueError("Unsupported process kinds for tree_amplitude")

    @staticmethod
    def tree_amplitude_gluons_color_ordered(momenta: np.ndarray, helicities: tuple[int, ...], order: tuple[int, ...] | None = None) -> complex:
        sh = SpinorHelicity.from_momenta(momenta)
        if order is not None:
            lam = sh.sp.lam[list(order)]
            lamt = sh.sp.lamt[list(order)]
            sp = SpinorPoint(lam=lam, lamt=lamt)
            hel = tuple(helicities[i] for i in order)
            return bcfw_color_ordered_tree(sp, hel, i=0, j=1)
        return bcfw_color_ordered_tree(sh.sp, helicities, i=0, j=1)

class ColorDecomposition:
    @staticmethod
    def matrix_element_squared_qqbar_ng(momenta: np.ndarray, helicities: tuple[int, ...], Nc: int = 3, g_s: float = 1.0) -> float:
        """Exact color-summed |M|^2 for ordering [q, g..., qb]."""
        sh = SpinorHelicity.from_momenta(momenta)
        return matrix_element_squared_qqbar_ng_exact_SU_N(sh.sp, helicities, Nc=Nc, g_s=g_s)


    @staticmethod
    def matrix_element_squared_gluons(momenta: np.ndarray, helicities: tuple[int, ...], Nc: int = 3, g_s: float = 1.0) -> float:
        sh = SpinorHelicity.from_momenta(momenta)
        return matrix_element_squared_gluons_exact_SU_N(sh.sp, helicities, Nc=Nc, g_s=g_s)
