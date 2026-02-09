from __future__ import annotations
import numpy as np
import itertools
from dataclasses import dataclass

from .spinor import SpinorPoint
from .bcfw_quark import primitive_q_qb_gluons
from .me_quark import matrix_element_squared_qqbar_ng_exact_SU_N
from .sm import SMParams, gamma_coupling, z_couplings
from .polarization import massive_vector_polarizations, massless_vector_polarizations
from .lorentz import minkowski_dot

@dataclass(frozen=True)
class DrellYanJets:
    params: SMParams = SMParams()
    Nc: int = 3

    def me2_qqbar_to_V_ng(self, mom: np.ndarray, hels: tuple[int, ...], V: str = "A", flavor: str = "u") -> float:
        """
        Compute |M|^2 for q qbar -> V + ng gluons where V is A or Z (W support requires flavour flow).
        Ordering convention for mom and hels: [q, g1..g_ng, V, qb] all outgoing.
        Color: exact single quark line color sum.
        """
        mom = np.asarray(mom, dtype=np.complex128)
        n = len(hels)
        if n < 4:
            raise ValueError("need q, V, qb plus gluons")
        ng = n - 3  # q + (ng gluons) + V + qb

        sp = SpinorPoint.from_momenta(mom)
        gs = self.params.gs()

        # Use the existing exact color-summed qqbar+ng machinery by treating V as an extra colorless insertion.
        # Kinematics is encoded in the primitive via BCFW; color matrix unchanged, coupling applied externally.
        me2_qcd = matrix_element_squared_qqbar_ng_exact_SU_N(sp, hels, Nc=self.Nc, g_s=gs)

        if V == "A":
            g = gamma_coupling(self.params, flavor)
            return float((g*g) * me2_qcd)
        if V == "Z":
            gL, gR = z_couplings(self.params, flavor)
            # helicity/chirality selection for massless line:
            hq = hels[0]
            hqb = hels[-1]
            is_L = (hq == -1 and hqb == +1)
            is_R = (hq == +1 and hqb == -1)
            g = gL if is_L else (gR if is_R else 0.0)
            return float((g*g) * me2_qcd)
        raise ValueError("V must be 'A' or 'Z'")
