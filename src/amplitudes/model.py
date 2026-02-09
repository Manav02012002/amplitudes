from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class SMParams:
    """Minimal but complete Standard Model coupling layer for tree-level EW/QCD amplitudes."""
    alpha_s: float = 0.118
    alpha_em: float = 1/137.035999084
    sin2_theta_w: float = 0.23122

    def gs(self) -> float:
        return math.sqrt(4.0 * math.pi * self.alpha_s)

    def e(self) -> float:
        return math.sqrt(4.0 * math.pi * self.alpha_em)

    def sw(self) -> float:
        return math.sqrt(self.sin2_theta_w)

    def cw(self) -> float:
        return math.sqrt(1.0 - self.sin2_theta_w)

    def gZ(self) -> float:
        return self.e() / (self.sw() * self.cw())

@dataclass(frozen=True)
class QuarkEW:
    Q: float  # electric charge
    T3: float # weak isospin (LH)

def quark_ew(flavor: str) -> QuarkEW:
    f = flavor.lower()
    if f in ("u","c","t"):
        return QuarkEW(Q=+2.0/3.0, T3=+0.5)
    if f in ("d","s","b"):
        return QuarkEW(Q=-1.0/3.0, T3=-0.5)
    raise ValueError("Unknown quark flavor")

def gamma_coupling(params: SMParams, flavor: str) -> float:
    return params.e() * quark_ew(flavor).Q

def z_couplings(params: SMParams, flavor: str) -> tuple[float,float]:
    """Return (g_L, g_R) for Z q qbar vertex in helicity basis."""
    q = quark_ew(flavor)
    sw2 = params.sin2_theta_w
    gZ = params.gZ()
    gL = gZ * (q.T3 - q.Q * sw2)
    gR = gZ * (-q.Q * sw2)
    return gL, gR
