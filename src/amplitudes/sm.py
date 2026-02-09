from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Dict, Tuple

@dataclass(frozen=True)
class SMParams:
    alpha_s: float = 0.118
    alpha_em: float = 1/137.035999084
    sin2_theta_w: float = 0.23122

    mZ: float = 91.1876
    gZ: float | None = None  # derived if None
    gammaZ: float = 2.4952

    mW: float = 80.379
    gammaW: float = 2.085

    mH: float = 125.25
    gammaH: float = 0.00407

    width_scheme: str = "fixed"  # 'fixed' or 'complex_mass'

    # Simple CKM (Wolfenstein-ish default ~ identity)
    Vckm: Dict[Tuple[str,str], complex] | None = None  # (up,down)->Vud

    def gs(self) -> float:
        return math.sqrt(4.0 * math.pi * self.alpha_s)

    def e(self) -> float:
        return math.sqrt(4.0 * math.pi * self.alpha_em)

    def sw(self) -> float:
        return math.sqrt(self.sin2_theta_w)

    def cw(self) -> float:
        return math.sqrt(1.0 - self.sin2_theta_w)

    def gZ_cpl(self) -> float:
        if self.gZ is not None:
            return self.gZ
        return self.e() / (self.sw() * self.cw())

    def gW_cpl(self) -> float:
        # g / sqrt(2) = e/(sqrt(2) s_w)
        return self.e() / (math.sqrt(2.0) * self.sw())

    def ckm(self, up: str, down: str) -> complex:
        if self.Vckm is None:
            return 1.0 + 0j if (up, down) in {("u","d"),("c","s"),("t","b")} else 0.0 + 0j
        return complex(self.Vckm.get((up, down), 0.0 + 0j))

@dataclass(frozen=True)
class FermionEW:
    Q: float
    T3: float  # LH
    is_neutrino: bool = False

def fermion_ew(flavor: str) -> FermionEW:
    f = flavor.lower()
    if f in ("u","c","t"):
        return FermionEW(Q=+2.0/3.0, T3=+0.5)
    if f in ("d","s","b"):
        return FermionEW(Q=-1.0/3.0, T3=-0.5)
    if f in ("e","mu","tau"):
        return FermionEW(Q=-1.0, T3=-0.5)
    if f in ("nu_e","nu_mu","nu_tau","nue","numu","nutau"):
        return FermionEW(Q=0.0, T3=+0.5, is_neutrino=True)
    raise ValueError(f"Unknown fermion flavor: {flavor}")

def gamma_coupling(params: SMParams, flavor: str) -> float:
    return params.e() * fermion_ew(flavor).Q

def z_couplings(params: SMParams, flavor: str) -> tuple[float, float]:
    # gL = gZ (T3 - Q s_w^2), gR = gZ (-Q s_w^2)
    fe = fermion_ew(flavor)
    sw2 = params.sin2_theta_w
    gZ = params.gZ_cpl()
    gL = gZ * (fe.T3 - fe.Q * sw2)
    gR = gZ * (-fe.Q * sw2)
    return gL, gR

def w_coupling_L(params: SMParams, up: str, down: str) -> complex:
    return params.gW_cpl() * params.ckm(up, down)
