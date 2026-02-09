from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Particle:
    """External particle label for amplitudes.

    kind:
      - QCD: 'g', 'q', 'qb'
      - leptons: 'l', 'lb' (charged lepton / anti-lepton)
      - EW gauge: 'A', 'Z', 'W+', 'W-'
      - colorless massless vector placeholder: 'v' (kept for backwards compatibility)
      - Higgs scalar: 'h' (optional in the engine)
    hel: -1/+1 for helicity (bosons), -1/+1 for fermion helicity in massless limit.
    flavor: for fermions: 'u','d','s','c','b','t','e','mu','tau','nu_e','nu_mu','nu_tau' etc.
    """
    kind: str
    hel: int
    flavor: str | None = None

def _check_hel(hel: int) -> None:
    if hel not in (-1, +1):
        raise ValueError("helicity must be -1 or +1")

def gluon(hel: int) -> Particle:
    _check_hel(hel)
    return Particle("g", hel)

def quark(hel: int, flavor: str = "u") -> Particle:
    _check_hel(hel)
    return Particle("q", hel, flavor=flavor)

def antiquark(hel: int, flavor: str = "u") -> Particle:
    _check_hel(hel)
    return Particle("qb", hel, flavor=flavor)

def lepton(hel: int, flavor: str = "e") -> Particle:
    _check_hel(hel)
    return Particle("l", hel, flavor=flavor)

def antilepton(hel: int, flavor: str = "e") -> Particle:
    _check_hel(hel)
    return Particle("lb", hel, flavor=flavor)

def photon(hel: int) -> Particle:
    _check_hel(hel)
    return Particle("A", hel)

def zboson(hel: int) -> Particle:
    _check_hel(hel)
    return Particle("Z", hel)

def wplus(hel: int) -> Particle:
    _check_hel(hel)
    return Particle("W+", hel)

def wminus(hel: int) -> Particle:
    _check_hel(hel)
    return Particle("W-", hel)

def vector(hel: int) -> Particle:
    _check_hel(hel)
    return Particle("v", hel)

def higgs() -> Particle:
    return Particle("h", +1)
