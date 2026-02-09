from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .particles import Particle

@dataclass(frozen=True)
class External:
    particle: Particle
    momentum: np.ndarray  # shape (4,), complex128 allowed
    incoming: bool = False

def cross_external(ext: External) -> External:
    """
    Cross an external leg between initial and final states into an all-outgoing convention.

    Convention used throughout this package (massless):
      - Incoming boson (g or v): becomes outgoing boson with momentum p_out = -p_in and SAME helicity label.
      - Incoming quark q: becomes outgoing antiquark qb with momentum -p and SAME helicity label.
      - Incoming antiquark qb: becomes outgoing quark q with momentum -p and SAME helicity label.

    This is a standard crossing choice in all-outgoing helicity amplitude codes; overall phases cancel in |M|^2.
    """
    p = np.asarray(ext.momentum, dtype=np.complex128)
    if not ext.incoming:
        return ext
    kind = ext.particle.kind
    hel = ext.particle.hel
    if kind == "q":
        pk = Particle("qb", hel)
    elif kind == "qb":
        pk = Particle("q", hel)
    elif kind in ("g", "v"):
        pk = Particle(kind, hel)
    else:
        raise ValueError(f"Unsupported particle kind for crossing: {kind}")
    return External(particle=pk, momentum=-p, incoming=False)

def to_all_outgoing(initial: list[External], final: list[External]) -> tuple[list[Particle], np.ndarray]:
    outs: list[External] = []
    for e in initial:
        outs.append(cross_external(e))
    for e in final:
        if e.incoming:
            raise ValueError("Final-state external must have incoming=False")
        outs.append(e)
    process = [e.particle for e in outs]
    mom = np.stack([e.momentum for e in outs]).astype(np.complex128)
    return process, mom
