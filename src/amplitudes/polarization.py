from __future__ import annotations
import numpy as np
from .lorentz import minkowski_dot, mass2, boost_matrix, apply_lorentz

def _orthonormal_basis(p_spatial: np.ndarray):
    p = np.asarray(p_spatial, dtype=np.float64)
    ph = p / (np.linalg.norm(p) + 1e-30)
    a = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(a, ph)) > 0.9:
        a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    e1 = np.cross(a, ph); e1 /= (np.linalg.norm(e1) + 1e-30)
    e2 = np.cross(ph, e1); e2 /= (np.linalg.norm(e2) + 1e-30)
    return e1, e2, ph

def massive_vector_polarizations(p: np.ndarray, m: float):
    p = np.asarray(p, dtype=np.complex128)
    if abs(m) < 1e-14:
        raise ValueError("m=0")
    pr = p.real
    E = float(pr[0])
    pvec = pr[1:].astype(np.float64)
    pabs = float(np.linalg.norm(pvec))

    # Build two transverse spacelike unit vectors with zero time component
    if pabs < 1e-12:
        e1 = np.array([1.0,0.0,0.0], dtype=np.float64)
        e2 = np.array([0.0,1.0,0.0], dtype=np.float64)
        eh = np.array([0.0,0.0,1.0], dtype=np.float64)
    else:
        eh = pvec / pabs
        a = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(np.dot(a, eh)) > 0.9:
            a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        e1 = np.cross(a, eh); e1 /= (np.linalg.norm(e1) + 1e-30)
        e2 = np.cross(eh, e1); e2 /= (np.linalg.norm(e2) + 1e-30)

    epsp = np.array([0.0, *((e1+1j*e2)/np.sqrt(2.0))], dtype=np.complex128)
    epsm = np.array([0.0, *((e1-1j*e2)/np.sqrt(2.0))], dtype=np.complex128)

    # Longitudinal polarization: (|p|/m, E/m * p_hat)
    if pabs < 1e-12:
        eps0 = np.array([0.0, *eh], dtype=np.complex128)
    else:
        eps0 = np.array([pabs/m, *(E/m * eh)], dtype=np.complex128)

    return epsp, epsm, eps0
    E = float(pr[0])
    beta = (pvec / E).astype(np.float64)
    L = boost_matrix(beta)
    e1,e2,eh = _orthonormal_basis(pvec)
    epsp_rf = np.array([0.0, *((e1+1j*e2)/np.sqrt(2.0))], dtype=np.complex128)
    epsm_rf = np.array([0.0, *((e1-1j*e2)/np.sqrt(2.0))], dtype=np.complex128)
    eps0_rf = np.array([0.0, *eh], dtype=np.complex128)
    epsp = apply_lorentz(L, epsp_rf)
    epsm = apply_lorentz(L, epsm_rf)
    eps0 = apply_lorentz(L, eps0_rf)
    for idx,e in enumerate([epsp, epsm, eps0]):
        if abs(minkowski_dot(p, e)) > 1e-8:
            e = e - minkowski_dot(p, e)*p/(mass2(p)+1e-30)
        if idx==0: epsp=e
        elif idx==1: epsm=e
        else: eps0=e
    return epsp, epsm, eps0

def massless_vector_polarizations(p: np.ndarray):
    p = np.asarray(p, dtype=np.complex128)
    pr = p.real
    pvec = pr[1:]
    if float(np.linalg.norm(pvec)) < 1e-14:
        raise ValueError("degenerate")
    e1,e2,_ = _orthonormal_basis(pvec)
    epsp = np.array([0.0, *((e1+1j*e2)/np.sqrt(2.0))], dtype=np.complex128)
    epsm = np.array([0.0, *((e1-1j*e2)/np.sqrt(2.0))], dtype=np.complex128)
    return epsp, epsm

def propagator_den(p2: complex, m: float, width: float, scheme: str = "fixed") -> complex:
    if scheme == "fixed":
        return p2 - (m*m) + 1j*m*width
    if scheme == "complex_mass":
        return p2 - (m*m - 1j*m*width)
    raise ValueError("Unknown scheme")
