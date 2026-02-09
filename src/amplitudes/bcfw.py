from __future__ import annotations
import numpy as np
from .spinor import SpinorPoint, mass2, sigma_dot, sandwich
from .parke_taylor import parke_taylor_mhv

def _three_point_amp(sp: SpinorPoint, hels: tuple[int, ...]) -> complex:
    if len(hels) != 3:
        raise ValueError("3-point only")
    neg = [i for i,h in enumerate(hels) if h == -1]
    pos = [i for i,h in enumerate(hels) if h == +1]
    if len(neg) == 2 and len(pos) == 1:
        i, j = neg
        k = pos[0]
        return (sp.ang(i, j) ** 3) / (sp.ang(j, k) * sp.ang(k, i))
    if len(pos) == 2 and len(neg) == 1:
        i, j = pos
        k = neg[0]
        return (sp.sqr(i, j) ** 3) / (sp.sqr(j, k) * sp.sqr(k, i))
    return 0.0 + 0j

def _shift_spinors(sp: SpinorPoint, i: int, j: int, z: complex) -> SpinorPoint:
    lam = sp.lam.copy()
    lamt = sp.lamt.copy()
    lam[i] = lam[i] + z * lam[j]
    lamt[j] = lamt[j] - z * lamt[i]
    return SpinorPoint(lam=lam, lamt=lamt)

def _P_of_subset(p: np.ndarray, subset: list[int]) -> np.ndarray:
    P = np.zeros(4, dtype=np.complex128)
    for idx in subset:
        P += p[idx]
    return P

def _between(n: int, a: int, b: int) -> list[int]:
    out = [a]
    k = a
    while k != b:
        k = (k + 1) % n
        out.append(k)
    return out

def _cyclic_partitions(n: int, i: int, j: int):
    order = list(range(n))
    out = []
    for k in range(n):
        if k == i:
            continue
        L = _between(n, i, k)
        R = [x for x in order if x not in L]
        if j in R and len(L) >= 2 and len(R) >= 2:
            out.append((L, R))
    uniq = []
    seen = set()
    for L, R in out:
        key = (tuple(L), tuple(R))
        if key not in seen:
            seen.add(key)
            uniq.append((L, R))
    return uniq

def bcfw_color_ordered_tree(sp: SpinorPoint, hels: tuple[int, ...], i: int = 0, j: int = 1) -> complex:
    n = len(hels)
    if n != sp.lam.shape[0]:
        raise ValueError("hels length must match spinor point")
    if n == 3:
        return _three_point_amp(sp, hels)

    # MHV shortcut (exact) for speed + stability:
    # for color-ordered gluons with exactly two negative helicities, the Parke–Taylor formula applies.
    neg = [k for k,h in enumerate(hels) if h == -1]
    if n >= 4 and len(neg) == 2:
        return parke_taylor_mhv(sp, neg[0], neg[1])
    if i == j:
        raise ValueError("i and j must differ")

    p = sp.momenta()
    total = 0.0 + 0j

    for L_ext, R_ext in _cyclic_partitions(n, i, j):
        P = _P_of_subset(p, L_ext)
        P2 = mass2(P)

        denom = sandwich(sp.lam[j], P, sp.lamt[i])  # <j|P|i]
        if np.abs(denom) < 1e-14:
            continue
        z = -P2 / denom

        spz = _shift_spinors(sp, i, j, z)
        pz = spz.momenta()
        Pz = _P_of_subset(pz, L_ext)
        if np.abs(mass2(Pz)) > 1e-6:
            continue

        M = sigma_dot(Pz)
        row0, row1 = M[0, :], M[1, :]
        if np.linalg.norm(row0) >= np.linalg.norm(row1):
            ltP = row0.copy()
            if np.linalg.norm(ltP) < 1e-14:
                ltP = row1.copy()
        else:
            ltP = row1.copy()
            if np.linalg.norm(ltP) < 1e-14:
                ltP = row0.copy()
        jj = 0 if np.abs(ltP[0]) >= np.abs(ltP[1]) else 1
        lamP = (M[:, jj] / ltP[jj]).astype(np.complex128)
        ltP = ltP.astype(np.complex128)

        phase = 1j
        lamPm = phase * lamP
        ltPm = (-phase) * ltP

        lamL = np.vstack([spz.lam[L_ext], lamP[None, :]])
        ltL  = np.vstack([spz.lamt[L_ext], ltP[None, :]])
        spL = SpinorPoint(lam=lamL, lamt=ltL)

        lamR = np.vstack([lamPm[None, :], spz.lam[R_ext]])
        ltR  = np.vstack([ltPm[None, :], spz.lamt[R_ext]])
        spR = SpinorPoint(lam=lamR, lamt=ltR)

        for h_int in (-1, +1):
            helL = tuple(hels[k] for k in L_ext) + (h_int,)
            helR = (-h_int,) + tuple(hels[k] for k in R_ext)

            AL = _three_point_amp(spL, helL) if len(helL) == 3 else bcfw_color_ordered_tree(spL, helL, i=0, j=1)
            AR = _three_point_amp(spR, helR) if len(helR) == 3 else bcfw_color_ordered_tree(spR, helR, i=0, j=1)

            total += AL * AR / P2

    return total
