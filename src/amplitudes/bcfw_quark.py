from __future__ import annotations
import numpy as np
from .spinor import SpinorPoint, mass2, sigma_dot, sandwich

def _three_point_q_qb_g(sp: SpinorPoint, hels: tuple[int, ...], kinds: tuple[str, ...]) -> complex:
    # expects ordering (q, g, qb) up to cyclic rotation within this 3-point object
    # Non-zero primitive 3pt amplitudes (all outgoing):
    # A3(q^-, g^-, qb^+) = <q g>^2 / <q qb>
    # A3(q^+, g^+, qb^-) = [q g]^2 / [q qb]
    if len(hels) != 3 or len(kinds) != 3:
        raise ValueError("3-point only")
    # find positions
    try:
        iq = kinds.index("q")
        iqb = kinds.index("qb")
        ig = kinds.index("g")
    except ValueError:
        return 0.0 + 0j

    hq, hqb, hg = hels[iq], hels[iqb], hels[ig]
    if (hq, hg, hqb) == (-1, -1, +1):
        return (sp.ang(iq, ig) ** 2) / (sp.ang(iq, iqb))
    if (hq, hg, hqb) == (+1, +1, -1):
        return (sp.sqr(iq, ig) ** 2) / (sp.sqr(iq, iqb))
    return 0.0 + 0j

def _three_point_gluon(sp: SpinorPoint, hels: tuple[int, ...]) -> complex:
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

def bcfw_tree_one_quark_line(sp: SpinorPoint, hels: tuple[int, ...], kinds: tuple[str, ...], i: int, j: int) -> complex:
    """
    BCFW recursion for a *single* quark line primitive amplitude with arbitrary gluons:
      ordering contains exactly one 'q' and one 'qb' (massless), rest 'g'.
    Uses <i j] shift on legs i and j.

    This is a numeric recursion intended for production use; it assumes complex kinematics as needed.
    """
    n = len(hels)
    if n != sp.lam.shape[0] or n != len(kinds):
        raise ValueError("shape mismatch")
    if i == j:
        raise ValueError("i and j must differ")

    if n == 3:
        if kinds.count("g") == 3:
            return _three_point_gluon(sp, hels)
        return _three_point_q_qb_g(sp, hels, kinds)

    p = sp.momenta()
    total = 0.0 + 0j

    for L_ext, R_ext in _cyclic_partitions(n, i, j):
        P = _P_of_subset(p, L_ext)
        P2 = mass2(P)
        denom = sandwich(sp.lam[j], P, sp.lamt[i])
        if np.abs(denom) < 1e-14:
            continue
        z = -P2 / denom

        spz = _shift_spinors(sp, i, j, z)
        pz = spz.momenta()
        Pz = _P_of_subset(pz, L_ext)
        if np.abs(mass2(Pz)) > 1e-6:
            continue

        # internal momentum spinors
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

        # Determine which particle type is exchanged across this channel
        kindsL = tuple(kinds[k] for k in L_ext)
        kindsR = tuple(kinds[k] for k in R_ext)
        has_q_L = "q" in kindsL
        has_qb_L = "qb" in kindsL
        has_q_R = "q" in kindsR
        has_qb_R = "qb" in kindsR

        # Build subproblems with internal leg inserted
        # Convention: Left ends with internal; Right begins with internal
        # For gluon internal: right sees opposite helicity (like in pure-gluon recursion).
        # For fermion internal: right sees opposite helicity as well (all outgoing).
        def build_sub(sp_base: SpinorPoint, ext_indices: list[int], add_front: bool, lam_int, lt_int):
            if add_front:
                lam = np.vstack([lam_int[None, :], sp_base.lam[ext_indices]])
                lamt = np.vstack([lt_int[None, :], sp_base.lamt[ext_indices]])
            else:
                lam = np.vstack([sp_base.lam[ext_indices], lam_int[None, :]])
                lamt = np.vstack([sp_base.lamt[ext_indices], lt_int[None, :]])
            return SpinorPoint(lam=lam, lamt=lamt)

        # choose a consistent (-P) representation for the other side
        phase = 1j
        lamPm = phase * lamP
        ltPm = (-phase) * ltP

        if (has_q_L and has_qb_L) or (has_q_R and has_qb_R):
            # internal is gluon
            spL = build_sub(spz, L_ext, add_front=False, lam_int=lamP, lt_int=ltP)
            spR = build_sub(spz, R_ext, add_front=True,  lam_int=lamPm, lt_int=ltPm)

            for h_int in (-1, +1):
                helL = tuple(hels[k] for k in L_ext) + (h_int,)
                helR = (-h_int,) + tuple(hels[k] for k in R_ext)

                kindsL2 = kindsL + ("g",)
                kindsR2 = ("g",) + kindsR

                AL = bcfw_tree_one_quark_line(spL, helL, kindsL2, i=0, j=1) if len(helL) > 3 else (
                    _three_point_gluon(spL, helL) if kindsL2.count("g")==3 else _three_point_q_qb_g(spL, helL, kindsL2)
                )
                AR = bcfw_tree_one_quark_line(spR, helR, kindsR2, i=0, j=1) if len(helR) > 3 else (
                    _three_point_gluon(spR, helR) if kindsR2.count("g")==3 else _three_point_q_qb_g(spR, helR, kindsR2)
                )
                total += AL * AR / P2
        else:
            # internal is fermion (quark propagator)
            # one side has q, other has qb
            # Left: append internal as qb if left has q (and vice versa) so each subamplitude has one q and one qb.
            left_has_q = has_q_L
            if left_has_q:
                kind_int_L = "qb"
                kind_int_R = "q"
            else:
                kind_int_L = "q"
                kind_int_R = "qb"

            spL = build_sub(spz, L_ext, add_front=False, lam_int=lamP, lt_int=ltP)
            spR = build_sub(spz, R_ext, add_front=True,  lam_int=lamPm, lt_int=ltPm)

            for h_int in (-1, +1):
                helL = tuple(hels[k] for k in L_ext) + (h_int,)
                helR = (-h_int,) + tuple(hels[k] for k in R_ext)

                kindsL2 = kindsL + (kind_int_L,)
                kindsR2 = (kind_int_R,) + kindsR

                AL = bcfw_tree_one_quark_line(spL, helL, kindsL2, i=0, j=1) if len(helL) > 3 else (
                    _three_point_q_qb_g(spL, helL, kindsL2) if len(helL)==3 else 0.0+0j
                )
                AR = bcfw_tree_one_quark_line(spR, helR, kindsR2, i=0, j=1) if len(helR) > 3 else (
                    _three_point_q_qb_g(spR, helR, kindsR2) if len(helR)==3 else 0.0+0j
                )
                total += AL * AR / P2

    return total

def primitive_q_qb_gluons(sp: SpinorPoint, hels: tuple[int, ...], i_shift: int | None = None, j_shift: int | None = None) -> complex:
    """
    Convenience wrapper for primitive A(q, g..., qb) with ordering [0..n-1] and
    kinds = ('q', 'g', ..., 'g', 'qb').
    Shift choice defaults to first and last gluon if available, else (0,1).
    """
    n = len(hels)
    if n < 3:
        raise ValueError("need at least 3 legs")
    kinds = ("q",) + ("g",) * (n - 2) + ("qb",)
    if i_shift is None or j_shift is None:
        if n >= 5:
            i_shift, j_shift = 1, n - 2
        else:
            i_shift, j_shift = 0, 1
    return bcfw_tree_one_quark_line(sp, hels, kinds, i=i_shift, j=j_shift)
