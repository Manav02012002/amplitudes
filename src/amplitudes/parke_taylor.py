from __future__ import annotations
from .spinor import SpinorPoint

def parke_taylor_mhv(sp: SpinorPoint, neg1: int, neg2: int) -> complex:
    n = sp.lam.shape[0]
    num = sp.ang(neg1, neg2) ** 4
    den = 1.0 + 0j
    for k in range(n):
        den *= sp.ang(k, (k + 1) % n)
    return num / den
