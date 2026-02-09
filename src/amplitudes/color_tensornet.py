from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class FierzKernel:
    Nc: int

    def apply(self, A: np.ndarray, B: np.ndarray) -> complex:
        """Compute sum_a Tr(A T^a) Tr(B T^a) via completeness.

        With Tr(T^a T^b)=1/2 δ^{ab}:
          sum_a (T^a)_{ij} (T^a)_{kl} = 1/2 (δ_il δ_jk - 1/Nc δ_ij δ_kl)

        Then:
          sum_a Tr(A T^a) Tr(B T^a) = 1/2 (Tr(A B) - 1/Nc Tr(A) Tr(B))
        where Tr is the ordinary matrix trace.
        """
        Nc = int(self.Nc)
        A = np.asarray(A, dtype=np.complex128)
        B = np.asarray(B, dtype=np.complex128)
        term1 = np.trace(A @ B)
        term2 = (np.trace(A) * np.trace(B)) / Nc
        return 0.5 * (term1 - term2)

def suN_generators(Nc: int) -> List[np.ndarray]:
    """Return SU(Nc) generators T^a in the fundamental rep with Tr(Ta Tb)=1/2 δ^{ab}."""
    Nc = int(Nc)
    if Nc < 2:
        raise ValueError("Nc must be >= 2")
    lambdas: List[np.ndarray] = []
    # Off-diagonal symmetric and antisymmetric matrices (generalized Gell-Mann)
    for i in range(Nc):
        for j in range(i+1, Nc):
            M = np.zeros((Nc,Nc), dtype=np.complex128)
            M[i,j] = 1.0
            M[j,i] = 1.0
            lambdas.append(M)
            M = np.zeros((Nc,Nc), dtype=np.complex128)
            M[i,j] = -1.0j
            M[j,i] =  1.0j
            lambdas.append(M)
    # Diagonal matrices
    for k in range(1, Nc):
        M = np.zeros((Nc,Nc), dtype=np.complex128)
        for i in range(k):
            M[i,i] = 1.0
        M[k,k] = -k
        M = M * np.sqrt(2.0/(k*(k+1.0)))
        lambdas.append(M)
    if len(lambdas) != Nc*Nc - 1:
        raise RuntimeError("generator construction failed")
    # Convert to T = λ/2, ensures Tr(Ta Tb)=1/2 δ^{ab}
    T = [L/2.0 for L in lambdas]
    return T

def explicit_sum_adjoints(A: np.ndarray, B: np.ndarray, Nc: int) -> complex:
    T = suN_generators(Nc)
    s = 0.0+0j
    for Ta in T:
        s += np.trace(A @ Ta) * np.trace(B @ Ta)
    return s

def self_test_fierz(Nc: int = 3, trials: int = 20, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    K = FierzKernel(Nc)
    for _ in range(trials):
        A = rng.normal(size=(Nc,Nc)) + 1j*rng.normal(size=(Nc,Nc))
        B = rng.normal(size=(Nc,Nc)) + 1j*rng.normal(size=(Nc,Nc))
        lhs = explicit_sum_adjoints(A,B,Nc)
        rhs = K.apply(A,B)
        if abs(lhs-rhs) > 1e-10 * (abs(lhs)+1e-12):
            raise AssertionError(f"Fierz mismatch Nc={Nc}: {lhs} vs {rhs}")
