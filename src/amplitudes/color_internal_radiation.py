from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple
from functools import lru_cache

from .color_interference import _eval_state, _state_canon

Trace = Tuple[int, ...]
State = Tuple[Trace, ...]

def adjoint_chain_to_trace_combo(internal_labels: Sequence[int], b: int, c: int) -> list[tuple[complex, Trace]]:
    """Expand the adjoint-chain color tensor between endpoints b,c into a linear combination of single traces.

    For n=0 (no internal emissions), the exchanged adjoint index is conserved:
        δ^{bc} = 2 Tr(T^b T^c)
    so we return [(2, (b,c))].

    For n>=1 with ordered internal labels a1..an on the exchanged line, the color tensor is:
        (F^{a1} ... F^{an})_{bc} with (F^a)_{xy} = -i f^{axy}
    Using f^{abc} = -2i Tr([T^a,T^b] T^c) and Tr(T^a T^b)=1/2 δ^{ab},
    one obtains:
        (F^{a1}...F^{an})_{bc} = (-2)^n Tr( ad(T^{an}) ... ad(T^{a1})(T^b) T^c )
    where ad(T^a)(X) = [T^a, X].

    We implement the nested commutator expansion as a word-level recursion:
        X -> T^a X  - X T^a
    then trace(X T^c).
    """
    labs = list(internal_labels)
    if len(labs) == 0:
        return [(2.0+0j, (b, c))]
    # represent X as linear combo of words (tuples)
    combo: dict[Trace, complex] = {(b,): 1.0+0j}
    for a in labs:
        nxt: dict[Trace, complex] = {}
        for w, coeff in combo.items():
            wL = (a,) + w
            wR = w + (a,)
            nxt[wL] = nxt.get(wL, 0.0+0j) + coeff
            nxt[wR] = nxt.get(wR, 0.0+0j) - coeff
        combo = nxt
    pref = (2.0) ** len(labs)
    out: list[tuple[complex, Trace]] = []
    for w, coeff in combo.items():
        out.append((pref * coeff, w + (c,)))
    return out

def color_gram_two_lines_internal(
    amp_w1: Sequence[int],
    amp_w2: Sequence[int],
    amp_wint: Sequence[int],
    conj_w1: Sequence[int],
    conj_w2: Sequence[int],
    conj_wint: Sequence[int],
    Nc: int = 3,
) -> float:
    """Exact SU(Nc) color Gram element for two quark lines with internal-gluon radiation on the exchanged line.

    Each external gluon label 1..ng must appear exactly once in the amplitude-side words among:
      amp_w1, amp_w2, amp_wint
    and exactly once in the conjugate-side words among:
      conj_w1, conj_w2, conj_wint

    The full color factor is:
      Tr( amp_w1, b, rev(conj_w1), b' ) * Tr( amp_w2, c, rev(conj_w2), c' )
      * (adjoint_chain(amp_wint))_{bc} * (adjoint_chain(conj_wint))_{b'c'}
    summed over all colors (including b,c,b',c').

    We expand each adjoint chain into a linear combo of traces and then evaluate the resulting multi-trace
    state using the exact Fierz recursion already implemented in color_interference.
    """
    b, c, bp, cp = 1000001, 1000002, 1000003, 1000004

    T1: Trace = tuple(amp_w1) + (b,) + tuple(reversed(tuple(conj_w1))) + (bp,)
    T2: Trace = tuple(amp_w2) + (c,) + tuple(reversed(tuple(conj_w2))) + (cp,)

    comboA = adjoint_chain_to_trace_combo(amp_wint, b, c)
    comboB = adjoint_chain_to_trace_combo(conj_wint, bp, cp)

    total = 0.0 + 0j
    for ca, trA in comboA:
        for cb, trB in comboB:
            state: State = (T1, T2, trA, trB)
            total += ca.conjugate() * cb * _eval_state(_state_canon(state), int(Nc))
    return float(total.real)

def color_matrix_two_lines_internal(
    basis_triplets: Sequence[tuple[Sequence[int], Sequence[int], Sequence[int]]],
    Nc: int = 3,
) -> list[list[float]]:
    """Build exact color interference matrix for a basis of (w1,wint,w2) triplets."""
    m = len(basis_triplets)
    C = [[0.0]*m for _ in range(m)]
    for i,(w1i, winti, w2i) in enumerate(basis_triplets):
        for j in range(i, m):
            w1j, wintj, w2j = basis_triplets[j]
            cij = color_gram_two_lines_internal(w1i,w2i,winti, w1j,w2j,wintj, Nc=Nc)
            C[i][j] = cij
            C[j][i] = cij
    return C
