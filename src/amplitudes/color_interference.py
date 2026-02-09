from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence, Tuple, List

# We compute exact SU(Nc) color sums for products of traces involving pairs of generators T^a,
# using the completeness relation:
#   sum_a T^a_{ij} T^a_{kl} = 1/2 (delta_il delta_kj - (1/Nc) delta_ij delta_kl)
# and the derived trace identities:
#   sum_a Tr(T^a A T^a B) = 1/2 (Tr(A) Tr(B) - (1/Nc) Tr(AB))
#   sum_a Tr(T^a A) Tr(T^a B) = 1/2 (Tr(AB) - (1/Nc) Tr(A) Tr(B))
#
# The recursion keeps track of MULTI-TRACE expressions where a given gluon label may appear in two different traces.

Trace = Tuple[int, ...]
State = Tuple[Trace, ...]

def _rot_canon(t: Trace) -> Trace:
    if not t:
        return t
    # canonical cyclic rotation: choose lexicographically smallest rotation
    n = len(t)
    rots = [t[i:]+t[:i] for i in range(n)]
    return min(rots)

def _state_canon(state: State) -> State:
    # remove explicit empty traces? keep them, but sort; empties represent Tr(I)=Nc factor at terminal
    can = tuple(sorted((_rot_canon(t) for t in state), key=lambda x: (len(x), x)))
    return can

def _all_labels(state: State) -> set[int]:
    s=set()
    for t in state:
        for x in t:
            s.add(x)
    return s

def _find_two_occurrences(state: State, lab: int):
    loc=[]
    for ti,t in enumerate(state):
        for pi,x in enumerate(t):
            if x==lab:
                loc.append((ti,pi))
    if len(loc)!=2:
        raise ValueError(f"Label {lab} occurs {len(loc)} times (expected 2).")
    return loc[0], loc[1]

def _cut_at(trace: Trace, pos: int) -> Trace:
    # rotate so trace[pos] becomes first element
    return trace[pos:]+trace[:pos]

@lru_cache(maxsize=None)
def _eval_state(state: State, Nc: int) -> complex:
    state = _state_canon(state)

    # terminal: no labels left
    labs = _all_labels(state)
    if not labs:
        # product of Tr(I) over traces
        out = 1.0
        for t in state:
            out *= Nc
        return complex(out)

    lab = min(labs)
    (t1,p1),(t2,p2) = _find_two_occurrences(state, lab)

    if t1 == t2:
        # both occurrences in same trace: sum_a Tr( T^a Y T^a Z ) = 1/2 (Tr(Z)Tr(Y) - (1/Nc) Tr(YZ))
        tr = state[t1]
        tr_rot = _cut_at(tr, p1)
        # now lab at index 0
        # find second lab position
        j = tr_rot.index(lab, 1)
        Y = tr_rot[1:j]
        Z = tr_rot[j+1:]
        # build states
        # term1: split into two traces: Z and Y (each cyclic), other traces unchanged
        st_other = list(state[:t1] + state[t1+1:])
        st1 = tuple(st_other + [Z, Y])
        term1 = 0.5 * _eval_state(_state_canon(st1), Nc)
        # term2: -1/(2Nc) * Tr(YZ)
        YZ = Y + Z
        st2 = tuple(st_other + [YZ])
        term2 = (-0.5 / Nc) * _eval_state(_state_canon(st2), Nc)
        return term1 + term2

    else:
        # occurrences in two different traces:
        # sum_a Tr(T^a A) Tr(T^a B) = 1/2 (Tr(AB) - (1/Nc) Tr(A)Tr(B))
        trA = state[t1]
        trB = state[t2]
        # rotate each so lab is first
        trA_rot = _cut_at(trA, p1)
        trB_rot = _cut_at(trB, p2)
        # Remove leading lab; A_rem is what's after lab in cyclic order
        Arem = trA_rot[1:]
        Brem = trB_rot[1:]
        st_other = [state[i] for i in range(len(state)) if i not in (t1,t2)]
        # term1: merged trace AB
        merged = Arem + Brem
        st1 = tuple(st_other + [merged])
        term1 = 0.5 * _eval_state(_state_canon(st1), Nc)
        # term2: -1/(2Nc) * separate traces
        st2 = tuple(st_other + [Arem, Brem])
        term2 = (-0.5 / Nc) * _eval_state(_state_canon(st2), Nc)
        return term1 + term2

def color_gram_multi_quark_lines(
    amp_words: Sequence[Sequence[int]],
    conj_words: Sequence[Sequence[int]],
    Nc: int = 3,
) -> float:
    """
    Compute the exact color-sum Gram element between two multi-quark-line open-string color factors.

    Each quark line ℓ has an ordered list of gluon labels attached to it in the amplitude (amp_words[ℓ])
    and in the conjugate amplitude (conj_words[ℓ]). The color-summed interference is:

      C = sum_{all gluon colors, all quark colors} ∏_ℓ Tr( T^{a_{amp,ℓ}} ... T^{a_{conj,ℓ}} ... )

    which reduces to evaluating multi-trace contractions with the SU(Nc) completeness relation.

    We represent each external gluon by a label integer that appears exactly twice globally:
      - once in some amp_words[ℓ]
      - once in some conj_words[m]
    (not necessarily the same line).

    The trace word for line ℓ is:
      amp_words[ℓ] + reversed(conj_words[ℓ])
    and the overall expression is the product over lines of these traces.

    Returns a real nonnegative float.
    """
    if len(amp_words) != len(conj_words):
        raise ValueError("amp_words and conj_words must have same number of quark lines")
    traces: List[Trace] = []
    for Aw, Cw in zip(amp_words, conj_words):
        traces.append(tuple(Aw) + tuple(reversed(tuple(Cw))))
    state: State = tuple(traces)
    val = _eval_state(_state_canon(state), int(Nc))
    # should be real for physical Gram matrices
    return float(val.real)

def color_matrix_for_basis(
    basis: Sequence[Sequence[Sequence[int]]],
    Nc: int = 3,
) -> list[list[float]]:
    """
    Given a basis of color assignments, where basis[i] is amp_words for basis element i
    (a list of per-line ordered gluon labels),
    build the color interference matrix C_{ij} = <c_i|c_j> summed over colors.
    """
    m = len(basis)
    C = [[0.0]*m for _ in range(m)]
    for i in range(m):
        for j in range(i, m):
            cij = color_gram_multi_quark_lines(basis[i], basis[j], Nc=Nc)
            C[i][j] = cij
            C[j][i] = cij
    return C
