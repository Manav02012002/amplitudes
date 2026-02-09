from __future__ import annotations
from typing import Sequence, Tuple
from .color_interference import _eval_state, _state_canon

Trace = Tuple[int, ...]
State = Tuple[Trace, ...]

def color_gram_two_lines_two_exchange(
    amp_w1: Sequence[int],
    amp_w2: Sequence[int],
    conj_w1: Sequence[int],
    conj_w2: Sequence[int],
    Nc: int = 3,
    ordering: str = "ladder",
) -> float:
    """Exact SU(Nc) Gram element for TWO exchanged gluons between two quark lines (no 3g coupling).

    ordering:
      - ladder: both lines see (b,c)
      - crossed: line1 sees (b,c), line2 sees (c,b)
    Evaluated via trace + exact Fierz recursion with delta insertions.
    """
    if ordering not in ("ladder","crossed"):
        raise ValueError("ordering must be ladder/crossed")

    b, c, bp, cp = 2100001, 2100002, 2100003, 2100004

    if ordering == "ladder":
        ins2 = (b, c); end2 = (bp, cp)
    else:
        ins2 = (c, b); end2 = (cp, bp)

    T1: Trace = tuple(amp_w1) + (b, c) + tuple(reversed(tuple(conj_w1))) + (bp, cp)
    T2: Trace = tuple(amp_w2) + tuple(ins2) + tuple(reversed(tuple(conj_w2))) + tuple(end2)

    Td1: Trace = (b, bp)
    Td2: Trace = (c, cp)

    state: State = (T1, T2, Td1, Td2)
    val = 4.0 * _eval_state(_state_canon(state), int(Nc))  # 2*2 from deltas
    return float(val.real)

def color_matrix_two_lines_two_exchange(
    basis_words: Sequence[tuple[Sequence[int], Sequence[int], str]],
    Nc: int = 3,
) -> list[list[float]]:
    m = len(basis_words)
    C = [[0.0]*m for _ in range(m)]
    for i,(w1i,w2i,ordi) in enumerate(basis_words):
        for j in range(i, m):
            w1j,w2j,ordj = basis_words[j]
            if ordi == ordj:
                cij = color_gram_two_lines_two_exchange(w1i,w2i,w1j,w2j,Nc=Nc,ordering=ordi)
            else:
                cij = 0.5*(
                    color_gram_two_lines_two_exchange(w1i,w2i,w1j,w2j,Nc=Nc,ordering="ladder") +
                    color_gram_two_lines_two_exchange(w1i,w2i,w1j,w2j,Nc=Nc,ordering="crossed")
                )
            C[i][j]=cij; C[j][i]=cij
    return C
