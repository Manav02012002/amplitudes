from __future__ import annotations
import numpy as np

def gell_mann() -> list[np.ndarray]:
    # Standard Gell-Mann matrices λ^a (a=1..8)
    zero = 0.0
    one = 1.0
    i = 1j
    l1 = np.array([[0,1,0],[1,0,0],[0,0,0]],dtype=np.complex128)
    l2 = np.array([[0,-i,0],[i,0,0],[0,0,0]],dtype=np.complex128)
    l3 = np.array([[1,0,0],[0,-1,0],[0,0,0]],dtype=np.complex128)
    l4 = np.array([[0,0,1],[0,0,0],[1,0,0]],dtype=np.complex128)
    l5 = np.array([[0,0,-i],[0,0,0],[i,0,0]],dtype=np.complex128)
    l6 = np.array([[0,0,0],[0,0,1],[0,1,0]],dtype=np.complex128)
    l7 = np.array([[0,0,0],[0,0,-i],[0,i,0]],dtype=np.complex128)
    l8 = (1/np.sqrt(3))*np.array([[1,0,0],[0,1,0],[0,0,-2]],dtype=np.complex128)
    return [l1,l2,l3,l4,l5,l6,l7,l8]

def su3_generators() -> list[np.ndarray]:
    # T^a = λ^a / 2, so Tr(T^a T^b)=1/2 δ^{ab}
    return [m/2.0 for m in gell_mann()]
