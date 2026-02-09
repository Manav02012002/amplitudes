import sympy as sp
from amplitudes.symbolic import Ang, schouten_identity, schouten_simplify

def test_schouten_simplify_to_zero():
    expr = schouten_identity(0,1,2,3, bracket=Ang)
    simp = schouten_simplify(expr, bracket=Ang)
    assert sp.simplify(simp) == 0
