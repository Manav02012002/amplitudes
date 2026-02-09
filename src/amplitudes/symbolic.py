from __future__ import annotations
import sympy as sp

def Ang(i: int, j: int) -> sp.Symbol:
    if i == j:
        return sp.Integer(0)
    a, b = (i, j)
    if a > b:
        return -Ang(b, a)
    return sp.Symbol(f"ang{a}{b}", commutative=True)

def Sqr(i: int, j: int) -> sp.Symbol:
    if i == j:
        return sp.Integer(0)
    a, b = (i, j)
    if a > b:
        return -Sqr(b, a)
    return sp.Symbol(f"sqr{a}{b}", commutative=True)

def schouten_identity(a: int, b: int, c: int, d: int, bracket=Ang):
    # <ab><cd> + <bc><ad> + <ca><bd> = 0 (and same for square brackets)
    return bracket(a,b)*bracket(c,d) + bracket(b,c)*bracket(a,d) + bracket(c,a)*bracket(b,d)

def schouten_simplify(expr: sp.Expr, bracket=Ang) -> sp.Expr:
    expr = sp.expand(expr)

    # try to reduce any explicit Schouten combinations found as addends
    # We do a heuristic pass: look for patterns matching the 3-term sum with common indices.
    terms = sp.Add.make_args(expr)
    if len(terms) < 3:
        return expr

    # Build map from a canonical "signature" to terms
    # Signature: multiset of involved bracket symbols per term
    # This is a best-effort simplifier; it won't be complete, but it's useful for learning/verification.
    remaining = list(terms)
    changed = True
    while changed:
        changed = False
        for i in range(len(remaining)):
            for j in range(i+1, len(remaining)):
                for k in range(j+1, len(remaining)):
                    tri = remaining[i] + remaining[j] + remaining[k]
                    # brute search over small index sets inferred from symbols present
                    syms = list(tri.free_symbols)
                    idx = set()
                    for s in syms:
                        name = str(s)
                        if name.startswith("ang") or name.startswith("sqr"):
                            for ch in name[3:]:
                                idx.add(int(ch))
                    idx = sorted(idx)
                    if len(idx) < 4:
                        continue
                    # try all 4-tuples
                    found = False
                    for a in idx:
                        for b in idx:
                            for c in idx:
                                for d in idx:
                                    if len({a,b,c,d}) != 4:
                                        continue
                                    ident = schouten_identity(a,b,c,d, bracket=bracket)
                                    if sp.simplify(tri - ident) == 0:
                                        # replace the triple by 0
                                        remaining.pop(k); remaining.pop(j); remaining.pop(i)
                                        changed = True
                                        found = True
                                        break
                                if found: break
                            if found: break
                        if found: break
                    if found:
                        break
                if changed:
                    break
            if changed:
                break
    return sp.simplify(sp.Add(*remaining))

def antisymmetrize(expr: sp.Expr) -> sp.Expr:
    # Our constructors already impose antisymmetry; this is mainly a wrapper for future extensions.
    return sp.simplify(expr)

def pretty(expr: sp.Expr) -> str:
    return sp.pretty(expr)
