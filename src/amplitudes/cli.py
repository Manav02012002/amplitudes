from __future__ import annotations
import typer
import numpy as np
from rich import print

from .phasespace import rambo_massless
from .spinor import SpinorPoint
from .bcfw import bcfw_color_ordered_tree
from .me import matrix_element_squared_gluons_exact_SU_N
from .me_quark import matrix_element_squared_qqbar_ng_exact_SU_N
from .xsec import xsec_gg_to_ng, xsec_qqbar_to_ng
from .xsec2n import xsec_2_to_n
from .particles import gluon, quark, antiquark, vector
from .bcfw_quark import primitive_q_qb_gluons
from .symbolic import Ang, schouten_identity, schouten_simplify, pretty
from .vegas import vegas_integrate

app = typer.Typer(add_completion=False)

def _parse_hels(s: str) -> tuple[int, ...]:
    s = s.replace(" ", "")
    out = []
    for ch in s:
        if ch == "+":
            out.append(+1)
        elif ch == "-":
            out.append(-1)
    return tuple(out)

@app.command()
def demo(n: int = 4, ecm: float = 1000.0, seed: int = 1):
    rng = np.random.default_rng(seed)
    p, w = rambo_massless(n, ecm, rng)
    sp = SpinorPoint.from_momenta(p)
    print("[bold]Momenta[/bold]")
    print(p)
    print("[bold]RAMBO weight[/bold]")
    print(w)
    print("[bold]Example brackets[/bold]")
    print("<01> =", sp.ang(0, 1))
    print("[01] =", sp.sqr(0, 1))

@app.command()
def partial(hels: str = "-- -- ++", n: int = 4, ecm: float = 1000.0, seed: int = 2):
    hel = _parse_hels(hels)
    if len(hel) != n:
        raise typer.BadParameter("hels length must match n")
    rng = np.random.default_rng(seed)
    p, _ = rambo_massless(n, ecm, rng)
    sp = SpinorPoint.from_momenta(p)
    A = bcfw_color_ordered_tree(sp, hel, i=0, j=1)
    print("[bold]Color-ordered BCFW amplitude[/bold]")
    print(A)

@app.command()
def me2(hels: str = "-- -- ++", n: int = 4, ecm: float = 1000.0, nc: int = 3, seed: int = 2):
    hel = _parse_hels(hels)
    if len(hel) != n:
        raise typer.BadParameter("hels length must match n")
    rng = np.random.default_rng(seed)
    p, _ = rambo_massless(n, ecm, rng)
    sp = SpinorPoint.from_momenta(p)
    val = matrix_element_squared_gluons_exact_SU_N(sp, hel, Nc=nc, g_s=1.0)
    print("[bold]Exact SU(Nc) color-summed |M|^2 (fixed helicities)[/bold]")
    print(val)

@app.command()
def xsec(
    hels: str = "-- -- ++",
    n: int = 4,
    ecm: float = 1000.0,
    nc: int = 3,
    neval: int = 20000,
    niter: int = 5,
    seed: int = 3,
):
    hel = _parse_hels(hels)
    if len(hel) != n:
        raise typer.BadParameter("hels length must match n")
    rng = np.random.default_rng(seed)
    s = ecm * ecm

    def integrand(_u):
        rr = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
        p, w = rambo_massless(n, ecm, rr)
        sp = SpinorPoint.from_momenta(p)
        me2 = matrix_element_squared_gluons_exact_SU_N(sp, hel, Nc=nc, g_s=1.0)
        flux = 2.0 * s
        return float((w * me2) / flux)

    I, err = vegas_integrate(integrand, ndim=6, neval=neval, niter=niter, rng=rng)
    print("[bold]σ estimate (2->n, COM, massless, fixed helicities)[/bold]")
    print("sigma =", I)
    print("stderr =", err)

@app.command()
def qamp(
    hels: str = "- - +",
    ecm: float = 1000.0,
    seed: int = 4,
):
    """Primitive amplitude A(q, g..., qb) with ordering q ... qb."""
    # hels expects like "- - + +" etc, length n, with first=q, last=qb
    hel = _parse_hels(hels)
    n = len(hel)
    rng = np.random.default_rng(seed)
    p, _ = rambo_massless(n, ecm, rng)
    sp = SpinorPoint.from_momenta(p)
    A = primitive_q_qb_gluons(sp, hel)
    print("[bold]Primitive q...qb amplitude (BCFW)[/bold]")
    print(A)

@app.command()
def sym_schouten(a: int = 0, b: int = 1, c: int = 2, d: int = 3):
    expr = schouten_identity(a,b,c,d, bracket=Ang)
    print("[bold]Schouten expression[/bold]")
    print(pretty(expr))
    simp = schouten_simplify(expr, bracket=Ang)
    print("[bold]Simplified[/bold]")
    print(pretty(simp))

@app.command()
def me2q(hels: str = "- -- ++ -", ecm: float = 1000.0, nc: int = 3, seed: int = 10):
    """Exact SU(Nc) color-summed |M|^2 for ordering [q, g..., qb]. Provide hels length n with q first and qb last."""
    hel = _parse_hels(hels)
    n = len(hel)
    if n < 3:
        raise typer.BadParameter("need at least q,g,qb")
    rng = np.random.default_rng(seed)
    p, _ = rambo_massless(n, ecm, rng)
    sp = SpinorPoint.from_momenta(p)
    val = matrix_element_squared_qqbar_ng_exact_SU_N(sp, hel, Nc=nc, g_s=1.0)
    print("me2 =", val)

@app.command()
def xsecgg(ng: int = 4, ecm: float = 1000.0, nc: int = 3, neval: int = 20000, niter: int = 5, seed: int = 1):
    """σ for gg -> ng gluons with helicity/color averages and helicity sums."""
    sig, err = xsec_gg_to_ng(ng, ecm, Nc=nc, neval=neval, niter=niter, seed=seed, sum_final_helicities=True)
    print("sigma =", sig)
    print("stderr =", err)

@app.command()
def xsecqq(ng: int = 4, ecm: float = 1000.0, nc: int = 3, neval: int = 20000, niter: int = 5, seed: int = 2):
    """σ for q qbar -> ng gluons with helicity/color averages and helicity sums."""
    sig, err = xsec_qqbar_to_ng(ng, ecm, Nc=nc, neval=neval, niter=niter, seed=seed, sum_final_helicities=True)
    print("sigma =", sig)
    print("stderr =", err)

@app.command()
def xsec2n(
    init: str = "g g",
    final: str = "g g g g",
    ecm: float = 1000.0,
    nc: int = 3,
    neval: int = 20000,
    niter: int = 5,
    seed: int = 5,
    ew: bool = False,
    flavor: str = "u",
):
    """
    General 2->n cross section with crossing-safe evaluation.

    Examples:
      amplitudes xsec2n --init "g g" --final "g g g g" --ecm 1000
      amplitudes xsec2n --init "q qb" --final "g g g g" --ecm 1000
      amplitudes xsec2n --init "q qb" --final "v g g" --ecm 1000 --ew --flavor u
    """
    def parse_side(s: str):
        toks = s.split()
        out = []
        for t in toks:
            if t == "g":
                out.append(gluon(+1))  # helicities summed internally; placeholder hel
            elif t == "q":
                out.append(quark(+1))
            elif t == "qb":
                out.append(antiquark(+1))
            elif t == "v":
                out.append(vector(+1))
            else:
                raise typer.BadParameter(f"Unknown particle token: {t}")
        return out

    init_p = parse_side(init)
    if len(init_p) != 2:
        raise typer.BadParameter("init must contain exactly 2 particles")
    fin_p = parse_side(final)

    sig, err = xsec_2_to_n(
        initial_particles=(init_p[0], init_p[1]),
        final_particles=fin_p,
        Ecm=ecm,
        Nc=nc,
        neval=neval,
        niter=niter,
        seed=seed,
        include_ew=ew,
        quark_flavor=flavor,
    )
    print("sigma =", sig)
    print("stderr =", err)

@app.command()
def diagram_demo(ecm: float = 200.0, seed: int = 3):
    """Demo: e- e+ -> mu- mu+ via gamma/Z tree amplitude (helicity-fixed)."""
    rng = np.random.default_rng(seed)
    # Simple back-to-back COM kinematics for 2->2 (massless)
    E = ecm/2.0
    p1 = np.array([E, 0.0, 0.0, +E], dtype=np.complex128)
    p2 = np.array([E, 0.0, 0.0, -E], dtype=np.complex128)
    # choose a random scattering angle for finals
    ct = rng.uniform(-1,1)
    st = np.sqrt(max(0.0,1-ct*ct))
    phi = rng.uniform(0, 2*np.pi)
    px = E*st*np.cos(phi); py = E*st*np.sin(phi); pz = E*ct
    k1 = np.array([E, px, py, pz], dtype=np.complex128)
    k2 = np.array([E, -px, -py, -pz], dtype=np.complex128)

    params = SMParams()
    init = [External(lepton(-1,"e"), p1, incoming=True), External(antilepton(+1,"e"), p2, incoming=True)]
    final = [External(lepton(-1,"mu"), k1, incoming=False), External(antilepton(+1,"mu"), k2, incoming=False)]
    amp = tree_amplitude_general(params, init, final)
    print("tree amplitude =", amp)
