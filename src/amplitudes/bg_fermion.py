from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Sequence, Tuple

from .particles import Particle
from .spinor import SpinorPoint
from .bg_currents import gluon_current_color_ordered
from .polarization import massless_vector_polarizations, massive_vector_polarizations
from .helas import slash, proj_L, proj_R, _GAMMA
from .sm import SMParams, gamma_coupling, z_couplings, w_coupling_L
from .lorentz import mass2

_eta = np.diag([1,-1,-1,-1]).astype(np.complex128)

def _ubar(psi: np.ndarray) -> np.ndarray:
    # psi is 4-component column; return row ubar = u^\dagger gamma^0
    return psi.conjugate().T @ _GAMMA[0]

def _vector_current_single(p: np.ndarray, kind: str, hel: int, params: SMParams) -> np.ndarray:
    if kind in ("g","A","v"):
        epsp, epsm = massless_vector_polarizations(p)
        return epsp if hel == +1 else epsm
    if kind == "Z":
        epsp, epsm, eps0 = massive_vector_polarizations(p, params.mZ)
        return epsp if hel == +1 else epsm
    if kind in ("W+","W-"):
        epsp, epsm, eps0 = massive_vector_polarizations(p, params.mW)
        return epsp if hel == +1 else epsm
    raise ValueError(f"Unsupported vector kind: {kind}")

def _segment_is_all_gluons(kinds: Sequence[str]) -> bool:
    return all(k == "g" for k in kinds)

def _segment_is_single_EW(kinds: Sequence[str]) -> bool:
    return len(kinds) == 1 and kinds[0] in ("A","Z","W+","W-","v")

@dataclass(frozen=True)
class FermionLineBG:
    params: SMParams = SMParams()

    def primitive_amplitude(
        self,
        mom: np.ndarray,
        legs: Sequence[Particle],
        Nc: int = 3,
    ) -> complex:
        """
        Compute a color-ordered primitive amplitude for a single fermion line:
          A(q, X1, X2, ..., Xn, qbar)
        where Xi are external vectors (gluons and/or EW bosons) with a fixed ordering.

        Assumptions for production correctness:
          - gluons interact among themselves via BG current (3/4-gluon), handled inside gluon_current_color_ordered
          - EW bosons do NOT interact with gluons in this primitive (no mixed currents); they attach directly to the fermion line
          - therefore BG partitions are allowed only for:
              * all-gluon segments
              * single EW boson segments
        This sums all tree diagrams consistent with the fixed color ordering for the QCD part.
        """
        mom = np.asarray(mom, dtype=np.complex128)
        if len(legs) < 2:
            raise ValueError("Need at least q and qbar.")
        if legs[0].kind != "q" or legs[-1].kind != "qb":
            raise ValueError("Ordering must be [q, ..., qb] for the primitive.")
        q = legs[0]
        qb = legs[-1]
        mid = list(legs[1:-1])
        n = len(mid)

# External spinors from spinor-helicity (Dirac spinor embedding, helicity-aware)
        def build_dirac_spinor(kind: str, p: np.ndarray, hel: int) -> np.ndarray:
            """Massless Dirac spinors in Weyl basis.

            Conventions (standard in spinor-helicity):
              u_-(p) = (|p>, 0),  u_+(p) = (0, |p])
              v_-(p) = (0, |p]),  v_+(p) = (|p>, 0)
            where the 2-component objects are (lam, lamt).

            This choice ensures physically consistent scaling under helicity sums.
            """
            spx = SpinorPoint.from_momenta(np.asarray([p], dtype=np.complex128))
            lam = spx.lam[0]; lamt = spx.lamt[0]
            s = np.zeros(4, dtype=np.complex128)
            if kind == "q":
                if hel == -1:
                    s[0:2] = lam
                elif hel == +1:
                    s[2:4] = lamt
                else:
                    raise ValueError("helicity must be ±1")
                return s
            if kind == "qb":
                if hel == -1:
                    s[2:4] = lamt
                elif hel == +1:
                    s[0:2] = lam
                else:
                    raise ValueError("helicity must be ±1")
                return s
            raise ValueError("kind must be 'q' or 'qb'")

        u_q = build_dirac_spinor('q', mom[0], q.hel)
        v_qb = build_dirac_spinor('qb', mom[-1], qb.hel)  # outgoing anti-fermion; for massless we can use same helicity embedding
        # Precompute kinds/hels and momenta for mid legs
        kinds = [p.kind for p in mid]
        hels = [p.hel for p in mid]

        # Precompute gluon BG currents for all-gluon contiguous segments
        # Jg[i][j] for segment i..j (inclusive) in mid indexing
        Jg = [[None]*n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                seg_kinds = kinds[i:j+1]
                if _segment_is_all_gluons(seg_kinds):
                    seg_mom = mom[1+i:1+j+1]
                    seg_hels = tuple(hels[i:j+1])
                    Jg[i][j] = gluon_current_color_ordered(seg_mom, seg_hels, g_s=self.params.gs())

        # Helper: vector current for a segment (i..j inclusive)
        def Jvec(i: int, j: int) -> np.ndarray:
            seg_kinds = kinds[i:j+1]
            if _segment_is_all_gluons(seg_kinds):
                cur = Jg[i][j]
                if cur is None:
                    raise RuntimeError("Missing gluon current")
                return cur
            if _segment_is_single_EW(seg_kinds):
                k = seg_kinds[0]
                return _vector_current_single(mom[1+i], k, hels[i], self.params)
            raise ValueError("Mixed segment not allowed (EW does not form a composite current with gluons in this primitive).")

        # Helper: coupling matrices for a segment (i..j), returning (gL,gR) and vector kind
        def coup(i: int, j: int) -> tuple[complex, complex]:
            seg_kinds = kinds[i:j+1]
            if _segment_is_all_gluons(seg_kinds):
                g = complex(self.params.gs())
                return g, g
            if _segment_is_single_EW(seg_kinds):
                k = seg_kinds[0]
                flav = q.flavor or "u"
                if k in ("A","v"):
                    g = complex(gamma_coupling(self.params, flav))
                    return g, g
                if k == "Z":
                    gL, gR = z_couplings(self.params, flav)
                    return complex(gL), complex(gR)
                if k in ("W+","W-"):
                    # infer up/down from flavor if possible; default u->d
                    up = flav if flav in ("u","c","t") else "u"
                    down = "d"
                    gL = w_coupling_L(self.params, up, down)
                    return complex(gL), 0.0+0j
            raise ValueError("Unsupported coupling segment")

        # Off-shell quark propagator: S(P) = slash(P)/P^2
        def S(P: np.ndarray) -> np.ndarray:
            den = mass2(P)
            return slash(P) / (den + 1e-30)

        # Dynamic programming for quark current after attaching first t mid-legs:
        # Q[t] is a 4-spinor including propagators up to that point (off-shell if t<n)
        Q = [np.zeros(4, dtype=np.complex128) for _ in range(n+1)]
        Q[0] = u_q.copy()

        # momentum prefix sums for quark + first t legs
        Ppref = [mom[0].copy()]
        for t in range(1, n+1):
            Ppref.append(Ppref[-1] + mom[1 + (t-1)])

        PL = proj_L()
        PR = proj_R()

        for t in range(1, n+1):
            accum = np.zeros(4, dtype=np.complex128)
            for k in range(0, t):
                # segment k..t-1 in mid indices => (k, t-1)
                seg_i = k
                seg_j = t-1
                seg_kinds = kinds[seg_i:seg_j+1]
                if not (_segment_is_all_gluons(seg_kinds) or _segment_is_single_EW(seg_kinds)):
                    continue
                J = Jvec(seg_i, seg_j)
                gL, gR = coup(seg_i, seg_j)
                Vmat = np.zeros((4,4), dtype=np.complex128)
                # gamma·J
                slashJ = J[0]*_GAMMA[0] + J[1]*_GAMMA[1] + J[2]*_GAMMA[2] + J[3]*_GAMMA[3]
                Vmat = slashJ @ (gL*PL + gR*PR)
                accum += Vmat @ Q[k]
            # attach propagator for off-shell quark after momentum Ppref[t]
            Q[t] = S(Ppref[t]) @ accum

        # Final contraction with external anti-fermion:
        # A = vbar(qb) * [ sum_k V(seg k..n-1) Q[k] ]  (no propagator on final external leg)
        out = 0.0 + 0j
        vbar = _ubar(v_qb)
        for k in range(0, n+1):
            if k == n:
                # no segment: directly connect? only possible if no mid legs; treat as 1
                if n == 0:
                    out += vbar @ Q[0]
                continue
            seg_i = k
            seg_j = n-1
            seg_kinds = kinds[seg_i:seg_j+1]
            # only allow if segment is all gluons OR single EW (for n-k ==1)
            if _segment_is_all_gluons(seg_kinds) or _segment_is_single_EW(seg_kinds):
                J = Jvec(seg_i, seg_j)
                gL, gR = coup(seg_i, seg_j)
                slashJ = J[0]*_GAMMA[0] + J[1]*_GAMMA[1] + J[2]*_GAMMA[2] + J[3]*_GAMMA[3]
                Vmat = slashJ @ (gL*PL + gR*PR)
                out += vbar @ (Vmat @ Q[k])
        return out


    def current_to_offshell_gluon(
        self,
        mom: np.ndarray,
        legs: Sequence[Particle],
        mu: int,
    ) -> complex:
        """Return the mu-component of the quark-line current emitting an off-shell gluon.

        This is the same ordered object as primitive_amplitude, but leaves the final gluon index open.
        It computes:
            J^mu = \bar v(qb) [ sum of ordered insertions ] gamma^mu [ sum of ordered insertions ] u(q)
        with all QCD gluon self-interactions included via the BG gluon current for contiguous gluon segments.
        """
        mom = np.asarray(mom, dtype=np.complex128)
        if legs[0].kind != "q" or legs[-1].kind != "qb":
            raise ValueError("Ordering must be [q, ..., qb].")

        def build_dirac_spinor(kind: str, p: np.ndarray, hel: int) -> np.ndarray:
            """Massless Dirac spinors in Weyl basis.

            Conventions:
              u_-(p) = (|p>, 0),  u_+(p) = (0, |p])
              v_-(p) = (0, |p]),  v_+(p) = (|p>, 0)
            """
            spx = SpinorPoint.from_momenta(np.asarray([p], dtype=np.complex128))
            lam = spx.lam[0]; lamt = spx.lamt[0]
            out = np.zeros(4, dtype=np.complex128)
            if kind == "q":
                if hel == -1:
                    out[0:2] = lam
                elif hel == +1:
                    out[2:4] = lamt
                else:
                    raise ValueError("helicity must be ±1")
                return out
            if kind == "qb":
                if hel == -1:
                    out[2:4] = lamt
                elif hel == +1:
                    out[0:2] = lam
                else:
                    raise ValueError("helicity must be ±1")
                return out
            raise ValueError("kind must be 'q' or 'qb'")
        q = legs[0]; qb = legs[-1]
        mid = list(legs[1:-1])
        n = len(mid)

        from .spinor import SpinorPoint
        from .bg_currents import gluon_current_color_ordered
        from .helas import proj_L, proj_R, _GAMMA
        from .lorentz import mass2
        from .helas import slash

        def build_dirac_spinor(kind: str, p: np.ndarray, hel: int) -> np.ndarray:
            """Massless Dirac spinors in Weyl basis.

            Conventions (standard in spinor-helicity):
              u_-(p) = (|p>, 0),  u_+(p) = (0, |p])
              v_-(p) = (0, |p]),  v_+(p) = (|p>, 0)
            where the 2-component objects are (lam, lamt).

            This choice ensures physically consistent scaling under helicity sums.
            """
            spx = SpinorPoint.from_momenta(np.asarray([p], dtype=np.complex128))
            lam = spx.lam[0]; lamt = spx.lamt[0]
            s = np.zeros(4, dtype=np.complex128)
            if kind == "q":
                if hel == -1:
                    s[0:2] = lam
                elif hel == +1:
                    s[2:4] = lamt
                else:
                    raise ValueError("helicity must be ±1")
                return s
            if kind == "qb":
                if hel == -1:
                    s[2:4] = lamt
                elif hel == +1:
                    s[0:2] = lam
                else:
                    raise ValueError("helicity must be ±1")
                return s
            raise ValueError("kind must be 'q' or 'qb'")

        u_q = build_dirac_spinor('q', mom[0], q.hel)
        v_qb = build_dirac_spinor('qb', mom[-1], qb.hel)
        vbar = v_qb.conjugate().T @ _GAMMA[0]

        kinds = [p.kind for p in mid]
        hels = [p.hel for p in mid]
        # gluon BG currents for contiguous segments
        Jg = [[None]*n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                if all(k=="g" for k in kinds[i:j+1]):
                    seg_mom = mom[1+i:1+j+1]
                    seg_hels = tuple(hels[i:j+1])
                    Jg[i][j] = gluon_current_color_ordered(seg_mom, seg_hels, g_s=self.params.gs())

        def Jvec(i: int, j: int) -> np.ndarray:
            if all(k=="g" for k in kinds[i:j+1]):
                cur = Jg[i][j]
                if cur is None:
                    raise RuntimeError("Missing gluon current")
                return cur
            raise ValueError("current_to_offshell_gluon supports only gluon insertions in step3.")

        PL = proj_L(); PR = proj_R()

        def S(P: np.ndarray) -> np.ndarray:
            return slash(P) / (mass2(P) + 1e-30)

        Q = [np.zeros(4, dtype=np.complex128) for _ in range(n+1)]
        Q[0] = u_q.copy()
        Ppref = [mom[0].copy()]
        for t in range(1, n+1):
            Ppref.append(Ppref[-1] + mom[1 + (t-1)])

        for t in range(1, n+1):
            accum = np.zeros(4, dtype=np.complex128)
            for k in range(0, t):
                J = Jvec(k, t-1)
                g = complex(self.params.gs())
                slashJ = J[0]*_GAMMA[0] + J[1]*_GAMMA[1] + J[2]*_GAMMA[2] + J[3]*_GAMMA[3]
                Vmat = slashJ @ (g*PL + g*PR)
                accum += Vmat @ Q[k]
            Q[t] = S(Ppref[t]) @ accum

        # Now attach open gamma^mu at the end (off-shell gluon)
        out = 0.0 + 0j
        g = complex(self.params.gs())
        Gmu = _GAMMA[mu] @ (g*PL + g*PR)
        for k in range(0, n+1):
            out += vbar @ (Gmu @ Q[k])
        return out


    def current_to_offshell_gluon_split(
        self,
        mom: np.ndarray,
        legs: Sequence[Particle],
        split: int,
        mu: int,
    ) -> complex:
        """Return mu-component of quark-line current with an exchange-point insertion.

        legs ordering: [q, X1..Xn, qb], with Xi gluons (kind='g') for this method.
        split is an integer in [0,n] indicating the exchange-point location:
            q -- (X1..Xsplit) -- [OFFSHELL GLUON] -- (Xsplit+1..Xn) -- qb

        This includes full BG gluon self-interactions within the left and right blocks separately.
        (Interactions that connect left and right blocks through gluon self-couplings belong to diagrams where
         radiation attaches to the exchanged gluon; those are handled in the next module.)
        """
        mom = np.asarray(mom, dtype=np.complex128)
        if legs[0].kind != "q" or legs[-1].kind != "qb":
            raise ValueError("Ordering must be [q, ..., qb].")
        mid = list(legs[1:-1])
        n = len(mid)
        if not (0 <= split <= n):
            raise ValueError("split out of range")
        if any(p.kind != "g" for p in mid):
            raise ValueError("current_to_offshell_gluon_split currently supports only gluon insertions.")

        from .spinor import SpinorPoint

        def build_dirac_spinor(kind: str, p: np.ndarray, hel: int) -> np.ndarray:
            """Massless Dirac spinors in Weyl basis.

            Conventions:
              u_-(p) = (|p>, 0),  u_+(p) = (0, |p])
              v_-(p) = (0, |p]),  v_+(p) = (|p>, 0)
            """
            spx = SpinorPoint.from_momenta(np.asarray([p], dtype=np.complex128))
            lam = spx.lam[0]; lamt = spx.lamt[0]
            out = np.zeros(4, dtype=np.complex128)
            if kind == "q":
                if hel == -1:
                    out[0:2] = lam
                elif hel == +1:
                    out[2:4] = lamt
                else:
                    raise ValueError("helicity must be ±1")
                return out
            if kind == "qb":
                if hel == -1:
                    out[2:4] = lamt
                elif hel == +1:
                    out[0:2] = lam
                else:
                    raise ValueError("helicity must be ±1")
                return out
            raise ValueError("kind must be 'q' or 'qb'")
        from .bg_currents import gluon_current_color_ordered
        from .helas import proj_L, proj_R, _GAMMA, slash
        from .lorentz import mass2

        def build_u(p: np.ndarray, hel: int) -> np.ndarray:
            spx = SpinorPoint.from_momenta(np.asarray([p], dtype=np.complex128))
            lam = spx.lam[0]; lamt = spx.lamt[0]
            u = np.zeros(4, dtype=np.complex128)
            if hel == -1:
                u[0:2] = lam
            elif hel == +1:
                u[2:4] = lamt
            else:
                raise ValueError("helicity must be ±1")
            return u


        def build_dirac_spinor(kind: str, p: np.ndarray, hel: int) -> np.ndarray:
            """Massless Dirac spinors in Weyl basis.

            Conventions:
              u_-(p) = (|p>, 0),  u_+(p) = (0, |p])
              v_-(p) = (0, |p]),  v_+(p) = (|p>, 0)
            """
            spx = SpinorPoint.from_momenta(np.asarray([p], dtype=np.complex128))
            lam = spx.lam[0]; lamt = spx.lamt[0]
            out = np.zeros(4, dtype=np.complex128)
            if kind == "q":
                if hel == -1:
                    out[0:2] = lam
                elif hel == +1:
                    out[2:4] = lamt
                else:
                    raise ValueError("helicity must be ±1")
                return out
            if kind == "qb":
                if hel == -1:
                    out[2:4] = lamt
                elif hel == +1:
                    out[0:2] = lam
                else:
                    raise ValueError("helicity must be ±1")
                return out
            raise ValueError("kind must be 'q' or 'qb'")
        q = legs[0]; qb = legs[-1]
        u_q = build_dirac_spinor('q', mom[0], q.hel)
        v_qb = build_dirac_spinor('qb', mom[-1], qb.hel)
        vbar = v_qb.conjugate().T @ _GAMMA[0]

        kinds = [p.kind for p in mid]
        hels = [p.hel for p in mid]

        # precompute BG gluon currents for left block and right block separately
        def build_J(block_mom: np.ndarray, block_hels: tuple[int,...]):
            m = len(block_hels)
            J = [[None]*m for _ in range(m)]
            for i in range(m):
                for j in range(i, m):
                    seg_mom = block_mom[i:j+1]
                    seg_hels = block_hels[i:j+1]
                    J[i][j] = gluon_current_color_ordered(seg_mom, tuple(seg_hels), g_s=self.params.gs())
            return J

        # Left block DP from quark
        L_mom = mom[1:1+split]
        L_hels = tuple(hels[:split])
        mL = split
        JL = build_J(L_mom, L_hels) if mL>0 else None

        PL = proj_L(); PR = proj_R()
        g = complex(self.params.gs())

        def S(P: np.ndarray) -> np.ndarray:
            return slash(P) / (mass2(P) + 1e-30)

        Q = [np.zeros(4, dtype=np.complex128) for _ in range(mL+1)]
        Q[0] = u_q.copy()
        Ppref = [mom[0].copy()]
        for t in range(1, mL+1):
            Ppref.append(Ppref[-1] + L_mom[t-1])

        for t in range(1, mL+1):
            accum = np.zeros(4, dtype=np.complex128)
            for k in range(0, t):
                if JL is None:
                    continue
                J = JL[k][t-1]
                slashJ = J[0]*_GAMMA[0] + J[1]*_GAMMA[1] + J[2]*_GAMMA[2] + J[3]*_GAMMA[3]
                Vmat = slashJ @ (g*PL + g*PR)
                accum += Vmat @ Q[k]
            Q[t] = S(Ppref[t]) @ accum

        U_left = Q[mL]

        # Right block DP from antiquark backwards (emit gluons toward exchange)
        R_mom = mom[1+split:-1]
        R_hels = tuple(hels[split:])
        mR = n - split

        # We propagate an adjoint spinor row W = vbar * ( ... ) including propagators.
        # Build BG currents on right block for segments, but note ordering from exchange to qb is X_{split+1}..X_n.
        JR = build_J(R_mom, R_hels) if mR>0 else None

        # Backward propagation: W[r] corresponds to vbar after absorbing last r gluons from the right.
        W = [None]*(mR+1)
        W[0] = vbar.copy()

        # suffix momenta from qb side: start with qb momentum
        Psuf = [mom[-1].copy()]
        for t in range(1, mR+1):
            Psuf.append(Psuf[-1] + R_mom[-t])

        for t in range(1, mR+1):
            accum = np.zeros(4, dtype=np.complex128)
            for k in range(0, t):
                # segment of last (t-k) gluons as a block
                i = mR-(t-k)
                j = mR-1
                if JR is None:
                    continue
                J = JR[i][j]
                slashJ = J[0]*_GAMMA[0] + J[1]*_GAMMA[1] + J[2]*_GAMMA[2] + J[3]*_GAMMA[3]
                Vmat = (g*PL + g*PR) @ slashJ  # acting to the left on row spinor
                accum += W[k] @ Vmat
            # attach propagator for off-shell quark going into exchange: S(Psuf[t])
            W[t] = accum @ S(Psuf[t])

        Vbar_right = W[mR]

        Gmu = _GAMMA[mu] @ (g*PL + g*PR)
        return Vbar_right @ (Gmu @ U_left)

    def current_to_two_offshell_gluons_split(
        self,
        mom: np.ndarray,
        legs: Sequence[Particle],
        split1: int,
        split2: int,
        mu: int,
        nu: int,
    ) -> complex:
        """Return the (mu,nu) component of a quark-line rank-2 current with two off-shell gluons inserted.

        Ordering: [q, X1..Xn, qb] with Xi gluons for this method.
        Splits define:
            q -- (X1..Xsplit1) -- [G(mu)] -- (Xsplit1+1..Xsplit2) -- [G(nu)] -- (Xsplit2+1..Xn) -- qb
        with 0 <= split1 <= split2 <= n.

        Includes full BG gluon self-interactions inside each contiguous gluon block.
        """
        mom = np.asarray(mom, dtype=np.complex128)
        if legs[0].kind != "q" or legs[-1].kind != "qb":
            raise ValueError("Ordering must be [q, ..., qb].")
        mid = list(legs[1:-1])
        n = len(mid)
        if not (0 <= split1 <= split2 <= n):
            raise ValueError("splits out of range")
        if any(p.kind != "g" for p in mid):
            raise ValueError("current_to_two_offshell_gluons_split supports only gluon insertions in this step.")

        from .spinor import SpinorPoint

        def build_dirac_spinor(kind: str, p: np.ndarray, hel: int) -> np.ndarray:
            """Massless Dirac spinors in Weyl basis.

            Conventions:
              u_-(p) = (|p>, 0),  u_+(p) = (0, |p])
              v_-(p) = (0, |p]),  v_+(p) = (|p>, 0)
            """
            spx = SpinorPoint.from_momenta(np.asarray([p], dtype=np.complex128))
            lam = spx.lam[0]; lamt = spx.lamt[0]
            out = np.zeros(4, dtype=np.complex128)
            if kind == "q":
                if hel == -1:
                    out[0:2] = lam
                elif hel == +1:
                    out[2:4] = lamt
                else:
                    raise ValueError("helicity must be ±1")
                return out
            if kind == "qb":
                if hel == -1:
                    out[2:4] = lamt
                elif hel == +1:
                    out[0:2] = lam
                else:
                    raise ValueError("helicity must be ±1")
                return out
            raise ValueError("kind must be 'q' or 'qb'")
        from .bg_currents import gluon_current_color_ordered
        from .helas import proj_L, proj_R, _GAMMA, slash
        from .lorentz import mass2

        def build_u(p: np.ndarray, hel: int) -> np.ndarray:
            spx = SpinorPoint.from_momenta(np.asarray([p], dtype=np.complex128))
            lam = spx.lam[0]; lamt = spx.lamt[0]
            u = np.zeros(4, dtype=np.complex128)
            if hel == -1:
                u[0:2] = lam
            elif hel == +1:
                u[2:4] = lamt
            else:
                raise ValueError("helicity must be ±1")
            return u

        q = legs[0]; qb = legs[-1]
        u_q = build_dirac_spinor('q', mom[0], q.hel)
        v_qb = build_dirac_spinor('qb', mom[-1], qb.hel)
        vbar = v_qb.conjugate().T @ _GAMMA[0]

        hels = [p.hel for p in mid]
        g = complex(self.params.gs())
        PL = proj_L(); PR = proj_R()

        def S(P: np.ndarray) -> np.ndarray:
            return slash(P) / (mass2(P) + 1e-30)

        def block_current(i: int, j: int) -> np.ndarray:
            # i..j inclusive block in mid; assumes all gluons
            seg_mom = mom[1+i:1+j+1]
            seg_hels = tuple(hels[i:j+1])
            return gluon_current_color_ordered(seg_mom, seg_hels, g_s=self.params.gs())

        # Build the three blocks: A=0..split1-1, B=split1..split2-1, C=split2..n-1 (in mid indices)
        # We'll propagate forward from q through block A, then insert Gmu, then block B, then insert Gnu, then block C, then vbar.
        # Use DP-like summation over partitions into contiguous blocks: here blocks are already contiguous, so each is a single BG current.

        P0 = mom[0].copy()
        psi = u_q.copy()

        # block A
        if split1 > 0:
            JA = block_current(0, split1-1)
            slashJA = JA[0]*_GAMMA[0] + JA[1]*_GAMMA[1] + JA[2]*_GAMMA[2] + JA[3]*_GAMMA[3]
            psi = S(P0 + np.sum(mom[1:1+split1], axis=0)) @ (slashJA @ (g*PL + g*PR) @ psi)
            P0 = P0 + np.sum(mom[1:1+split1], axis=0)

        # insert G(mu)
        Gmu = _GAMMA[mu] @ (g*PL + g*PR)
        psi = S(P0) @ (Gmu @ psi)

        # block B
        if split2 > split1:
            JB = block_current(split1, split2-1)
            slashJB = JB[0]*_GAMMA[0] + JB[1]*_GAMMA[1] + JB[2]*_GAMMA[2] + JB[3]*_GAMMA[3]
            psi = S(P0 + np.sum(mom[1+split1:1+split2], axis=0)) @ (slashJB @ (g*PL + g*PR) @ psi)
            P0 = P0 + np.sum(mom[1+split1:1+split2], axis=0)

        # insert G(nu)
        Gnu = _GAMMA[nu] @ (g*PL + g*PR)
        psi = S(P0) @ (Gnu @ psi)

        # block C
        if split2 < n:
            JC = block_current(split2, n-1)
            slashJC = JC[0]*_GAMMA[0] + JC[1]*_GAMMA[1] + JC[2]*_GAMMA[2] + JC[3]*_GAMMA[3]
            psi = S(P0 + np.sum(mom[1+split2:-1], axis=0)) @ (slashJC @ (g*PL + g*PR) @ psi)
            P0 = P0 + np.sum(mom[1+split2:-1], axis=0)

        return complex(vbar @ psi)
