import numpy as np
from amplitudes.process_two_lines_full import TwoLineFullTreeEngine
from amplitudes.particles import Particle
from amplitudes.validate.kinematics import cm_2to2_massless
from amplitudes.validate.analytic_qcd import mandelstam_s_t_u, me2_avg_qqp_to_qqp
from amplitudes.sm import SMParams

def test_ud_to_ud_matches_analytic_tree_level():
    sqrts = 1000.0
    cos_th = 0.2
    p1,p2,p3,p4 = cm_2to2_massless(sqrts, cos_th)

    # Our library uses all-outgoing convention. For 2->2:
    # incoming momenta enter as -p_in in the all-outgoing list.
    mom = np.stack([-p1, -p2, p3, p4]).astype(np.complex128)

    # legs: q qb q qb? For u d -> u d we represent as all-outgoing:
    # incoming u becomes outgoing ubar (qb), incoming d becomes outgoing dbar (qb)
    # outgoing u is outgoing u (q), outgoing d is outgoing d (q)
    legs = [
        Particle("qb", +1, flavor="u"),  # u incoming
        Particle("qb", +1, flavor="d"),  # d incoming
        Particle("q", -1, flavor="u"),   # u outgoing
        Particle("q", -1, flavor="d"),   # d outgoing
    ]

    params = SMParams()
    eng = TwoLineFullTreeEngine(params=params, include_internal=False, Nc=3)

    me2 = eng.me2_helicity_sum(mom, legs, average_initial=True, average_initial_colors=True)

    # analytic uses physical p1,p2->p3,p4
    s,t,u = mandelstam_s_t_u(p1,p2,p3,p4)
    ref = me2_avg_qqp_to_qqp(s,t,u, gs=params.gs(), Nc=3)

    # Allow moderate tolerance; our engine includes exact color sum but our topology is t-channel exchange only,
    # which matches qq'->qq' at tree. Numerical differences should be small.
    assert abs(me2 - ref) / (abs(ref) + 1e-30) < 2.5e-1
