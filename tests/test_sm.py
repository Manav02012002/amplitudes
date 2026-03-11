import math

from amplitudes.sm import SMParams, gamma_coupling, z_couplings, w_coupling_L


def test_sm_params_gs_matches_alpha_s_definition():
    params = SMParams(alpha_s=0.25)
    assert math.isclose(params.gs(), math.sqrt(math.pi), rel_tol=1e-15, abs_tol=0.0)


def test_gamma_and_z_couplings_match_tree_level_ew_formulas():
    params = SMParams(alpha_em=1.0 / 128.0, sin2_theta_w=0.25)

    e = math.sqrt(4.0 * math.pi * params.alpha_em)
    g_z = e / (math.sqrt(params.sin2_theta_w) * math.sqrt(1.0 - params.sin2_theta_w))

    assert math.isclose(gamma_coupling(params, "u"), (2.0 / 3.0) * e, rel_tol=1e-15, abs_tol=0.0)
    assert math.isclose(gamma_coupling(params, "e"), -e, rel_tol=1e-15, abs_tol=0.0)

    g_l_u, g_r_u = z_couplings(params, "u")
    assert math.isclose(
        g_l_u,
        g_z * (0.5 - (2.0 / 3.0) * params.sin2_theta_w),
        rel_tol=1e-15,
        abs_tol=0.0,
    )
    assert math.isclose(
        g_r_u,
        g_z * (-(2.0 / 3.0) * params.sin2_theta_w),
        rel_tol=1e-15,
        abs_tol=0.0,
    )

    g_l_e, g_r_e = z_couplings(params, "e")
    assert math.isclose(
        g_l_e,
        g_z * (-0.5 + params.sin2_theta_w),
        rel_tol=1e-15,
        abs_tol=0.0,
    )
    assert math.isclose(
        g_r_e,
        g_z * params.sin2_theta_w,
        rel_tol=1e-15,
        abs_tol=0.0,
    )


def test_w_coupling_uses_default_and_custom_ckm_entries():
    params = SMParams(alpha_em=1.0 / 128.0, sin2_theta_w=0.25)
    g_w = math.sqrt(4.0 * math.pi * params.alpha_em) / (math.sqrt(2.0) * math.sqrt(params.sin2_theta_w))

    assert complex(w_coupling_L(params, "u", "d")) == complex(g_w)
    assert w_coupling_L(params, "u", "s") == 0.0 + 0.0j

    custom = SMParams(
        alpha_em=1.0 / 128.0,
        sin2_theta_w=0.25,
        Vckm={("u", "s"): 0.22 + 0.0j},
    )
    assert complex(w_coupling_L(custom, "u", "s")) == complex(g_w * 0.22)
