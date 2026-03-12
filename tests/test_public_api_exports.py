import importlib


def test_public_api_all_matches_expected_export_set():
    amplitudes = importlib.import_module("amplitudes")

    assert set(amplitudes.__all__) == {
        "BCFW",
        "ColorDecomposition",
        "External",
        "FermionLineBG",
        "Particle",
        "ProcessSpec",
        "SMParams",
        "SpinorHelicity",
        "TreeEngine",
        "antiquark",
        "gluon",
        "quark",
        "rambo_massless",
        "symbolic",
        "vector",
        "vegas_integrate",
        "xsec_2_to_n",
    }


def test_every_public_export_is_importable_from_top_level():
    amplitudes = importlib.import_module("amplitudes")
    namespace: dict[str, object] = {}
    exec(f"from amplitudes import {', '.join(amplitudes.__all__)}", {}, namespace)

    for name in amplitudes.__all__:
        assert namespace[name] is getattr(amplitudes, name)


def test_stale_top_level_spinor_import_is_not_advertised():
    amplitudes = importlib.import_module("amplitudes")
    assert "spinor" not in amplitudes.__all__
