from amplitudes.validate import madgraph

def test_validation_harness_imports():
    assert hasattr(madgraph, "mg5_available")
