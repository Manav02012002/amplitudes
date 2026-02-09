from amplitudes.color_tensornet import self_test_fierz

def test_fierz_kernel_matches_explicit_su3():
    self_test_fierz(Nc=3, trials=10, seed=1)
