import logging

import numpy as np

from amplitudes.spinor import SpinorPoint
from amplitudes.validate.diagnostics import momentum_conservation_residual, spinor_consistency_residuals


def test_momentum_conservation_residual_vanishes_for_balanced_configuration():
    momenta = np.array(
        [
            [100.0, 0.0, 0.0, 100.0],
            [100.0, 0.0, 0.0, -100.0],
            [-100.0, 0.0, 0.0, -100.0],
            [-100.0, 0.0, 0.0, 100.0],
        ],
        dtype=np.complex128,
    )
    assert momentum_conservation_residual(momenta) == 0.0


def test_spinor_consistency_residuals_are_small_for_massless_point():
    momenta = np.array(
        [
            [100.0, 0.0, 0.0, 100.0],
            [100.0, 60.0, 0.0, 80.0],
            [100.0, -60.0, 0.0, -80.0],
        ],
        dtype=np.complex128,
    )
    sp = SpinorPoint.from_momenta(momenta)
    residuals = spinor_consistency_residuals(sp, momenta)
    assert max(residuals.values(), default=0.0) < 1e-12


def test_diagnostics_emit_warning_only_above_threshold(caplog):
    logger = logging.getLogger("amplitudes")
    momenta = np.array(
        [
            [100.0, 0.0, 0.0, 100.0],
            [100.0, 0.0, 0.0, -99.0],
        ],
        dtype=np.complex128,
    )

    with caplog.at_level(logging.WARNING, logger="amplitudes"):
        residual = momentum_conservation_residual(momenta, warn_threshold=1e-12, logger=logger)

    assert residual > 0.0
    assert "Momentum conservation residual" in caplog.text
