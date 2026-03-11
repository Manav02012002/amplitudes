import logging
import math

import numpy as np

from amplitudes.spinor import SpinorPoint
from amplitudes.validate.diagnostics import (
    DEFAULT_MOMENTUM_WARN_THRESHOLD,
    DEFAULT_SPINOR_WARN_THRESHOLD,
    momentum_conservation_residual,
    spinor_consistency_residuals,
)


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


def test_momentum_conservation_residual_uses_supplied_ecm_scale():
    momenta = np.array(
        [
            [100.0, 0.0, 0.0, 100.0],
            [100.0, 0.0, 0.0, -100.0],
            [-100.0, 0.0, 0.0, -100.0],
            [-99.0, 0.0, 0.0, 100.0],
        ],
        dtype=np.complex128,
    )
    expected = 1.0 / 200.0
    assert math.isclose(momentum_conservation_residual(momenta, ecm=200.0), expected, rel_tol=1e-15, abs_tol=0.0)


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


def test_spinor_consistency_residuals_respect_requested_pairs():
    momenta = np.array(
        [
            [100.0, 0.0, 0.0, 100.0],
            [100.0, 60.0, 0.0, 80.0],
            [100.0, -60.0, 0.0, -80.0],
            [100.0, 80.0, 60.0, 0.0],
        ],
        dtype=np.complex128,
    )
    sp = SpinorPoint.from_momenta(momenta)
    residuals = spinor_consistency_residuals(sp, momenta, pairs=((0, 1), (2, 3)))

    assert set(residuals) == {(0, 1), (2, 3)}
    assert max(residuals.values(), default=0.0) < 1e-12


def test_diagnostics_emit_warning_only_above_threshold(caplog):
    logger = logging.getLogger("amplitudes")
    bad_momenta = np.array(
        [
            [100.0, 0.0, 0.0, 100.0],
            [100.0, 0.0, 0.0, -100.0],
            [-100.0, 0.0, 0.0, -100.0],
            [-99.0, 0.0, 0.0, 100.0],
        ],
        dtype=np.complex128,
    )
    good_momenta = np.array(
        [
            [100.0, 0.0, 0.0, 100.0],
            [100.0, 0.0, 0.0, -100.0],
            [-100.0, 0.0, 0.0, -100.0],
            [-100.0, 0.0, 0.0, 100.0],
        ],
        dtype=np.complex128,
    )

    with caplog.at_level(logging.WARNING, logger="amplitudes"):
        low_residual = momentum_conservation_residual(
            good_momenta,
            warn_threshold=DEFAULT_MOMENTUM_WARN_THRESHOLD,
            logger=logger,
        )
        high_residual = momentum_conservation_residual(
            bad_momenta,
            warn_threshold=DEFAULT_MOMENTUM_WARN_THRESHOLD,
            logger=logger,
        )

    assert low_residual == 0.0
    assert high_residual > DEFAULT_MOMENTUM_WARN_THRESHOLD
    assert "Momentum conservation residual" in caplog.text
    assert caplog.text.count("Momentum conservation residual") == 1


def test_spinor_consistency_warning_emits_only_above_threshold(caplog):
    logger = logging.getLogger("amplitudes")
    momenta = np.array(
        [
            [100.0, 0.0, 0.0, 100.0],
            [100.0, 60.0, 0.0, 80.0],
            [100.0, -60.0, 0.0, -80.0],
        ],
        dtype=np.complex128,
    )
    sp = SpinorPoint.from_momenta(momenta)

    with caplog.at_level(logging.WARNING, logger="amplitudes"):
        quiet = spinor_consistency_residuals(
            sp,
            momenta,
            warn_threshold=DEFAULT_SPINOR_WARN_THRESHOLD,
            logger=logger,
        )
        noisy = spinor_consistency_residuals(
            sp,
            momenta,
            warn_threshold=0.0,
            logger=logger,
        )

    assert max(quiet.values(), default=0.0) < DEFAULT_SPINOR_WARN_THRESHOLD
    assert "Spinor consistency residual" in caplog.text
    assert caplog.text.count("Spinor consistency residual") == 1
    assert noisy == quiet
