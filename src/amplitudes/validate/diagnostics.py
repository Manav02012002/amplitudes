from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import numpy as np

from ..lorentz import minkowski_dot

if TYPE_CHECKING:
    from ..spinor import SpinorPoint

DEFAULT_MOMENTUM_WARN_THRESHOLD = 1e-10
DEFAULT_SPINOR_WARN_THRESHOLD = 1e-10


def _energy_scale(momenta: np.ndarray) -> float:
    momenta = np.asarray(momenta, dtype=np.complex128)
    return max(float(np.sum(np.abs(momenta[:, 0])) / 2.0), 1e-30)


def momentum_conservation_residual(
    momenta: np.ndarray,
    ecm: float | None = None,
    *,
    warn_threshold: float | None = None,
    logger: logging.Logger | None = None,
) -> float:
    momenta = np.asarray(momenta, dtype=np.complex128)
    scale = max(float(ecm) if ecm is not None else _energy_scale(momenta), 1e-30)
    residual = float(np.linalg.norm(np.sum(momenta, axis=0)) / scale)
    if warn_threshold is not None and residual > warn_threshold and logger is not None:
        logger.warning(
            "Momentum conservation residual %.3e exceeds threshold %.3e",
            residual,
            warn_threshold,
        )
    return residual


def spinor_consistency_residuals(
    sp: SpinorPoint,
    momenta: np.ndarray,
    pairs: Sequence[tuple[int, int]] | None = None,
    s: float | None = None,
    *,
    warn_threshold: float | None = None,
    logger: logging.Logger | None = None,
) -> dict[tuple[int, int], float]:
    momenta = np.asarray(momenta, dtype=np.complex128)
    n = momenta.shape[0]
    if pairs is None:
        if n < 2:
            pairs = ()
        elif n <= 4:
            pairs = tuple((i, j) for i in range(n) for j in range(i + 1, n))
        else:
            pairs = tuple(dict.fromkeys(((0, 1), (0, n - 1), (1, n - 1))))

    scale = max(float(s) if s is not None else _energy_scale(momenta) ** 2, 1e-30)
    residuals: dict[tuple[int, int], float] = {}
    for i, j in pairs:
        lhs = sp.ang(i, j) * sp.sqr(i, j)
        rhs = 2.0 * minkowski_dot(momenta[i], momenta[j])
        residuals[(i, j)] = float(np.abs(lhs - rhs) / scale)

    if warn_threshold is not None and logger is not None:
        offenders = {pair: res for pair, res in residuals.items() if res > warn_threshold}
        if offenders:
            worst_pair = max(offenders, key=offenders.get)
            logger.warning(
                "Spinor consistency residual %.3e exceeds threshold %.3e for pair %s",
                offenders[worst_pair],
                warn_threshold,
                worst_pair,
            )
    return residuals
