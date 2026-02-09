from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np

from .particles import Particle
from .sm import SMParams
from .process_two_lines_full import TwoLineFullTreeEngine
from .process_two_lines_two_exchange import TwoLineTwoExchangeEngine

@dataclass(frozen=True)
class TwoLineTopologySumEngine:
    params: SMParams = SMParams()
    Nc: int = 3

    def me2(self, mom: np.ndarray, legs: Sequence[Particle]) -> float:
        eng1 = TwoLineFullTreeEngine(params=self.params, Nc=self.Nc, include_internal=True)
        eng2 = TwoLineTwoExchangeEngine(params=self.params, Nc=self.Nc)
        return float(eng1.me2_exact_color_sum(mom, legs) + eng2.me2_exact_color_sum(mom, legs))
