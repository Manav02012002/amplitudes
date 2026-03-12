from __future__ import annotations
from dataclasses import dataclass
from .crossing import External
from .process import ProcessSpec, matrix_element_squared
from .sm import SMParams

@dataclass(frozen=True)
class WidthScheme:
    scheme: str = "fixed"

@dataclass(frozen=True)
class TreeEngine:
    params: SMParams = SMParams()
    Nc: int = 3
    width_scheme: WidthScheme = WidthScheme()

    def me2(self, initial: list[External], final: list[External], sum_helicities: bool = True, average_initial: bool = True) -> float:
        proc = ProcessSpec(initial=tuple(initial), final=tuple(final), Nc=self.Nc, params=self.params)
        return matrix_element_squared(proc, sum_helicities=sum_helicities, average_initial=average_initial)
