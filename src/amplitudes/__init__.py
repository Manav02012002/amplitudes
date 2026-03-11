from __future__ import annotations
import logging as _logging

_package_logger = _logging.getLogger("amplitudes")
if not any(isinstance(handler, _logging.NullHandler) for handler in _package_logger.handlers):
    _package_logger.addHandler(_logging.NullHandler())

from .api import SpinorHelicity, BCFW, ColorDecomposition
from .bg_fermion import FermionLineBG
from .crossing import External
from .particles import Particle, gluon, quark, antiquark, vector
from .phasespace import rambo_massless
from . import symbolic
from .tree_engine import TreeEngine
from .sm import SMParams
from .vegas import vegas_integrate
from .xsec2n import xsec_2_to_n

__all__: list[str] = [
    "SpinorHelicity",
    "BCFW",
    "ColorDecomposition",
    "rambo_massless",
    "vegas_integrate",
    "Particle",
    "gluon",
    "quark",
    "antiquark",
    "vector",
    "symbolic",
    "TreeEngine",
    "FermionLineBG",
    "External",
    "SMParams",
    "xsec_2_to_n",
]
