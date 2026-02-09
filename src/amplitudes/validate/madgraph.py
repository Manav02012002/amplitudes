from __future__ import annotations
import os
import subprocess
from pathlib import Path
from typing import Sequence

def mg5_available() -> bool:
    """Return True if MG5 is configured via MG5_PATH env var."""
    p = os.environ.get("MG5_PATH")
    return bool(p) and Path(p).exists()

def export_phase_space_point_lhe(momenta, pdg_ids: Sequence[int], out_path: str) -> None:
    """Export a single phase-space point to an LHE-like event snippet.

    This is a utility for validation workflows where you compare against MG5/Sherpa/etc.

    Parameters
    ----------
    momenta:
        array-like shape (n,4) with (E,px,py,pz) in GeV, all-outgoing convention.
    pdg_ids:
        PDG codes for each external particle in the same order.
    out_path:
        output file path.
    """
    import numpy as np
    mom = np.asarray(momenta, dtype=float)
    if mom.ndim != 2 or mom.shape[1] != 4:
        raise ValueError("momenta must be shape (n,4) with (E,px,py,pz)")
    if len(pdg_ids) != mom.shape[0]:
        raise ValueError("pdg_ids length mismatch")

    lines = []
    lines.append("<event>")
    lines.append(f"{mom.shape[0]} 1 1.0 1.0 1.0 1.0")
    # LHE columns: id status mother1 mother2 color1 color2 px py pz E m lifetime spin
    for pid, p in zip(pdg_ids, mom):
        E, px, py, pz = p
        lines.append(f"{int(pid)} 1 0 0 0 0 {px:.10e} {py:.10e} {pz:.10e} {E:.10e} 0.0 0.0 9.0")
    lines.append("</event>\n")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")

def run_mg5_tree_validation(*args, **kwargs):
    """Placeholder for an MG5-driven validation runner.

    MG5 needs a full process directory, model, param card, and a way to feed a fixed phase-space point.
    Because those are installation- and workflow-specific, this function is intentionally not implemented here.

    Recommended approach:
      - use export_phase_space_point_lhe(...) to save a fixed point
      - generate process code in MG5
      - evaluate matrix element squared at that point and compare

    Set MG5_PATH to enable wiring this up locally/CI.
    """
    raise NotImplementedError("MG5 validation runner requires site-specific MG5 configuration.")
