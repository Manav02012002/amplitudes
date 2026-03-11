from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class MG5RunResult:
    mg5_executable: str
    process: str
    model: str
    process_dir: str
    command_card: str
    stdout: str
    stderr: str
    returncode: int


def _resolve_mg5_executable(mg5_path: str | None = None) -> Path:
    raw = mg5_path or os.environ.get("MG5_PATH")
    if not raw:
        raise FileNotFoundError("MG5_PATH is not set.")

    candidate = Path(raw).expanduser()
    if candidate.is_dir():
        candidate = candidate / "bin" / "mg5_aMC"

    if not candidate.exists():
        raise FileNotFoundError(f"MadGraph executable not found: {candidate}")
    if not candidate.is_file():
        raise ValueError(f"MadGraph path is not a file: {candidate}")
    return candidate


def mg5_available() -> bool:
    """Return True if MG5 is configured via MG5_PATH env var."""
    try:
        _resolve_mg5_executable()
    except (FileNotFoundError, ValueError):
        return False
    return True


def export_phase_space_point_lhe(
    momenta: Sequence[Sequence[float]],
    pdg_ids: Sequence[int],
    out_path: str,
) -> None:
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
    for pid, p in zip(pdg_ids, mom):
        E, px, py, pz = p
        lines.append(f"{int(pid)} 1 0 0 0 0 {px:.10e} {py:.10e} {pz:.10e} {E:.10e} 0.0 0.0 9.0")
    lines.append("</event>\n")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def run_mg5_tree_validation(
    process: str,
    *,
    model: str = "sm",
    output_dir: str | None = None,
    mg5_path: str | None = None,
    extra_commands: Sequence[str] = (),
    timeout: float | None = 300.0,
) -> MG5RunResult:
    """
    Generate a fixed tree-level MadGraph5 process directory in batch mode.

    Parameters
    ----------
    process:
        MG5 process string, e.g. ``"u u~ > g g"``.
    model:
        MG5 model name passed via ``import model``.
    output_dir:
        Directory in which the MG5 process directory and command card are created.
        If omitted, a temporary directory is created and retained.
    mg5_path:
        Optional explicit MG5 executable path or MG5 install directory. Falls back to ``MG5_PATH``.
    extra_commands:
        Additional MG5 batch commands appended after ``output``.
    timeout:
        Timeout passed to ``subprocess.run`` in seconds.
    """
    mg5_executable = _resolve_mg5_executable(mg5_path)
    base_dir = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="amplitudes-mg5-"))
    base_dir.mkdir(parents=True, exist_ok=True)

    process_dir = base_dir / "mg5_process"
    command_card = base_dir / "mg5_commands.mg5"
    commands = [
        "set automatic_html_opening False",
        f"import model {model}",
        f"generate {process}",
        f"output {process_dir} -f",
        *extra_commands,
    ]
    command_card.write_text("\n".join(commands) + "\n", encoding="utf-8")

    completed = subprocess.run(
        [str(mg5_executable), str(command_card)],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    return MG5RunResult(
        mg5_executable=str(mg5_executable),
        process=process,
        model=model,
        process_dir=str(process_dir),
        command_card=str(command_card),
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )
