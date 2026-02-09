from __future__ import annotations
import json, subprocess
from pathlib import Path
from typing import Sequence

def write_point_json(momenta, pdg_ids: Sequence[int], out_path: str) -> None:
    import numpy as np
    mom = np.asarray(momenta, dtype=float).tolist()
    data = {"pdg_ids": list(map(int, pdg_ids)), "momenta": mom}
    Path(out_path).write_text(json.dumps(data, indent=2), encoding="utf-8")

def run_external_me2(cmd: Sequence[str], point_json: str) -> float:
    p = subprocess.run(list(cmd)+[point_json], capture_output=True, text=True, check=True)
    return float(p.stdout.strip().split()[0])
