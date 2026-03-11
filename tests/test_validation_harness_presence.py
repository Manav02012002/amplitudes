import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from amplitudes.validate import madgraph


def test_validation_harness_imports():
    assert hasattr(madgraph, "mg5_available")


def test_mg5_runner_accepts_install_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    mg5_exec = tmp_path / "MG5_aMC_v3_5_0" / "bin" / "mg5_aMC"
    mg5_exec.parent.mkdir(parents=True)
    mg5_exec.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("MG5_PATH", str(mg5_exec.parent.parent))

    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((cmd, kwargs))
        return SimpleNamespace(stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(madgraph.subprocess, "run", fake_run)

    result = madgraph.run_mg5_tree_validation("u u~ > g g", output_dir=str(tmp_path / "run"))

    assert result.mg5_executable == str(mg5_exec)
    assert result.process == "u u~ > g g"
    assert result.returncode == 0
    assert Path(result.command_card).exists()
    assert Path(result.command_card).read_text(encoding="utf-8").splitlines() == [
        "set automatic_html_opening False",
        "import model sm",
        "generate u u~ > g g",
        f"output {Path(result.process_dir)} -f",
    ]
    assert calls == [
        (
            [str(mg5_exec), result.command_card],
            {
                "check": False,
                "capture_output": True,
                "text": True,
                "timeout": 300.0,
            },
        )
    ]


def test_mg5_runner_accepts_direct_executable_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    mg5_exec = tmp_path / "mg5_aMC"
    mg5_exec.write_text("#!/bin/sh\n", encoding="utf-8")

    def fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(stdout="generated", stderr="", returncode=0)

    monkeypatch.setattr(madgraph.subprocess, "run", fake_run)

    result = madgraph.run_mg5_tree_validation(
        "e+ e- > mu+ mu-",
        model="sm-no_b_mass",
        mg5_path=str(mg5_exec),
        output_dir=str(tmp_path / "job"),
        extra_commands=("display diagrams",),
        timeout=12.5,
    )

    card = Path(result.command_card).read_text(encoding="utf-8").splitlines()
    assert card == [
        "set automatic_html_opening False",
        "import model sm-no_b_mass",
        "generate e+ e- > mu+ mu-",
        f"output {Path(result.process_dir)} -f",
        "display diagrams",
    ]
    assert result.returncode == 0
    assert result.stdout == "generated"


@pytest.mark.skipif("MG5_PATH" not in os.environ, reason="MG5_PATH not set")
def test_mg5_runner_integration_generates_process_directory(tmp_path: Path):
    result = madgraph.run_mg5_tree_validation("u u~ > g g", output_dir=str(tmp_path))

    assert result.returncode == 0
    assert Path(result.command_card).exists()
    assert Path(result.process_dir).exists()
