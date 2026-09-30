"""Contract tests for the Full Chain launcher."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_launcher_honors_python_override_and_model(tmp_path):
    """Operators can select a managed Python without editing the launcher."""
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$OLLAMA_MODEL\" \"$1\"\n"
    )
    fake_python.chmod(0o755)

    env = {
        **os.environ,
        "PYTHON_BIN": str(fake_python),
        "OLLAMA_MODEL": "launcher-contract-model",
    }
    result = subprocess.run(
        [str(REPO_ROOT / "start-fullchain.sh")],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    lines = result.stdout.splitlines()
    assert lines[0] == "launcher-contract-model"
    assert lines[1] == str(REPO_ROOT / "targets" / "rag_server_fullchain.py")


def test_launcher_default_model_matches_documented_prerequisite(tmp_path):
    """A first-time run uses the model the README tells operators to pull."""
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$OLLAMA_MODEL\" \"$1\"\n"
    )
    fake_python.chmod(0o755)

    env = {**os.environ, "PYTHON_BIN": str(fake_python)}
    env.pop("OLLAMA_MODEL", None)
    result = subprocess.run(
        [str(REPO_ROOT / "start-fullchain.sh")],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    lines = result.stdout.splitlines()
    assert lines[0] == "llama3.2"
    assert lines[1] == str(REPO_ROOT / "targets" / "rag_server_fullchain.py")
