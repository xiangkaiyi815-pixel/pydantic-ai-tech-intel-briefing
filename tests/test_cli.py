import os
import subprocess
import sys
from pathlib import Path


def test_cli_ask_returns_answer_package(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "search_assistant.cli",
            "ask",
            "What should be verified in current data?",
            "--data-dir",
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )

    assert "answer_text" in result.stdout
    assert "classification" in result.stdout
