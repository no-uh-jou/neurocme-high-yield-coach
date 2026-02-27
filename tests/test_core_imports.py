from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cme_core_import_without_streamlit_dependency() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    command = [
        sys.executable,
        "-c",
        "import cme_core, sys; assert 'streamlit' not in sys.modules; print('ok')",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True, env=env, cwd=ROOT)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"
