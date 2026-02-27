#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! python3 - <<'PY' >/dev/null 2>&1
import playwright  # noqa: F401
PY
then
  echo "Missing dependency: install with python3 -m pip install -e \".[smoke]\"" >&2
  exit 1
fi

read -r -a PLAYWRIGHT_INSTALL_ARGS <<< "${PLAYWRIGHT_INSTALL_ARGS:-chromium}"
python3 -m playwright install "${PLAYWRIGHT_INSTALL_ARGS[@]}"
python3 -m pytest smoke_tests/test_playwright_ui_smoke.py -q

if [[ ! -f "agent_artifacts/last_run/playwright_ui_smoke.png" ]]; then
  echo "Playwright smoke failed: missing agent_artifacts/last_run/playwright_ui_smoke.png" >&2
  exit 1
fi
