#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHER_SCRIPT="$ROOT_DIR/scripts/launch_neurocme.sh"
DESKTOP_APP_PATH="$HOME/Desktop/NeuroCME.app"

escape_applescript() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

[[ -x "$LAUNCHER_SCRIPT" ]] || chmod +x "$LAUNCHER_SCRIPT"

if [[ -e "$DESKTOP_APP_PATH" ]]; then
  if [[ ! -d "$DESKTOP_APP_PATH" ]]; then
    echo "Cannot install launcher because $DESKTOP_APP_PATH already exists and is not an app bundle." >&2
    exit 1
  fi
  rm -rf "$DESKTOP_APP_PATH"
fi

APPLE_LAUNCHER_PATH="$(escape_applescript "$LAUNCHER_SCRIPT")"

osacompile -o "$DESKTOP_APP_PATH" \
  -e 'on run' \
  -e "set launcherPath to \"$APPLE_LAUNCHER_PATH\"" \
  -e 'do shell script "/usr/bin/nohup " & quoted form of launcherPath & " >/dev/null 2>&1 &"' \
  -e 'end run'

echo "Created Desktop launcher at $DESKTOP_APP_PATH"
