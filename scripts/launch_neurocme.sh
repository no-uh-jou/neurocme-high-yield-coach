#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_PATH="$ROOT_DIR/streamlit_app/app.py"
HOST="${NEUROCME_HOST:-127.0.0.1}"
REQUESTED_PORT="${NEUROCME_PORT:-}"
PORT="${REQUESTED_PORT:-8501}"
PORT_SCAN_LIMIT="${NEUROCME_PORT_SCAN_LIMIT:-20}"
URL=""
OPEN_BROWSER="${NEUROCME_OPEN_BROWSER:-1}"
LOG_DIR="$ROOT_DIR/agent_artifacts/launcher"
LOG_FILE="${NEUROCME_LOG_FILE:-$LOG_DIR/neurocme.log}"
PID_FILE=""
LOCK_DIR="$LOG_DIR/launch.lock"

mkdir -p "$LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >>"$LOG_FILE"
}

escape_applescript() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

show_dialog() {
  local message="$1"
  local escaped_message
  escaped_message="$(escape_applescript "$message")"
  /usr/bin/osascript -e "display dialog \"$escaped_message\" buttons {\"OK\"} default button \"OK\" with title \"NeuroCME Launcher\"" >/dev/null 2>&1 || true
}

fail() {
  local message="$1"
  log "ERROR: $message"
  echo "$message" >&2
  show_dialog "$message"
  exit 1
}

pick_python() {
  local candidate
  for candidate in \
    "$ROOT_DIR/.venv/bin/python3" \
    "$ROOT_DIR/venv/bin/python3" \
    "$(command -v python3 2>/dev/null || true)"
  do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

update_url() {
  URL="http://${HOST}:${PORT}"
}

listener_pid_for_port() {
  local port="$1"
  lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | head -n 1
}

port_in_use() {
  local port="$1"
  [[ -n "$(listener_pid_for_port "$port")" ]]
}

port_has_our_server() {
  local port="$1"
  local listener_pid
  local listener_command

  listener_pid="$(listener_pid_for_port "$port")"
  if [[ -z "$listener_pid" ]]; then
    return 1
  fi

  listener_command="$(ps -p "$listener_pid" -o command= 2>/dev/null || true)"
  [[ "$listener_command" == *"$APP_PATH"* ]]
}

select_port() {
  local candidate
  local max_port

  if [[ -n "$REQUESTED_PORT" ]]; then
    if port_in_use "$PORT" && ! port_has_our_server "$PORT"; then
      fail "Port $PORT is already in use by another app. Set NEUROCME_PORT to a different port and retry."
    fi
    update_url
    return 0
  fi

  max_port=$((PORT + PORT_SCAN_LIMIT - 1))

  for ((candidate = PORT; candidate <= max_port; candidate += 1)); do
    if port_has_our_server "$candidate"; then
      PORT="$candidate"
      update_url
      return 0
    fi
  done

  if ! port_in_use "$PORT"; then
    update_url
    return 0
  fi

  for ((candidate = PORT + 1; candidate <= max_port; candidate += 1)); do
    if ! port_in_use "$candidate"; then
      log "Port $PORT is busy; using $candidate instead."
      PORT="$candidate"
      update_url
      return 0
    fi
  done

  fail "Could not find an open port between $PORT and $max_port."
}

is_healthy() {
  curl -fsS --max-time 2 "$URL/_stcore/health" >/dev/null 2>&1
}

wait_for_health() {
  local attempts="${1:-30}"
  local attempt
  for ((attempt = 1; attempt <= attempts; attempt += 1)); do
    if is_healthy; then
      return 0
    fi
    sleep 1
  done
  return 1
}

open_browser() {
  if [[ "$OPEN_BROWSER" == "1" ]]; then
    open "$URL" >/dev/null 2>&1 || true
  fi
}

cd "$ROOT_DIR"

[[ -f "$APP_PATH" ]] || fail "Could not find Streamlit entrypoint at $APP_PATH."

PYTHON_BIN="$(pick_python)" || fail "Could not find python3. Install Python 3.9+ and retry."

if ! "$PYTHON_BIN" -c 'import streamlit' >/dev/null 2>&1; then
  fail "Streamlit is not installed for $PYTHON_BIN. Run: python3 -m pip install -e \".[core,ui,dev]\""
fi

select_port
PID_FILE="$LOG_DIR/neurocme_${PORT}.pid"

if port_has_our_server "$PORT" && is_healthy; then
  log "Reusing existing NeuroCME server at $URL"
  open_browser
  exit 0
fi

cleanup_lock() {
  rmdir "$LOCK_DIR" >/dev/null 2>&1 || true
}

if ! mkdir "$LOCK_DIR" >/dev/null 2>&1; then
  log "Another launcher is already starting NeuroCME; waiting for readiness."
  if wait_for_health 30; then
    open_browser
    exit 0
  fi
  fail "NeuroCME launch is already in progress but the server did not become ready. Check $LOG_FILE."
fi

trap cleanup_lock EXIT

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(<"$PID_FILE")"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" >/dev/null 2>&1; then
    log "Found existing launcher PID $existing_pid; waiting for readiness."
    if wait_for_health 30; then
      open_browser
      exit 0
    fi
  else
    rm -f "$PID_FILE"
  fi
fi

log "Starting NeuroCME with $PYTHON_BIN on $URL"
nohup "$PYTHON_BIN" -m streamlit run "$APP_PATH" \
  --server.address "$HOST" \
  --server.port "$PORT" \
  --server.headless true \
  --browser.gatherUsageStats false \
  >>"$LOG_FILE" 2>&1 &
new_pid=$!
printf '%s\n' "$new_pid" >"$PID_FILE"

for ((attempt = 1; attempt <= 30; attempt += 1)); do
  if is_healthy; then
    log "NeuroCME is ready at $URL"
    open_browser
    exit 0
  fi
  if ! kill -0 "$new_pid" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

fail "NeuroCME did not become ready. Check $LOG_FILE for the startup log."
