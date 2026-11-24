#!/usr/bin/env bash
# Start the API (:8000) and the Streamlit UI (:8501) natively, after checking that
# Ollama is running and the configured model is pulled. Ctrl+C stops both.
set -euo pipefail

cd "$(dirname "$0")/.."

# Same host and model the app reads (STUDY_OLLAMA_HOST / STUDY_LLM_MODEL in config.py).
OLLAMA_HOST="${STUDY_OLLAMA_HOST:-http://localhost:11434}"
MODEL="${STUDY_LLM_MODEL:-gemma2:9b}"
API_PORT=8000
UI_PORT=8501
HEALTH_TIMEOUT_S=120

API_PID=""
UI_PID=""

cleanup() {
  # Runs once on EXIT; INT and TERM are routed here through exit.
  trap - EXIT
  for pid in "$API_PID" "$UI_PID"; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
  if [ -n "$API_PID" ] || [ -n "$UI_PID" ]; then
    echo "Stopped."
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

check_ollama() {
  if ! curl -sf "$OLLAMA_HOST/api/version" >/dev/null; then
    echo "Ollama is not running. Start it with: ollama serve"
    exit 1
  fi
}

check_model() {
  local name="$MODEL"
  # A bare name such as "gemma2" means "gemma2:latest" in `ollama list`.
  case "$name" in
    *:*) ;;
    *) name="$name:latest" ;;
  esac
  local names
  names="$(ollama list 2>/dev/null | awk 'NR > 1 { print $1 }')" || true
  if ! grep -qxF "$name" <<<"$names"; then
    echo "Model $MODEL is not pulled. Run: ollama pull $MODEL"
    exit 1
  fi
}

check_port_free() {
  local port="$1"
  if curl -s -o /dev/null "http://127.0.0.1:$port/"; then
    echo "Port $port is already in use. Stop the other process first."
    exit 1
  fi
}

start_services() {
  # Run the venv binaries directly so the PIDs we kill are the real servers.
  uv sync --locked
  .venv/bin/uvicorn study_assistant.api.main:create_app --factory \
    --host 127.0.0.1 --port "$API_PORT" &
  API_PID=$!
  .venv/bin/streamlit run ui/app.py --server.port "$UI_PORT" \
    --server.address 127.0.0.1 --server.headless true &
  UI_PID=$!
}

wait_for_api() {
  local waited=0
  until curl -sf "http://127.0.0.1:$API_PORT/health" >/dev/null; do
    if ! kill -0 "$API_PID" 2>/dev/null; then
      echo "The API stopped during startup. See the messages above."
      exit 1
    fi
    if [ "$waited" -ge "$HEALTH_TIMEOUT_S" ]; then
      echo "The API did not become healthy within ${HEALTH_TIMEOUT_S}s."
      exit 1
    fi
    sleep 1
    waited=$((waited + 1))
  done
}

main() {
  check_ollama
  check_model
  check_port_free "$API_PORT"
  check_port_free "$UI_PORT"
  start_services
  wait_for_api
  echo ""
  echo "AI Study Assistant is running:"
  echo "  UI:  http://localhost:$UI_PORT"
  echo "  API: http://localhost:$API_PORT (docs: http://localhost:$API_PORT/docs)"
  echo "Press Ctrl+C to stop."
  # Keep running until either server exits, then EXIT cleanup stops the other one.
  while kill -0 "$API_PID" 2>/dev/null && kill -0 "$UI_PID" 2>/dev/null; do
    sleep 1
  done
  echo "A server exited unexpectedly."
  exit 1
}

main "$@"
