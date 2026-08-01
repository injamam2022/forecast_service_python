#!/usr/bin/env bash
# Start local / EC2 Python forecast API using the project venv
set -euo pipefail
cd "$(dirname "$0")"

VENV_PY=".venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "Creating venv..."
  python3 -m venv .venv
  "$VENV_PY" -m pip install --upgrade pip
  "$VENV_PY" -m pip install -r requirements.txt
fi

# Load .env if present
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export FORECAST_PORT="${FORECAST_PORT:-8001}"
export FORECAST_HOST="${FORECAST_HOST:-0.0.0.0}"

echo "Starting forecast API on http://${FORECAST_HOST}:${FORECAST_PORT} ..."
exec "$VENV_PY" -m uvicorn main:app --host "$FORECAST_HOST" --port "$FORECAST_PORT"
