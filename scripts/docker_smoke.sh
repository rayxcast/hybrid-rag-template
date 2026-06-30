#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${SMOKE_BASE_URL:-http://127.0.0.1:8000}"
TIMEOUT_SECONDS="${SMOKE_TIMEOUT_SECONDS:-180}"
STARTED_STACK=0
PYTHON_BIN="${PYTHON_BIN:-}"

if [ -z "$PYTHON_BIN" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "python3 or python is required for the smoke test" >&2
    exit 1
  fi
fi

if [ ! -f .env ]; then
  cp .env.example .env
fi

cleanup() {
  if [ "$STARTED_STACK" -eq 1 ]; then
    docker compose down
  fi
}
trap cleanup EXIT

docker compose up -d --build
STARTED_STACK=1

deadline=$((SECONDS + TIMEOUT_SECONDS))
until "$PYTHON_BIN" - "$BASE_URL" <<'PY'
import json
import sys
import urllib.request

base_url = sys.argv[1].rstrip("/")
try:
    with urllib.request.urlopen(f"{base_url}/readyz", timeout=5) as response:
        payload = json.load(response)
except Exception:
    raise SystemExit(1)
if payload.get("ready") is not True:
    raise SystemExit(1)
PY
do
  if [ "$SECONDS" -ge "$deadline" ]; then
    docker compose ps
    docker compose logs --tail=100 app reranker_service qdrant redis
    exit 1
  fi
  sleep 5
done

"$PYTHON_BIN" - "$BASE_URL" <<'PY'
import json
import sys
import urllib.request

base_url = sys.argv[1].rstrip("/")
for path in ("/healthz", "/readyz", "/status/"):
    with urllib.request.urlopen(f"{base_url}{path}", timeout=5) as response:
        payload = json.load(response)
    if payload.get("status") not in {"ok", "degraded"}:
        raise SystemExit(f"{path} returned an unexpected payload: {payload}")
print("Docker smoke test passed")
PY
