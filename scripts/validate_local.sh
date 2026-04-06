#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[validate] running test suite"
python -m pytest -q

echo "[validate] building docker image"
docker build -t aitea-local:latest .

echo "[validate] starting container"
CID="$(docker run -d -p 7860:7860 aitea-local:latest)"

cleanup() {
  if docker ps -q --no-trunc | grep -q "$CID"; then
    docker stop "$CID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "[validate] waiting for API"
for i in {1..30}; do
  if curl -fsS http://127.0.0.1:7860/health >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

curl -fsS http://127.0.0.1:7860/health >/dev/null
echo "[validate] health endpoint ok"

echo "[validate] reset endpoint"
curl -fsS -X POST http://127.0.0.1:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name":"execution_easy","episode_id":"local-check"}' >/dev/null

echo "[validate] state endpoint"
curl -fsS http://127.0.0.1:7860/state >/dev/null

echo "[validate] openenv validation"
if command -v openenv >/dev/null 2>&1; then
  openenv validate openenv.yaml
elif python -m openenv --help >/dev/null 2>&1; then
  python -m openenv validate openenv.yaml
else
  echo "[validate] openenv CLI not found; skipping explicit validation command"
fi

echo "[validate] done"
