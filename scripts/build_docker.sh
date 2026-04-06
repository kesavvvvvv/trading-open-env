#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

IMAGE_NAME="${1:-aitea-local:latest}"

docker build -t "$IMAGE_NAME" .
echo "[build] created image: $IMAGE_NAME"
