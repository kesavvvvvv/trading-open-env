#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:7860}"
export MODEL_NAME="${MODEL_NAME:-gpt-4.1-mini}"
export HF_TOKEN="${HF_TOKEN:-}"

python inference.py
