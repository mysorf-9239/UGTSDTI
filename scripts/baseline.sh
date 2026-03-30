#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"
echo "[baseline-smoke] synthetic fixture workflow for CI/local smoke only."
python "$ROOT_DIR/examples/baseline.py" "$@"
