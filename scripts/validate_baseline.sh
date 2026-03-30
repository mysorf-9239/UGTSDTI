#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/baseline_reference.yaml}"

cd "$ROOT_DIR"
python -m ugtsdti validate "$CONFIG_PATH"
