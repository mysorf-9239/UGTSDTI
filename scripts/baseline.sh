#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/profiles/baseline_local.yaml}"

cd "$ROOT_DIR"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing config: $CONFIG_PATH" >&2
  exit 1
fi

echo "[baseline] config-driven run; train/eval stay artifact-backed and may bootstrap artifacts if data.source.auto_prepare=true."
python -m ugtsdti validate "$CONFIG_PATH"

TRAIN_LOG="$(mktemp)"
TEMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/ugtsdti-baseline-real.XXXXXX.yaml")"
cleanup() {
  rm -f "$TRAIN_LOG" "$TEMP_CONFIG"
}
trap cleanup EXIT

python -m ugtsdti train "$CONFIG_PATH" | tee "$TRAIN_LOG"

CHECKPOINT_PATH="$(
python - "$TRAIN_LOG" <<'PY'
import json
import sys
from pathlib import Path

payload = Path(sys.argv[1]).read_text(encoding="utf-8")
summary = {}
for line in reversed(payload.splitlines()):
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        candidate = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(candidate, dict):
        summary = candidate
        break
print(summary.get("best_checkpoint") or summary.get("checkpoint") or "")
PY
)"

if [[ -z "$CHECKPOINT_PATH" ]]; then
  echo "Could not determine checkpoint path from train summary." >&2
  exit 1
fi

python - "$CONFIG_PATH" "$CHECKPOINT_PATH" "$TEMP_CONFIG" <<'PY'
import sys
from pathlib import Path

import yaml
from ugtsdti.config.loader import ConfigLoader

config_path = Path(sys.argv[1])
checkpoint_path = sys.argv[2]
destination = Path(sys.argv[3])
cfg = ConfigLoader().load(config_path)
cfg.setdefault("runtime", {})
cfg["runtime"]["checkpoint_path"] = checkpoint_path
destination.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY

python -m ugtsdti eval "$TEMP_CONFIG"
