#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/baseline_reference.yaml}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$ROOT_DIR/artifacts}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT_DIR/checkpoints}"

if [[ ! -f "$DATA_DIR/processed/davis/v1/dataset_version.json" ]]; then
  echo "Missing baseline artifacts under $DATA_DIR"
  echo "Run: python $ROOT_DIR/scripts/prepare_baseline_fixture.py --data-dir \"$DATA_DIR\""
  exit 1
fi

TMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/ugtsdti_baseline_train.XXXXXX.yaml")"
trap 'rm -f "$TMP_CONFIG"' EXIT

python - "$CONFIG_PATH" "$TMP_CONFIG" "$DATA_DIR" "$ARTIFACTS_DIR" "$CHECKPOINT_DIR" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
tmp_config = Path(sys.argv[2])
data_dir = sys.argv[3]
artifacts_dir = sys.argv[4]
checkpoint_dir = sys.argv[5]

cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
cfg.setdefault("runtime", {})
cfg["runtime"]["data_dir"] = data_dir
cfg["runtime"]["artifacts_dir"] = artifacts_dir
cfg["runtime"]["checkpoint_dir"] = checkpoint_dir
tmp_config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY

cd "$ROOT_DIR"
python -m ugtsdti train "$TMP_CONFIG"
