#!/usr/bin/env bash
# =============================================================================
# scripts/train_flow1_full.sh
# Full training run cho Flow 1 Teacher-Student pipeline
#
# Epochs config: configs/flow1_teacher_student.yaml → training.loop.epochs
# Override bằng env vars:
#   EPOCHS=200 BATCH_SIZE=32 bash scripts/train_flow1_full.sh
# =============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_BASE="$ROOT_DIR/configs/flow1_teacher_student.yaml"

# ── Tham số ──────────────────────────────────────────────────────────────────
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-cpu}"
SEED="${SEED:-42}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$ROOT_DIR/artifacts/flow1_full}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT_DIR/checkpoints/flow1_full}"
WANDB_PROJECT="${WANDB_PROJECT:-ugtsdti-flow1}"
WANDB_MODE="${WANDB_MODE:-online}"   # online | offline | disabled
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-15}"

# ── WandB API Key ─────────────────────────────────────────────────────────────
if [ "$WANDB_MODE" != "disabled" ]; then
    # Ưu tiên: biến môi trường WANDB_API_KEY → nếu chưa có thì hỏi user
    if [ -z "${WANDB_API_KEY:-}" ]; then
        # Kiểm tra đã login chưa
        EXISTING_KEY=$(conda run -n ugtsdti python -c \
            "import wandb; print(wandb.api.api_key or '')" 2>/dev/null || echo "")
        if [ -z "$EXISTING_KEY" ]; then
            echo ""
            echo "  WandB API key chưa được cấu hình."
            echo "  Lấy key tại: https://wandb.ai/authorize"
            echo ""
            read -rp "  Nhập WandB API key (để trống để dùng offline): " INPUT_KEY
            if [ -n "$INPUT_KEY" ]; then
                export WANDB_API_KEY="$INPUT_KEY"
                conda run -n ugtsdti wandb login "$INPUT_KEY" --relogin 2>/dev/null || true
            else
                echo "  Không có key → chuyển sang offline mode."
                WANDB_MODE="offline"
            fi
        fi
    else
        conda run -n ugtsdti wandb login "$WANDB_API_KEY" --relogin 2>/dev/null || true
    fi
fi

# ── Chuẩn bị data nếu chưa có ────────────────────────────────────────────────
DATASET_VERSION="$DATA_DIR/processed/davis/v1/dataset_version.json"
SPLIT_MANIFEST="$DATA_DIR/splits/davis/v1/v1/manifest.json"

if [ ! -f "$DATASET_VERSION" ] || [ ! -f "$SPLIT_MANIFEST" ]; then
    echo ""
    echo "============================================================"
    echo " Data chưa có — bắt đầu chuẩn bị DAVIS dataset..."
    echo "============================================================"

    RAW_TAB="$DATA_DIR/raw/davis.tab"
    if [ ! -f "$RAW_TAB" ]; then
        echo "[data] Lấy davis.tab từ git main branch..."
        mkdir -p "$DATA_DIR/raw"
        git show main:data/davis.tab > "$RAW_TAB" 2>/dev/null || {
            echo "[data] ERROR: Không tìm thấy data/davis.tab trên nhánh main."
            echo "       Chạy thủ công: git show main:data/davis.tab > data/raw/davis.tab"
            exit 1
        }
        echo "[data] OK: $(wc -l < "$RAW_TAB") dòng"
    fi

    echo "[data] Encoding và tạo splits s1-s4..."
    conda run -n ugtsdti python "$ROOT_DIR/scripts/prepare_davis_data.py"
    echo "[data] Data sẵn sàng."
else
    RECORD_COUNT=$(conda run -n ugtsdti python -c \
        "import json; d=json.load(open('$DATASET_VERSION')); print(d['record_count'])" 2>/dev/null || echo "?")
    echo "[data] Dataset đã có ($RECORD_COUNT records) — bỏ qua bước chuẩn bị."
fi

# ── Tạo temp config ───────────────────────────────────────────────────────────
TEMP_CONFIG="$(mktemp /tmp/flow1_full_XXXXXX).yaml"
trap 'rm -f "$TEMP_CONFIG"' EXIT

conda run -n ugtsdti python "$ROOT_DIR/scripts/_make_train_config.py" \
    "$CONFIG_BASE" "$TEMP_CONFIG" \
    "$DATA_DIR" "$ARTIFACTS_DIR" "$CHECKPOINT_DIR" \
    "$BATCH_SIZE" "$SEED" "$DEVICE" \
    "$WANDB_MODE" "$WANDB_PROJECT" \
    "$EPOCHS" "$EARLY_STOPPING_PATIENCE"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Flow 1 Full Training"
echo " Epochs           : $EPOCHS"
echo " Batch size        : $BATCH_SIZE"
echo " Device            : $DEVICE"
echo " Seed              : $SEED"
echo " Early stop patience: $EARLY_STOPPING_PATIENCE"
echo " WandB mode        : $WANDB_MODE  (project: $WANDB_PROJECT)"
echo " Artifacts         : $ARTIFACTS_DIR"
echo " Checkpoints       : $CHECKPOINT_DIR"
echo "============================================================"
echo ""

# ── Validate ──────────────────────────────────────────────────────────────────
echo "[train] Validating config..."
conda run -n ugtsdti python -m ugtsdti validate "$TEMP_CONFIG"

# ── Train ─────────────────────────────────────────────────────────────────────
echo "[train] Starting..."
conda run -n ugtsdti python -m ugtsdti train "$TEMP_CONFIG"

# ── Sync nếu offline ──────────────────────────────────────────────────────────
if [ "$WANDB_MODE" = "offline" ]; then
    echo ""
    echo "[wandb] Syncing offline runs..."
    conda run -n ugtsdti wandb sync --sync-all 2>&1 | tail -5
fi

echo ""
echo "[train] Done."
