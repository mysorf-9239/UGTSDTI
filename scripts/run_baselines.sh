#!/bin/bash
# run_baselines.sh — Quick smoke tests for all available model configs (2 epochs each).
# Usage: bash scripts/run_baselines.sh
#
# Models covered:
#   Student-only:  baseline_student
#   Teacher-only:  baseline_teacher (Embedding), gcn_teacher (GCN)
#   Hybrid:        hybrid_baseline (Embedding teacher), hybrid_gcn (GCN teacher)
#
# All runs use WANDB_MODE=disabled to avoid polluting experiment logs.

set -e
cd "$(dirname "$0")/.." || exit 1

export WANDB_MODE=disabled
EPOCHS=2
DATA=tdc_davis
TRAINER=default_trainer

run() {
    local label="$1"
    local model="$2"
    local extra="${3:-}"
    echo ""
    echo ">>> [$label]"
    conda run -n ugtsdti python -m ugtsdti.main \
        model="$model" \
        data="$DATA" \
        trainer="$TRAINER" \
        trainer.params.epochs="$EPOCHS" \
        $extra
}

echo "=========================================="
echo " UGTS-DTI Baseline Smoke Tests"
echo "=========================================="

# --- Student-only ---
run "1/6 | only_student (baseline)" only_student

# --- Teacher-only ---
run "2/6 | only_teacher (Embedding, dummy)" only_teacher
run "3/6 | only_teacher_gcn (GCN, real signal)" only_teacher_gcn

# --- Hybrid: baseline teacher + BCE ---
run "4/6 | hybrid_baseline + BCE" hybrid_baseline \
    "trainer.loss.name=bce_with_logits"

# --- Hybrid: baseline teacher + KD loss ---
run "5/6 | hybrid_baseline + KD loss" hybrid_baseline \
    "trainer.loss.name=kd_dual_loss +trainer.loss.alpha=0.5"

# --- Hybrid: GCN teacher + KD loss ---
run "6/6 | hybrid_gcn + KD loss" hybrid_gcn \
    "trainer.loss.name=kd_dual_loss +trainer.loss.alpha=0.5"

echo ""
echo "=========================================="
echo " All smoke tests passed."
echo "=========================================="
