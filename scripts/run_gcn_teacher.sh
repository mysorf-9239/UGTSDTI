#!/bin/bash
# run_gcn_teacher.sh — Experiments specifically for GCNTeacher variants.
# Usage: bash scripts/run_gcn_teacher.sh
#
# Runs:
#   1. GCN teacher standalone (only_teacher_gcn)
#   2. Hybrid: GCN teacher + baseline student + PairGate + BCE
#   3. Hybrid: GCN teacher + baseline student + PairGate + KD loss
#
# Set EPOCHS env var to override (default 100).

set -e
cd "$(dirname "$0")/.." || exit 1

EPOCHS="${EPOCHS:-100}"
DATA=tdc_davis
TRAINER=default_trainer

echo "=========================================="
echo " GCNTeacher Experiments  (epochs=$EPOCHS)"
echo "=========================================="

echo ""
echo ">>> [1/3] GCN Teacher standalone"
conda run -n ugtsdti python -m ugtsdti.main \
    model=only_teacher_gcn \
    data="$DATA" \
    trainer="$TRAINER" \
    trainer.params.epochs="$EPOCHS" \
    run_name="gcn_teacher_only"

echo ""
echo ">>> [2/3] Hybrid GCN + BCE"
conda run -n ugtsdti python -m ugtsdti.main \
    model=hybrid_gcn \
    data="$DATA" \
    trainer="$TRAINER" \
    trainer.params.epochs="$EPOCHS" \
    trainer.loss.name=bce_with_logits \
    run_name="hybrid_gcn_bce"

echo ""
echo ">>> [3/3] Hybrid GCN + KD loss (alpha=0.5)"
conda run -n ugtsdti python -m ugtsdti.main \
    model=hybrid_gcn \
    data="$DATA" \
    trainer="$TRAINER" \
    trainer.params.epochs="$EPOCHS" \
    trainer.loss.name=kd_dual_loss \
    "+trainer.loss.alpha=0.5" \
    run_name="hybrid_gcn_kd"

echo ""
echo "=========================================="
echo " GCNTeacher experiments complete."
echo "=========================================="
