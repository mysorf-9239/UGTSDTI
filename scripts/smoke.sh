#!/bin/bash
# smoke.sh — Quick smoke test for the canonical slot-based experiment set.

set -e
cd "$(dirname "$0")/.." || exit 1

export WANDB_MODE=disabled
EPOCHS=2

run() {
    local label="$1"; shift
    echo ""
    echo ">>> [$label]"
    conda run -n ugtsdti python -m ugtsdti.main \
        trainer.params.epochs="$EPOCHS" \
        "$@"
}

echo "=========================================="
echo " UGTS-DTI Smoke Tests  (epochs=$EPOCHS)"
echo "=========================================="

run "student_baseline_bce_davis_s1" \
    model=hybrid teacher=none student=baseline fusion=none \
    data=tdc_davis_s1 trainer=default_trainer loss=bce
run "teacher_baseline_bce_davis_s1" \
    model=hybrid teacher=baseline student=none fusion=none \
    data=tdc_davis_s1 trainer=default_trainer loss=bce
run "teacher_gcn_bce_davis_s2" \
    model=hybrid teacher=gcn student=none fusion=none \
    data=tdc_davis_s2 trainer=default_trainer loss=bce
run "hybrid_baseline_baseline_ug_bce_davis_s4" \
    model=hybrid teacher=baseline student=baseline fusion=ug \
    data=tdc_davis_s4 trainer=default_trainer loss=bce
run "hybrid_gcn_baseline_ug_bce_davis_s4" \
    model=hybrid teacher=gcn student=baseline fusion=ug \
    data=tdc_davis_s4 trainer=default_trainer loss=bce
run "hybrid_gcn_baseline_ug_kd_davis_s4" \
    model=hybrid teacher=gcn student=baseline fusion=ug \
    data=tdc_davis_s4 trainer=default_trainer loss=kd loss.params.alpha=0.5

echo ""
echo "=========================================="
echo " All smoke tests passed."
echo "=========================================="
