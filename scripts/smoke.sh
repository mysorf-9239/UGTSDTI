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

run "_.baseline.bce._.davis_s1" \
    model=hybrid teacher=none student=baseline fusion=none \
    data=tdc_davis_s1 trainer=default_trainer loss=bce

run "baseline._.bce._.davis_s1" \
    model=hybrid teacher=baseline student=none fusion=none \
    data=tdc_davis_s1 trainer=default_trainer loss=bce

run "gcn._.bce._.davis_s2" \
    model=hybrid teacher=gcn student=none fusion=none \
    data=tdc_davis_s2 trainer=default_trainer loss=bce

run "baseline.baseline.ug.bce.davis_s4" \
    model=hybrid teacher=baseline student=baseline fusion=ug \
    data=tdc_davis_s4 trainer=default_trainer loss=bce

run "gcn.baseline.ug.bce.davis_s4" \
    model=hybrid teacher=gcn student=baseline fusion=ug \
    data=tdc_davis_s4 trainer=default_trainer loss=bce

run "gcn.baseline.ug.kd.davis_s4" \
    model=hybrid teacher=gcn student=baseline fusion=ug \
    data=tdc_davis_s4 trainer=default_trainer loss=kd loss.params.alpha=0.5

echo ""
echo "=========================================="
echo " All smoke tests passed."
echo "=========================================="
