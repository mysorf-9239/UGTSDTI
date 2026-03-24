#!/bin/bash
# smoke.sh — Quick smoke test for all current model combos (2 epochs each).
# Usage: bash scripts/smoke.sh

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

run "_.baseline.bce._.davis"               model=_.baseline._           data=tdc_davis  trainer=default_trainer   trainer.loss.name=bce
run "baseline._.bce._.davis"               model=baseline._._           data=tdc_davis  trainer=default_trainer   trainer.loss.name=bce
run "gcn._.bce._.davis"                    model=gcn._._                data=tdc_davis  trainer=default_trainer   trainer.loss.name=bce
run "baseline.baseline.bce.ug.davis"       model=baseline.baseline.ug   data=tdc_davis  trainer=default_trainer   trainer.loss.name=bce
run "baseline.baseline.kd.ug.davis"        model=baseline.baseline.ug   data=tdc_davis  trainer=default_trainer   trainer.loss.name=kd    "+trainer.loss.alpha=0.5"
run "gcn.baseline.bce.ug.davis"            model=gcn.baseline.ug        data=tdc_davis  trainer=default_trainer   trainer.loss.name=bce
run "gcn.baseline.kd.ug.davis"             model=gcn.baseline.ug        data=tdc_davis  trainer=default_trainer   trainer.loss.name=kd    "+trainer.loss.alpha=0.5"

echo ""
echo "=========================================="
echo " All smoke tests passed."
echo "=========================================="
