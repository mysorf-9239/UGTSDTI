#!/bin/bash
# gcn.baseline.bce.ug.davis.sh
# Hybrid — GCN teacher + baseline student, BCE loss, PairGate, DAVIS.
# Usage: bash scripts/gcn.baseline.bce.ug.davis.sh

set -e
cd "$(dirname "$0")/.." || exit 1

EPOCHS="${EPOCHS:-100}"

conda run -n ugtsdti python -m ugtsdti.main \
    model=gcn.baseline.ug \
    data=tdc_davis \
    trainer=default_trainer \
    trainer.params.epochs="$EPOCHS" \
    trainer.loss.name=bce \
    run_name="gcn.baseline.bce.ug.davis"
