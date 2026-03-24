#!/bin/bash
# gcn.baseline.kd.ug.davis.sh
# Hybrid — GCN teacher + baseline student, KD loss, PairGate, DAVIS.
# Usage: bash scripts/gcn.baseline.kd.ug.davis.sh

set -e
cd "$(dirname "$0")/.." || exit 1

EPOCHS="${EPOCHS:-100}"

conda run -n ugtsdti python -m ugtsdti.main \
    model=gcn.baseline.ug \
    data=tdc_davis \
    trainer=default_trainer \
    trainer.params.epochs="$EPOCHS" \
    trainer.loss.name=kd \
    "+trainer.loss.alpha=0.5" \
    run_name="gcn.baseline.kd.ug.davis"
