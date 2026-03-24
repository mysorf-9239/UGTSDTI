#!/bin/bash
# baseline.baseline.bce.ug.davis.sh
# Hybrid — baseline teacher + baseline student, BCE loss, PairGate, DAVIS.
# Usage: bash scripts/baseline.baseline.bce.ug.davis.sh

set -e
cd "$(dirname "$0")/.." || exit 1

EPOCHS="${EPOCHS:-100}"

conda run -n ugtsdti python -m ugtsdti.main \
    model=baseline.baseline.ug \
    data=tdc_davis \
    trainer=default_trainer \
    trainer.params.epochs="$EPOCHS" \
    trainer.loss.name=bce \
    run_name="baseline.baseline.bce.ug.davis"
