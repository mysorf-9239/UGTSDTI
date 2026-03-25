#!/bin/bash
# Hybrid baseline teacher + baseline student + UG on DAVIS S4 with BCE.

set -e
cd "$(dirname "$0")/.." || exit 1

EPOCHS="${EPOCHS:-100}"

conda run -n ugtsdti python -m ugtsdti.main \
    model=hybrid teacher=baseline student=baseline fusion=ug \
    data=tdc_davis_s4 \
    trainer=default_trainer \
    trainer.params.epochs="$EPOCHS" \
    loss=bce \
    run_name="baseline.baseline.ug.bce.davis_s4"
