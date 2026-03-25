#!/bin/bash
# Teacher-only baseline embedding model on DAVIS S1.

set -e
cd "$(dirname "$0")/.." || exit 1

EPOCHS="${EPOCHS:-100}"

conda run -n ugtsdti python -m ugtsdti.main \
    model=hybrid teacher=baseline student=none fusion=none \
    data=tdc_davis_s1 \
    trainer=default_trainer \
    trainer.params.epochs="$EPOCHS" \
    loss=bce \
    run_name="baseline._.bce._.davis_s1"
