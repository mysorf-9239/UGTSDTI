#!/bin/bash
# Teacher-only GCN model on DAVIS S2.

set -e
cd "$(dirname "$0")/.." || exit 1

EPOCHS="${EPOCHS:-100}"

conda run -n ugtsdti python -m ugtsdti.main \
    model=hybrid teacher=gcn student=none fusion=none \
    data=tdc_davis_s2 \
    trainer=default_trainer \
    trainer.params.epochs="$EPOCHS" \
    loss=bce \
    run_name="gcn._.bce._.davis_s2"
