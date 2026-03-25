#!/bin/bash
# Hybrid GCN teacher + baseline student + UG on DAVIS S4 with KD.

set -e
cd "$(dirname "$0")/.." || exit 1

EPOCHS="${EPOCHS:-100}"

conda run -n ugtsdti python -m ugtsdti.main \
    model=hybrid teacher=gcn student=baseline fusion=ug \
    data=tdc_davis_s4 \
    trainer=default_trainer \
    trainer.params.epochs="$EPOCHS" \
    loss=kd \
    loss.params.alpha=0.5 \
    run_name="gcn.baseline.ug.kd.davis_s4"
