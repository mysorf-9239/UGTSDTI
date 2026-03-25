#!/bin/bash
# Student-only baseline on DAVIS S1.

set -e
cd "$(dirname "$0")/.." || exit 1

EPOCHS="${EPOCHS:-100}"

conda run -n ugtsdti python -m ugtsdti.main \
    model=hybrid teacher=none student=baseline fusion=none \
    data=tdc_davis_s1 \
    trainer=default_trainer \
    trainer.params.epochs="$EPOCHS" \
    loss=bce \
    run_name="student_baseline_bce_davis_s1"
