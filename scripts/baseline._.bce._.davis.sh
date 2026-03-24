#!/bin/bash
# baseline._.bce._.davis.sh
# Only Teacher — baseline embedding (dummy), BCE loss, no gate, DAVIS.
# Usage: bash scripts/baseline._.bce._.davis.sh

set -e
cd "$(dirname "$0")/.." || exit 1

EPOCHS="${EPOCHS:-100}"

conda run -n ugtsdti python -m ugtsdti.main \
    model=baseline._._ \
    data=tdc_davis \
    trainer=default_trainer \
    trainer.params.epochs="$EPOCHS" \
    trainer.loss.name=bce \
    run_name="baseline._.bce._.davis"
