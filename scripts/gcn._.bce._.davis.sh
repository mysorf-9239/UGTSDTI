#!/bin/bash
# gcn._.bce._.davis.sh
# Only Teacher — GCN encoder, BCE loss, no gate, DAVIS.
# Usage: bash scripts/gcn._.bce._.davis.sh

set -e
cd "$(dirname "$0")/.." || exit 1

EPOCHS="${EPOCHS:-100}"

conda run -n ugtsdti python -m ugtsdti.main \
    model=gcn._._ \
    data=tdc_davis \
    trainer=default_trainer \
    trainer.params.epochs="$EPOCHS" \
    trainer.loss.name=bce \
    run_name="gcn._.bce._.davis"
