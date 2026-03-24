#!/bin/bash
# _.baseline.bce._.davis.sh
# Only Student — baseline encoder, BCE loss, no gate, DAVIS.
# Usage: bash scripts/_.baseline.bce._.davis.sh

set -e
cd "$(dirname "$0")/.." || exit 1

EPOCHS="${EPOCHS:-100}"

conda run -n ugtsdti python -m ugtsdti.main \
    model=_.baseline._ \
    data=tdc_davis \
    trainer=default_trainer \
    trainer.params.epochs="$EPOCHS" \
    trainer.loss.name=bce \
    run_name="_.baseline.bce._.davis"
