#!/bin/bash
# run_experiments.sh — Full ablation sweep via Hydra multirun.
# Usage: bash scripts/run_experiments.sh
#
# Sweep matrix:
#   models : only_student, only_teacher, only_teacher_gcn, hybrid_baseline, hybrid_gcn
#   losses : bce_with_logits, kd_dual_loss
#   splits : cold_split (S1 default via tdc_davis)
#
# Results are logged to WandB. Set WANDB_PROJECT before running.
# Override epochs: EPOCHS=50 bash scripts/run_experiments.sh

set -e
cd "$(dirname "$0")/.." || exit 1

EPOCHS="${EPOCHS:-100}"
DATA=tdc_davis
TRAINER=default_trainer

echo "=========================================="
echo " UGTS-DTI Ablation Sweep"
echo " epochs=$EPOCHS | data=$DATA"
echo "=========================================="

# Full ablation: all usable models x both losses
conda run -n ugtsdti python -m ugtsdti.main -m \
    data="$DATA" \
    trainer="$TRAINER" \
    trainer.params.epochs="$EPOCHS" \
    model=only_student,only_teacher,only_teacher_gcn,hybrid_baseline,hybrid_gcn \
    trainer.loss.name=bce_with_logits,kd_dual_loss

echo ""
echo "Sweep complete. Check WandB for results."
