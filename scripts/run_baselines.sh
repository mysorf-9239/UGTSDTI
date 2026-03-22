#!/bin/bash
# run_baselines.sh - Demonstrates the 4 ablation modes (Offline Baseline Checks)

cd "$(dirname "$0")/.." || exit 1

export WANDB_MODE=disabled

echo "=========================================================="
echo "Starting UGTS-DTI Baseline Ablation Tests"
echo "=========================================================="

# 1. Evaluate Student Only Branch
echo "[1/4] Running ONLY STUDENT Baseline..."
conda run -n ugtsdti python -m ugtsdti.main \
    model=only_student \
    data=tdc_davis \
    trainer.params.epochs=2

# 2. Evaluate Teacher Only Branch
echo "[2/4] Running ONLY TEACHER Baseline..."
conda run -n ugtsdti python -m ugtsdti.main \
    model=only_teacher \
    data=tdc_davis \
    trainer.params.epochs=2

# 3. Evaluate Hybrid DTI without Knowledge Distillation
echo "[3/4] Running HYBRID BASELINE (No Distillation, PairGate Fusion Only)..."
conda run -n ugtsdti python -m ugtsdti.main \
    model=hybrid_baseline \
    data=tdc_davis \
    trainer.params.epochs=2 \
    trainer.loss.name=bce_with_logits

# 4. Hybrid Baseline with KD Loss + PairGate MC-Dropout Active
echo "[4/4] Running HYBRID BASELINE (With KD Loss Plugin & MC-Dropout)..."
conda run -n ugtsdti python -m ugtsdti.main model=hybrid_baseline data=tdc_davis trainer.params.epochs=2 trainer.loss.name=kd_dual_loss +trainer.loss.alpha=0.5

echo "=========================================================="
echo "All baseline ablations completed successfully!"
echo "=========================================================="
