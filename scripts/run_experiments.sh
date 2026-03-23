#!/bin/bash
# run_experiments.sh — Ablation sweep across model configs using Hydra multirun
cd "$(dirname "$0")/.." || exit 1

echo "Running ablation sweep on UGTSDTI..."

conda run -n ugtsdti python -m ugtsdti.main -m \
    data=tdc_davis \
    model=hybrid_baseline,only_student,only_teacher \
    trainer.params.epochs=2

echo "Experiments completed! Check WandB for results."
