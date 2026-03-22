#!/bin/bash
# A sample script to run multiple experiments using Hydra's multirun feature
cd "$(dirname "$0")/.." || exit 1

echo "Running full S1-S4 Evaluation Suite on UGTSDTI..."

# Run the Baseline models sequentially using Hydra sweeping (Fast Validation)
conda run -n ugtsdti python -m ugtsdti.main -m data=davis_s1,davis_s4 model=hybrid_baseline,only_student,only_teacher trainer.params.epochs=2

echo "Experiments completed! Check WandB for results."
