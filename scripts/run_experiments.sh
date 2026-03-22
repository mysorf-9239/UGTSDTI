#!/bin/bash
# A sample script to run multiple experiments using Hydra's multirun feature

echo "Running full S1-S4 Evaluation Suite on UGTSDTI..."

# Run the 4 scenarios sequentially using Hydra sweeping
python -m ugtsdti.main -m data=davis_s1,davis_s4 model=default_hybrid trainer=default_trainer

echo "Experiments completed! Check WandB for results."
