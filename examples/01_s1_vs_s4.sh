#!/bin/bash
# Example script: How to quickly test out S1 vs S4 with the ESM-based Student

echo "Running Warm Start (S1) Benchmark via PyTDC..."
python3 -m ugtsdti.main data=davis_s1 data.params.split_scenario="S1"

echo "--------------------------------------------------------"
echo "Running Novelty Drug/Target (S4) Benchmark via PyTDC..."
python3 -m ugtsdti.main data=davis_s4 data.params.split_scenario="S4"
