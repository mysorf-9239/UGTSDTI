# UGTSDTI

**Uncertainty-Gated Teacher–Student Learning for Drug–Target Interaction Prediction**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

---

## Research Question

- When does a **teacher** (richer modality, pretrained) help a **student** (deployable, inductive)?
- When does **knowledge distillation** improve the student?
- When does **uncertainty** predict branch reliability?
- When does an **uncertainty-gated decision** outperform static rules?
- How do these behaviors shift across warm and cold DTI scenarios?

---

## Pipeline

The execution pipeline is fixed and invariant across all experiments:

```
Batch → Graph → Role Binding → Interaction (KD + Uncertainty) → Decision (Gate) → Loss + Metrics
```

Experiments vary by swapping plugins and config — not by rewriting the pipeline.

---

## Cold-Start Scenarios

| Scenario | Drug seen | Target seen | Meaning |
|----------|:---------:|:-----------:|---------|
| `S1` | yes | yes | warm-start |
| `S2` | no | yes | cold drug |
| `S3` | yes | no | cold target |
| `S4` | no | no | fully cold |

`S4` is the strongest indicator of genuine inductive behavior.

---

## Installation

```bash
git clone <repo>
cd UGTSDTI

# Development environment (recommended)
conda env create -f conda-recipes/dev.yaml
conda activate ugtsdti

# GPU environment
conda env create -f conda-recipes/gpu.yaml
conda activate ugtsdti-gpu

# CPU-only
conda env create -f conda-recipes/cpu.yaml
conda activate ugtsdti-cpu
```

Available environments: `conda-recipes/base.yaml`, `conda-recipes/dev.yaml`, `conda-recipes/cpu.yaml`, `conda-recipes/gpu.yaml`, `conda-recipes/kaggle.yaml`

---

## Quick Start

```bash
# Validate config before running
python -m ugtsdti validate --config configs/baseline.yaml

# Train
python -m ugtsdti train --config configs/baseline.yaml

# Evaluate on all scenarios (S1–S4)
python -m ugtsdti eval --config configs/baseline.yaml --checkpoint artifacts/<run_id>/model.pt

# Debug run (minimal config, dry-run)
python -m ugtsdti train --config configs/minimal.yaml --dry-run

# Hyperparameter sweep
python -m ugtsdti sweep --config configs/sweep.yaml
```

Or via Makefile:

```bash
make validate-config   # dry-run config validation
make train             # train with baseline config
make eval              # evaluate
make debug             # minimal dry-run
make sweep             # hyperparameter sweep
make test              # run all tests
make test-pbt          # run property-based tests
```

---

## Project Structure

```
UGTSDTI/
├── ugtsdti/
│   ├── core/          # State, ExecutionContext, error taxonomy, schema
│   ├── config/        # Loader, validator, normalizer, NormalizedConfig
│   ├── graph/         # DAG builder, planner, engine
│   ├── nodes/         # Node plugins: encoder/, fusion/, head/
│   ├── roles/         # Role binding (teacher.logits, student.logits)
│   ├── interaction/   # KD, uncertainty, diagnostics
│   ├── decision/      # Trust estimator, decision policy, gate modules
│   ├── postprocess/   # Loss composer, metrics reporter
│   ├── data/          # Acquisition, preprocessing, splitting, loader, validation
│   ├── runtime/       # Seed, identity, adapter, checkpoint
│   ├── logging/       # Logger abstraction (file, wandb, composite)
│   ├── trainer/       # Trainer, evaluator, pipeline executor
│   └── cli/           # CLI entrypoints
├── configs/
│   ├── minimal.yaml   # Student-only baseline (Checkpoint 1)
│   ├── baseline.yaml  # Full UGTS pipeline
│   └── sweep.yaml     # Hyperparameter sweep
├── conda-recipes/
│   ├── base.yaml
│   ├── dev.yaml
│   ├── cpu.yaml
│   ├── gpu.yaml
│   └── kaggle.yaml
├── data/
│   ├── raw/           # Raw CSV snapshots (gitignored)
│   ├── processed/     # Materialized tensors (gitignored)
│   └── splits/        # Deterministic split artifacts (gitignored)
├── artifacts/         # Experiment outputs (gitignored)
├── tests/
│   ├── pbt/           # Property-based tests (hypothesis)
│   ├── quality/       # Unit tests for contracts and invariants
│   └── integration/   # End-to-end pipeline tests
├── .docs/             # System design documentation
├── .kiro/specs/       # Spec: requirements, design, tasks
├── Makefile
├── pyproject.toml
└── README.md
```

---

## Data Preparation

Dataset acquisition is a one-time step, separate from training runtime:

```bash
# 1. Acquire raw data (requires PyTDC)
pip install PyTDC
python -m ugtsdti.data.acquisition --dataset davis

# 2. Preprocess into tensors
python -m ugtsdti.data.preprocessing --dataset davis

# 3. Generate deterministic splits (S1–S4)
python -m ugtsdti.data.splitting --dataset davis --seed 42

# 4. Validate data artifacts
make validate-data
```

Training reads only from materialized `.pt` files — no network calls at runtime.

---

## Configuration

Experiments are fully config-driven. Example minimal config:

```yaml
version: v1
data:
  dataset: davis
scenario:
  train: s1
  eval: [s1, s2, s3, s4]
modalities:
  available: [sequence]
  student:
    uses: [sequence]
graph:
  nodes:
    student_encoder:
      type: encoder.baseline
      inputs: [drug_seq, protein_seq]
    student_head:
      type: head.linear
      inputs: [student_encoder.embedding]
roles:
  student:
    outputs: [student_head.logits]
interaction:
  order: []
decision:
  type: identity
training:
  student:
    freeze: false
loss:
  type: composite
  hard_weight: 1.0
```

Config supports `extends` for inheritance and `sweep` for hyperparameter search.

---

## Reproducibility

Every run produces an artifact bundle at `artifacts/<run_id>/`:

```
artifacts/<run_id>/
  config.yaml          # normalized config snapshot
  identity.json        # run_id, config_hash, git_commit, timestamp
  metrics.json         # per-scenario metrics (S1–S4)
  diagnostics.json     # gate_alpha, disagreement, uncertainty_error
  split_manifest.json  # dataset/split version metadata
  model.pt             # model checkpoint
  logs/
```

Full reproducibility key: `(config_hash, dataset_version, preprocessing_version, split_version, seed)`

---

## Development

```bash
make lint        # ruff check
make typecheck   # mypy --strict
make test        # pytest
make test-pbt    # property-based tests with fixed seed
make pre-commit  # run all pre-commit hooks
```

---

## Documentation

System design documentation is in `.docs/`:

- `.docs/architecture.md` — pipeline structure and teacher-student semantics
- `.docs/contracts.md` — invariants, interfaces, naming rules
- `.docs/config_schema.md` — configuration API
- `.docs/node_spec.md` — graph node behavior
- `.docs/graph_execution.md` — DAG construction and execution
- `.docs/experiment.md` — research protocol and evaluation

Spec files are in `.kiro/specs/ugtsdti-framework/`:
- `requirements.md`
- `design.md`
- `tasks.md`

---

## Citation

```bibtex
@misc{ugtsdti2026,
    title  = {UGTSDTI: Uncertainty-Gated Teacher--Student Learning for Drug--Target Interaction Prediction},
    author = {Mysorf},
    year   = {2026},
    note   = {Work in progress}
}
```

## License

[MIT License](LICENSE)
