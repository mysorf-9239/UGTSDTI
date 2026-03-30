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
python -m ugtsdti validate configs/baseline_reference.yaml
```

### Baseline Run Matrix

| Workflow | Purpose | Entry point |
| --- | --- | --- |
| Synthetic smoke | CI / local sanity check, no real data required | [examples/baseline.py](/Users/mysorf/PythonProject/Bioinformatics/UGTSDTI/examples/baseline.py), [scripts/baseline.sh](/Users/mysorf/PythonProject/Bioinformatics/UGTSDTI/scripts/baseline.sh) |
| Real artifact-backed baseline | Train/eval on prepared `processed/` + `splits/` artifacts | [scripts/baseline_real.sh](/Users/mysorf/PythonProject/Bioinformatics/UGTSDTI/scripts/baseline_real.sh) |
| Data preparation for real baseline | Materialize raw/PyTDC data into artifact-backed layout | [scripts/prepare_baseline_artifacts.py](/Users/mysorf/PythonProject/Bioinformatics/UGTSDTI/scripts/prepare_baseline_artifacts.py) |

### Synthetic smoke workflow

```bash
conda activate ugtsdti
cd UGTSDTI
bash scripts/baseline.sh
```

This path creates a synthetic fixture and runs `validate -> train -> eval`. It is for engineering smoke only, not for reporting research metrics.

### Real baseline workflow

1. Prepare artifact-backed data once:

```bash
conda activate ugtsdti
cd UGTSDTI
python scripts/prepare_baseline_artifacts.py --source csv --dataset davis --raw-csv /path/to/davis.csv
```

Or, if `PyTDC` is installed in the environment:

```bash
python scripts/prepare_baseline_artifacts.py --source pytdc --dataset davis
```

2. Validate a baseline profile:

```bash
python -m ugtsdti validate configs/profiles/baseline_cpu.yaml
```

3. Train + evaluate with the selected checkpoint policy:

```bash
bash scripts/baseline_real.sh configs/profiles/baseline_cpu.yaml
```

`baseline_real.sh` always runs `validate -> train -> eval` and injects the chosen checkpoint (`best` or `last`) into the eval run.

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
│   ├── baseline_reference.yaml
│   ├── profiles/baseline_local.yaml
│   ├── profiles/baseline_cpu.yaml
│   ├── profiles/baseline_gpu.yaml
│   ├── profiles/baseline_kaggle.yaml
│   └── profiles/baseline_wandb.yaml
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

Baseline train/eval is **artifact-backed only**. Runtime never downloads data and never performs preprocessing during `train` or `eval`.

The expected layout is:

```text
data/
  raw/
    davis.csv
  processed/
    davis/v1/
      dataset_version.json
      records.jsonl
  splits/
    davis/v1/v1/
      manifest.json
      s1/train.jsonl
      s1/val.jsonl
      s1/test.jsonl
      ...
```

Prepare those artifacts before training:

```bash
python scripts/prepare_baseline_artifacts.py --source csv --dataset davis --raw-csv /path/to/davis.csv
```

Or use `PyTDC` for one-time acquisition/prep:

```bash
pip install -e ".[acquisition]"
python scripts/prepare_baseline_artifacts.py --source pytdc --dataset davis
```

`PyTDC` is used only for acquisition/prep. Training and evaluation still read only from materialized artifacts.

---

## Configuration

Experiments are fully config-driven. The canonical baseline graph lives in [configs/baseline_reference.yaml](/Users/mysorf/PythonProject/Bioinformatics/UGTSDTI/configs/baseline_reference.yaml), and deployment-ready profiles extend it:

| Profile | Intended environment | Default paths / behavior |
| --- | --- | --- |
| [baseline_local.yaml](/Users/mysorf/PythonProject/Bioinformatics/UGTSDTI/configs/profiles/baseline_local.yaml) | local dev run | CPU, small batch, trace enabled |
| [baseline_cpu.yaml](/Users/mysorf/PythonProject/Bioinformatics/UGTSDTI/configs/profiles/baseline_cpu.yaml) | local CPU training | CPU, deterministic, artifact-backed |
| [baseline_gpu.yaml](/Users/mysorf/PythonProject/Bioinformatics/UGTSDTI/configs/profiles/baseline_gpu.yaml) | local GPU training | CUDA, mixed precision, pin memory |
| [baseline_kaggle.yaml](/Users/mysorf/PythonProject/Bioinformatics/UGTSDTI/configs/profiles/baseline_kaggle.yaml) | Kaggle GPU | `/kaggle/input/ugtsdti-data`, `/kaggle/working/artifacts/baseline` |
| [baseline_wandb.yaml](/Users/mysorf/PythonProject/Bioinformatics/UGTSDTI/configs/profiles/baseline_wandb.yaml) | optional logging overlay | same as local baseline, `logging.backend: wandb` |

Example baseline config:

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
    outputs: [head.logits]
interaction:
  order: [noop]
decision:
  type: identity
loss:
  type: hard
  hard_weight: 1.0
```

Config supports `extends` for inheritance and `sweep` for hyperparameter search.

### `wandb`

`wandb` remains optional. The framework degrades gracefully if it is unavailable.

To enable it:

```bash
pip install -e ".[wandb]"
python -m ugtsdti validate configs/profiles/baseline_wandb.yaml
bash scripts/baseline_real.sh configs/profiles/baseline_wandb.yaml
```

The built-in logger initializes `wandb` in offline mode by default. Kaggle and local file logging both default to `logging.backend: file` unless you explicitly choose the `wandb` profile or override the config.

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
