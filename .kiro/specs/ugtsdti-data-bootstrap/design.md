# UGTSDTI Data Bootstrap Design

## Summary

Thiết kế thêm một lớp `DataBootstrapOrchestrator` đứng trước runtime CLI hiện có.

Flow mới:

1. load raw config
2. validate raw config
3. normalize config
4. run bootstrap against normalized config
5. hand off unchanged runtime config into existing `train` / `eval`

Core runtime graph/roles/interaction/decision/loss pipeline không đổi.

## Architecture

### New module

Add:

- `ugtsdti/data/bootstrap.py`

Primary abstraction:

- `DataBootstrapOrchestrator`

### Responsibilities

`DataBootstrapOrchestrator` is responsible for:

1. resolving required artifact paths from config
2. checking whether artifacts already exist
3. deciding whether preparation is required/allowed
4. preparing artifacts via acquisition/preprocessing/splitting when needed
5. returning a bootstrap report for CLI logging/tests

### Non-responsibilities

Bootstrap MUST NOT:

- load training batches
- invoke graph execution
- bypass trainer/evaluator
- alter role/interaction/decision semantics

## Config Surface

### Canonical shape

```yaml
data:
  dataset: davis
  preprocessing_version: v1
  split_version: v1
  source:
    type: artifacts
    auto_prepare: false
    raw_csv: null
    tdc_name: DAVIS
    label_threshold: 7.0
    label_order: descending
    drug_max_len: 64
    protein_max_len: 512
```

### Semantics

- `type=artifacts`
  - bootstrap only checks processed/split artifacts
- `type=csv`
  - bootstrap reads `raw_csv`
  - exports raw snapshot
  - materializes processed records
  - generates splits
- `type=pytdc`
  - bootstrap fetches rows from PyTDC
  - normalizes them to the same row schema as `csv`
  - materializes processed records
  - generates splits

## Bootstrap Row Schema

After source normalization, each row MUST contain:

- `drug_id`
- `protein_id`
- `drug_seq`
- `protein_seq`
- `labels`

The current baseline preprocessing contract remains sequence-token-array based.

## Integration with CLI

### `validate`

`validate` continues to validate config only. It MUST NOT perform online fetch or artifact generation.

### `train`

Before `_run_train`, CLI MUST:

- normalize config
- invoke bootstrap orchestrator
- fail early on bootstrap errors
- then continue into existing runtime flow

### `eval`

Before `_run_eval`, CLI MUST:

- normalize config
- invoke bootstrap orchestrator
- fail early on bootstrap errors
- then continue into existing runtime flow

This allows one-command baseline runs while preserving the runtime artifact boundary.

## Identity and Hashing

Bootstrap-only convenience fields SHOULD be excluded from config hashing when they do not change experiment semantics, including path-like source fields such as:

- `data.source.raw_csv`
- `data.source.auto_prepare`
- `data.source.type`
- `data.source.tdc_name`

Actual reproducibility remains anchored on:

- `config_hash`
- `dataset_version`
- `preprocessing_version`
- `split_version`
- `seed`

## User-facing Workflow

### Smoke path

- `examples/baseline.py`
- synthetic fixture
- not for research reporting

### Real path

- `scripts/baseline.sh`
- one-command `validate -> bootstrap -> train -> eval`
- checkpoint injection for final eval remains unchanged

### Secondary utility

- `scripts/prepare_baseline_artifacts.py`
- retained as manual/advanced utility
- may internally reuse bootstrap logic
