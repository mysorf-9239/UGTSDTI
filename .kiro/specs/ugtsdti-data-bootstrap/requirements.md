# UGTSDTI Data Bootstrap Requirements

## Summary

Spec này thêm một lớp **data bootstrap** cho baseline và các model sau này, để user có thể chạy từ config + một entrypoint mà không phải chuẩn bị artifacts thủ công trước. Bootstrap được phép:

- acquire raw data
- preprocess raw data
- generate split artifacts

nhưng chỉ **trước** khi `train` / `eval` bắt đầu. Runtime `train` / `eval` vẫn phải giữ nguyên nguyên tắc **artifact-backed only**.

## Normative Language

- `MUST`, `MUST NOT`, `SHOULD`, `MAY` dùng theo nghĩa chuẩn RFC 2119.

## Requirements

### `REQ-DB-001` Runtime artifact boundary

`train` và `eval` MUST remain artifact-backed only. Chúng MUST NOT trực tiếp:

- download data
- call PyTDC
- preprocess raw rows
- generate splits

### `REQ-DB-002` Config-driven source declaration

The system MUST support a config-driven data source declaration under `data.source`.

Supported source types:

- `artifacts`
- `csv`
- `pytdc`

### `REQ-DB-003` Auto-prepare control

The system MUST support `data.source.auto_prepare`.

- When `true`, bootstrap MAY prepare artifacts before runtime.
- When `false`, bootstrap MUST only validate presence of required artifacts.

### `REQ-DB-004` Fail-closed on missing artifacts

When `data.source.auto_prepare=false`, missing processed/split artifacts MUST fail closed before training or evaluation starts.

### `REQ-DB-005` Bootstrap-only preparation

When `data.source.auto_prepare=true`, the system MAY materialize:

- raw snapshot
- processed records
- split manifest and scenario partitions

before calling train/eval.

### `REQ-DB-006` No responsibility drift into trainer/runtime

Bootstrap logic MUST NOT be moved into:

- `Trainer`
- `Evaluator`
- `PipelineExecutor`
- `DataLoaderFactory`

### `REQ-DB-007` Preserve artifact layout

Bootstrap MUST preserve the existing artifact layout:

- `data/raw/`
- `data/processed/<dataset>/<preprocessing_version>/`
- `data/splits/<dataset>/<preprocessing_version>/<split_version>/`

### `REQ-DB-008` Preserve version semantics

Bootstrap MUST preserve current data version semantics:

- `dataset_version`
- `preprocessing_version`
- `split_version`

### `REQ-DB-009` One-command baseline run

Baseline real-run UX MUST support:

- one config
- one command/script
- optional auto-prepare
- then `validate -> train -> eval`

### `REQ-DB-010` Optional PyTDC dependency

`PyTDC` MUST remain optional.

- If `data.source.type=pytdc` and dependency is unavailable, bootstrap MUST fail early and clearly.
- Other source types MUST continue to work without `PyTDC`.

### `REQ-DB-011` Minimal baseline surface

This feature MUST preserve the simplified baseline surface:

- one canonical baseline config
- two official baseline profiles: `local`, `kaggle`
- one smoke workflow
- one real workflow

### `REQ-DB-012` Bootstrap-only knobs must not pollute run identity

Bootstrap convenience knobs that do not change experiment semantics MUST NOT silently perturb experiment identity/config hashing.

### `REQ-DB-013` Reusable beyond baseline

The bootstrap mechanism SHOULD be reusable later for teacher/student models without changing the core runtime contract.
