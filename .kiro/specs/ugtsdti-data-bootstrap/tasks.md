# UGTSDTI Data Bootstrap Tasks

## Task 1 — Config surface

- [x] Add `data.source.*` validation
- [x] Add `data.source.*` normalization defaults
- [x] Add config tests for `artifacts`, `csv`, `pytdc`

## Task 2 — Bootstrap orchestrator

- [x] Add `ugtsdti/data/bootstrap.py`
- [x] Implement artifact inspection
- [x] Implement `artifacts` source behavior
- [x] Implement `csv` source behavior
- [x] Implement `pytdc` source behavior
- [x] Add bootstrap report surface

## Task 3 — CLI integration

- [x] Run bootstrap before `train`
- [x] Run bootstrap before `eval`
- [x] Preserve artifact-backed runtime boundary
- [x] Add integration tests for fail-closed and auto-prepare flows

## Checkpoint 1

- [x] Config supports `data.source.*`
- [x] Bootstrap can validate artifact-only configs

## Task 4 — Baseline workflow integration

- [x] Update `scripts/baseline.sh` to use bootstrap-driven happy path
- [x] Keep `examples/baseline.py` as smoke-only
- [x] Keep `prepare_baseline_artifacts.py` as secondary/manual utility
- [x] Add one-command baseline integration test

## Task 5 — PyTDC path

- [x] Add mocked `pytdc` acquisition coverage
- [x] Fail clearly when `pytdc` source is requested but dependency is unavailable
- [x] Keep tests offline

## Task 6 — Docs and traceability

- [x] Update README for new happy path
- [x] Document smoke vs real run
- [x] Document `artifacts|csv|pytdc`
- [x] Sync `.docs/traceability.md`

## Checkpoint 2

- [x] CSV auto-prepare works end-to-end

## Checkpoint 3

- [x] One-command baseline run works with config-driven auto-prepare

## Checkpoint 4

- [x] Full suite passes
- [x] README/docs aligned with implementation
