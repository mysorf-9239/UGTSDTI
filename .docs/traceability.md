# Traceability Matrix

## Purpose

Matrix nay map requirement IDs sang design/docs, config surface, implementation hien tai, va quality tests dang cover.

## Matrix

| Requirement | Design / Docs | Config Surface | Implementation | Validation |
| --- | --- | --- | --- | --- |
| `REQ-ARCH-001` Fixed pipeline order | `contracts.md`, `architecture.md` | implicit runtime pipeline | `ugtsdti/trainer/trainer.py`, `ugtsdti/trainer/evaluator.py` | `tests/quality/test_pipeline.py`, `tests/quality/test_orchestration.py` |
| `REQ-ARCH-002` Teacher-student asymmetry | `experiment.md`, `contracts.md` | `roles`, `training.teacher`, `training.student`, `modalities.*` | `ugtsdti/roles/binder.py`, `ugtsdti/config/validate.py`, `ugtsdti/trainer/trainer.py` | `tests/quality/test_pipeline.py`, `tests/quality/test_config.py` |
| `REQ-ARCH-004` Stable naming | `contracts.md`, `config_schema.md` | `graph.nodes.*`, `roles.*`, `interaction.*`, `loss.map` | `ugtsdti/core/schema.py`, `ugtsdti/config/validate.py`, `ugtsdti/postprocess/metrics.py` | `tests/quality/test_config.py`, `tests/quality/test_interaction.py`, `tests/quality/test_postprocess.py` |
| `REQ-CONF-001` / `REQ-CONF-003` Config loading and normalization | `config_schema.md` | `extends`, `sweep`, all top-level sections | `ugtsdti/config/loader.py`, `ugtsdti/config/normalize.py`, `ugtsdti/cli/main.py` | `tests/quality/test_config.py`, `tests/quality/test_cli.py` |
| `REQ-CONF-002` Config cross-section validation | `contracts.md`, `config_schema.md` | `roles`, `interaction`, `decision`, `loss`, `modalities` | `ugtsdti/config/validate.py` | `tests/quality/test_config.py` |
| `REQ-GRAPH-001` / `REQ-GRAPH-002` Node spec and DAG planning | `node_spec.md`, `graph_execution.md` | `graph.nodes.*` | `ugtsdti/graph/specs.py`, `ugtsdti/graph/builder.py`, `ugtsdti/graph/planner.py`, `ugtsdti/graph/registry.py` | `tests/quality/test_graph.py` |
| `REQ-GRAPH-003` Execution trace and boundary summaries | `graph_execution.md` | `logging.trace_execution`, debug runtime | `ugtsdti/graph/engine.py`, `ugtsdti/trainer/trainer.py`, `ugtsdti/runtime/artifacts.py` | `tests/quality/test_pipeline.py`, `tests/quality/test_artifacts.py` |
| `REQ-ROLE-001` Canonical role binding | `contracts.md` | `roles.*.outputs`, `roles.*.aggregation` | `ugtsdti/roles/binder.py` | `tests/quality/test_roles.py` |
| `REQ-INT-001` Interaction ordering / dependency graph | `contracts.md` | `interaction.order`, `interaction.dependencies` | `ugtsdti/interaction/registry.py`, `ugtsdti/config/validate.py` | `tests/quality/test_interaction.py`, `tests/quality/test_config.py` |
| `REQ-INT-002` KD interaction | `experiment.md`, `contracts.md` | `interaction.kd`, `loss.map.kd`, `training.kd.*` | `ugtsdti/interaction/kd.py`, `ugtsdti/postprocess/loss.py`, `ugtsdti/trainer/trainer.py` | `tests/quality/test_interaction.py`, `tests/quality/test_pipeline.py`, `tests/quality/test_orchestration.py` |
| `REQ-INT-003` Uncertainty interaction | `experiment.md`, `contracts.md` | `interaction.uncertainty`, `decision.use_uncertainty` | `ugtsdti/interaction/uncertainty.py`, `ugtsdti/decision/trust.py` | `tests/quality/test_interaction.py`, `tests/quality/test_decision.py`, `tests/quality/test_pipeline.py` |
| `REQ-DEC-001` / `REQ-DEC-002` Decision stage contracts | `architecture.md`, `contracts.md` | `decision.*` | `ugtsdti/decision/base.py`, `ugtsdti/decision/module.py`, `ugtsdti/decision/policy.py`, `ugtsdti/decision/trust.py` | `tests/quality/test_decision.py`, `tests/quality/test_pipeline.py` |
| `REQ-POST-001` Explicit loss composition | `contracts.md` | `loss.type`, `loss.hard_weight`, `loss.map` | `ugtsdti/postprocess/loss.py` | `tests/quality/test_postprocess.py`, `tests/quality/test_pipeline.py` |
| `REQ-POST-002` Metrics and diagnostics reporting | `experiment.md`, `contracts.md` | `metrics.*`, `diagnostics.*` | `ugtsdti/postprocess/metrics.py`, `ugtsdti/trainer/evaluator.py` | `tests/quality/test_postprocess.py`, `tests/quality/test_orchestration.py` |
| `REQ-DATA-001` to `REQ-DATA-004` Data acquisition, preprocessing, loading, validation | `architecture.md`, `contracts.md` | `data.*`, `scenario.*`, graph-required batch keys | `ugtsdti/data/acquisition.py`, `ugtsdti/data/preprocessing.py`, `ugtsdti/data/splitting.py`, `ugtsdti/data/loader.py`, `ugtsdti/data/validate.py` | `tests/quality/test_data.py` |
| `REQ-TRAIN-001` Training dynamics | `experiment.md` | `training.teacher`, `training.student`, `training.kd`, `training.gate` | `ugtsdti/trainer/trainer.py` | `tests/quality/test_orchestration.py` |
| `REQ-EVAL-001` Scenario-aware evaluation | `experiment.md` | `scenario.eval`, `metrics.by_scenario` | `ugtsdti/trainer/evaluator.py`, `ugtsdti/postprocess/metrics.py` | `tests/quality/test_orchestration.py`, `tests/quality/test_postprocess.py` |
| `REQ-ABL-001` Ablation and sweep support | `config_schema.md`, `implementation_plan.md` | `extends`, `sweep`, sample configs | `ugtsdti/cli/main.py`, `configs/ablations/*.yaml`, `configs/sweeps/full.yaml` | `tests/quality/test_cli.py`, `tests/quality/test_orchestration.py` |
| `REQ-REPRO-001` Reproducibility controls | `requirements.md` section 7 | `runtime.seed`, `runtime.deterministic` | `ugtsdti/runtime/seed.py`, `ugtsdti/runtime/adapter.py` | `tests/quality/test_runtime.py` |
| `REQ-REPRO-002` Experiment identity and artifact bundle | `requirements.md` section 7 | `runtime.artifacts_dir`, normalized config snapshot, split metadata | `ugtsdti/runtime/identity.py`, `ugtsdti/runtime/artifacts.py` | `tests/quality/test_runtime.py`, `tests/quality/test_artifacts.py` |
| `REQ-OPS-001` / `REQ-OPS-002` Logging abstraction and CLI | `architecture.md`, `implementation_plan.md` | `logging.*`, CLI commands | `ugtsdti/logging/*.py`, `ugtsdti/cli/main.py`, `ugtsdti/__main__.py` | `tests/quality/test_runtime.py`, `tests/quality/test_cli.py` |
| `REQ-OPS-003` Runtime adaptation | `requirements.md` section 7 | `runtime.*` | `ugtsdti/runtime/adapter.py` | `tests/quality/test_runtime.py` |
| `REQ-QUAL-001` / `REQ-QUAL-002` Failure taxonomy and numerical safety | `contracts.md` | stage-boundary validation, strict runtime | `ugtsdti/core/errors.py`, `ugtsdti/graph/engine.py`, `ugtsdti/config/validate.py` | `tests/quality/test_config.py`, `tests/quality/test_graph_engine.py`, `tests/quality/test_interaction.py` |
| `REQ-QUAL-003` Checkpoint / resume | `requirements.md` section 8 | checkpoint bundle and resume validation | `ugtsdti/runtime/checkpoint.py` | `tests/quality/test_runtime.py` |
| `REQ-QUAL-004` Performance and concurrency constraints | `graph_execution.md`, `contracts.md` | graph execution and state model | `ugtsdti/core/state.py`, `ugtsdti/graph/engine.py` | `tests/quality/test_artifacts.py`, `tests/quality/test_graph_engine.py` |

## Notes

- Minimal baseline hien duoc phan anh boi `decision.type: identity`, `interaction.order: [noop]`, va `tests/quality/test_pipeline.py`.
- Full pipeline sample hien duoc phan anh boi [base_full.yaml](/Users/mysorf/PythonProject/Bioinformatics/UGTSDTI/configs/base_full.yaml).
- Matrix nay nen duoc cap nhat moi khi them REQ ID moi, doi module owner, hoac doi test owner.
