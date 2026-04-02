# Traceability Matrix

## Purpose

Matrix này map `REQ-INTG-*` của pha integrity hardening sang design, config surface, implementation, và tests đang cover.

## Matrix

| Requirement | Design / Docs | Config Surface | Implementation | Validation |
| --- | --- | --- | --- | --- |
| `REQ-INTG-001` Normalization preserves semantics | `.kiro/specs/ugtsdti-integrity-hardening/design.md` Section 4 | `graph.nodes[*].inputs`, `roles.*.outputs`, hashing canonicalization | `ugtsdti/config/normalize.py`, `ugtsdti/runtime/identity.py` | `tests/quality/test_config.py`, `tests/pbt/test_config_properties.py`, `tests/quality/test_runtime.py` |
| `REQ-INTG-002` State read immutability | `.kiro/specs/ugtsdti-integrity-hardening/design.md` Section 5 | implicit state read policy for sensitive namespaces | `ugtsdti/core/state.py`, `ugtsdti/trainer/trainer.py` | `tests/quality/test_state.py`, `tests/integration/test_pipeline.py` |
| `REQ-INTG-003` Graph runtime lifecycle explicit | `.kiro/specs/ugtsdti-integrity-hardening/design.md` Section 6 | executor / runtime cache lifecycle | `ugtsdti/graph/engine.py` | `tests/quality/test_graph_engine.py` |
| `REQ-INTG-004` Interaction runtime enforces declared inputs | `.kiro/specs/ugtsdti-integrity-hardening/design.md` Section 7 | `interaction.order`, module declared `inputs` | `ugtsdti/interaction/engine.py`, `ugtsdti/interaction/registry.py` | `tests/quality/test_interaction.py`, `tests/integration/test_pipeline.py` |
| `REQ-INTG-005` Checkpoint restore fail-closed | `.kiro/specs/ugtsdti-integrity-hardening/design.md` Section 8 | `runtime.checkpoint_path` | `ugtsdti/trainer/trainer.py`, `ugtsdti/runtime/checkpoint.py` | `tests/quality/test_runtime.py`, `tests/integration/test_pipeline.py`, `tests/integration/test_cli.py` |
| `REQ-INTG-006` Reproducibility hash reflects research semantics | `.kiro/specs/ugtsdti-integrity-hardening/design.md` Section 9 | `runtime.*` operational paths excluded from hash | `ugtsdti/runtime/identity.py` | `tests/quality/test_runtime.py` |
| `REQ-INTG-007` Scenario coverage explicit and complete | `.kiro/specs/ugtsdti-integrity-hardening/design.md` Section 10 | `scenario.eval`, loader permissive vs strict coverage | `ugtsdti/data/loader.py`, `ugtsdti/cli/main.py` | `tests/quality/test_data.py`, `tests/integration/test_cli.py` |
| `REQ-INTG-008` Validator/runtime canonical semantics | `.kiro/specs/ugtsdti-integrity-hardening/design.md` Section 11 | validator injection, decision fallback, plugin registrars | `ugtsdti/config/validate.py`, `ugtsdti/cli/main.py`, `ugtsdti/runtime/plugins.py` | `tests/quality/test_config.py`, `tests/integration/test_cli.py` |
| `REQ-INTG-009` Role aggregation must not fail silently | `.kiro/specs/ugtsdti-integrity-hardening/design.md` Section 12 | `roles.*.aggregation`, `roles.*.allow_multi_output_first` | `ugtsdti/roles/binder.py`, `ugtsdti/config/validate.py`, `ugtsdti/trainer/trainer.py` | `tests/quality/test_roles.py`, `tests/quality/test_config.py`, `tests/integration/test_pipeline.py` |
| `REQ-INTG-010` Artifact emission respects lifecycle | `.kiro/specs/ugtsdti-integrity-hardening/design.md` Section 13 | canonical final bundle vs `snapshots/` bundles | `ugtsdti/runtime/artifacts.py`, `ugtsdti/trainer/trainer.py`, `ugtsdti/trainer/evaluator.py`, `ugtsdti/cli/main.py` | `tests/quality/test_artifacts.py`, `tests/integration/test_orchestration.py`, `tests/integration/test_cli.py` |
| `REQ-INTG-011` Traceability update | `.kiro/specs/ugtsdti-integrity-hardening/{requirements,design,tasks}.md` | N/A | `.docs/traceability.md` | repo audit + spec/task completion |

## Notes

- Runtime lifecycle hiện được chốt là persistent theo `GraphEngine` / `PipelineExecutor`; determinism claim chỉ defensible trong executor lifecycle đã được kiểm soát.
- `State` hiện có isolated read path cho namespaces nhạy cảm; graph hot path vẫn giữ raw read semantics vì performance.
- `Decision` fallback semantics được sync theo runtime hiện có: một branch còn sống vẫn hợp lệ, không cần khai báo fallback tường minh chỉ để qua validator.
- `roles.<name>.allow_multi_output_first` là explicit escape hatch cho các case debug/ablation; mặc định hardening sẽ reject multi-output `first`.
- Canonical artifact bundle hiện nằm ở `artifacts/<run_id>/`, còn trainer snapshots nằm ở `artifacts/<run_id>/snapshots/<label>/`.

## Baseline Reference Matrix

Matrix này map `REQ-BL-*` của baseline reference model sang config surface, implementation, và tests đang cover.

| Requirement | Design / Docs | Config Surface | Implementation | Validation |
| --- | --- | --- | --- | --- |
| `REQ-BL-001` Baseline respects fixed pipeline | `.kiro/specs/ugtsdti-baseline-model/design.md` Sections 2-3 | `interaction.noop`, `decision.identity`, `loss.hard` | `ugtsdti/trainer/trainer.py`, `ugtsdti/interaction/noop.py` | `tests/integration/test_baseline_reference.py` |
| `REQ-BL-002` Student-only baseline canonical path | `.kiro/specs/ugtsdti-baseline-model/requirements.md` Section 5 | `roles.student.outputs`, `decision.source_key` | `configs/baseline_reference.yaml`, `ugtsdti/roles/binder.py` | `tests/integration/test_baseline_reference.py` |
| `REQ-BL-003` Baseline graph topology | `.kiro/specs/ugtsdti-baseline-model/design.md` Section 3 | `graph.nodes` order and dependencies | `ugtsdti/graph/builder.py`, `ugtsdti/graph/planner.py` | `tests/integration/test_baseline_reference.py`, `tests/quality/test_graph.py` |
| `REQ-BL-004` `encoder.simple_drug` | `.kiro/specs/ugtsdti-baseline-model/design.md` Section 4.1 | `graph.nodes[*].type: encoder.simple_drug` | `ugtsdti/nodes/baseline.py`, `ugtsdti/runtime/defaults.py` | `tests/quality/test_baseline_nodes.py` |
| `REQ-BL-005` `encoder.cnn_protein` | `.kiro/specs/ugtsdti-baseline-model/design.md` Section 4.2 | `graph.nodes[*].type: encoder.cnn_protein` | `ugtsdti/nodes/baseline.py`, `ugtsdti/runtime/defaults.py` | `tests/quality/test_baseline_nodes.py` |
| `REQ-BL-006` `fusion.concat` | `.kiro/specs/ugtsdti-baseline-model/design.md` Section 4.3 | `graph.nodes[*].type: fusion.concat` | `ugtsdti/nodes/baseline.py`, `ugtsdti/runtime/defaults.py` | `tests/quality/test_baseline_nodes.py` |
| `REQ-BL-007` `head.mlp` raw-logits head | `.kiro/specs/ugtsdti-baseline-model/design.md` Section 4.4 | `graph.nodes[*].type: head.mlp` | `ugtsdti/nodes/baseline.py`, `ugtsdti/runtime/defaults.py` | `tests/quality/test_baseline_nodes.py` |
| `REQ-BL-008` Deterministic fail-closed registry integration | `.kiro/specs/ugtsdti-baseline-model/design.md` Section 5 | graph node type keys | `ugtsdti/graph/registry.py`, `ugtsdti/runtime/defaults.py` | `tests/quality/test_graph.py`, `tests/quality/test_baseline_nodes.py` |
| `REQ-BL-009` Canonical baseline config | `.kiro/specs/ugtsdti-baseline-model/design.md` Section 6 | `configs/baseline_reference.yaml` | `configs/baseline_reference.yaml` | `tests/integration/test_baseline_reference.py` |
| `REQ-BL-010` Forward materializes canonical keys | `.kiro/specs/ugtsdti-baseline-model/design.md` Section 7 | graph / roles / decision / metrics keys | `ugtsdti/trainer/trainer.py` | `tests/integration/test_baseline_reference.py` |
| `REQ-BL-011` Training is learnable | `.kiro/specs/ugtsdti-baseline-model/design.md` Section 8 | `training.optimizer`, `training.loop` | `ugtsdti/core/state.py`, `ugtsdti/trainer/trainer.py`, `ugtsdti/nodes/baseline.py` | `tests/integration/test_baseline_reference.py`, `tests/quality/test_state.py` |
| `REQ-BL-012` Scenario-aware evaluation support | `.kiro/specs/ugtsdti-baseline-model/design.md` Section 8 | `metrics.by_scenario`, `scenario.eval` | `ugtsdti/postprocess/metrics.py`, `ugtsdti/trainer/evaluator.py` | `tests/integration/test_baseline_reference.py` |
| `REQ-BL-013` Runtime state round-trip | `.kiro/specs/ugtsdti-baseline-model/design.md` Section 5 | model state / checkpoint lineage | `ugtsdti/trainer/trainer.py`, `ugtsdti/nodes/baseline.py` | `tests/integration/test_baseline_reference.py`, `tests/quality/test_baseline_nodes.py` |
| `REQ-BL-014` Minimal surface, no extra research logic | `.kiro/specs/ugtsdti-baseline-model/requirements.md` Section 5 | no teacher/KD/uncertainty in canonical config | `configs/baseline_reference.yaml` | config inspection + `tests/integration/test_baseline_reference.py` |
| `REQ-BL-015` Traceability update | `.kiro/specs/ugtsdti-baseline-model/{requirements,design,tasks}.md` | N/A | `.docs/traceability.md` | repo audit + task completion |

## Baseline Training Hardening Matrix

Matrix này map `REQ-BTH-*` của baseline training hardening sang config surface, implementation, và tests đang cover.

| Requirement | Design / Docs | Config Surface | Implementation | Validation |
| --- | --- | --- | --- | --- |
| `REQ-BTH-001` Fixed pipeline compatibility | `.kiro/specs/ugtsdti-baseline-training-hardening/design.md` Sections 1, 5 | `interaction.noop`, `decision.identity` | `ugtsdti/trainer/trainer.py`, `configs/baseline_reference.yaml` | `tests/integration/test_baseline_reference.py` |
| `REQ-BTH-002` Long-run trainability | `.kiro/specs/ugtsdti-baseline-training-hardening/requirements.md` | `training.loop.*` | `ugtsdti/cli/main.py`, `ugtsdti/trainer/trainer.py` | `tests/integration/test_baseline_reference.py`, `tests/integration/test_cli.py` |
| `REQ-BTH-003` Optimizer config surface | `.kiro/specs/ugtsdti-baseline-training-hardening/design.md` Section 2.1 | `training.optimizer.type` | `ugtsdti/config/normalize.py`, `ugtsdti/config/validate.py`, `ugtsdti/cli/main.py` | `tests/integration/test_cli.py` |
| `REQ-BTH-004` Scheduler config surface | `.kiro/specs/ugtsdti-baseline-training-hardening/design.md` Section 2.2 | `training.scheduler.type` | `ugtsdti/config/normalize.py`, `ugtsdti/config/validate.py`, `ugtsdti/cli/main.py` | `tests/integration/test_cli.py` |
| `REQ-BTH-005` Gradient safety | `.kiro/specs/ugtsdti-baseline-training-hardening/design.md` Section 3.6 | `training.loop.max_grad_norm`, `training.loop.fail_on_nonfinite_*` | `ugtsdti/trainer/trainer.py` | `tests/integration/test_baseline_reference.py`, `tests/integration/test_cli.py` |
| `REQ-BTH-006` Resume integrity | `.kiro/specs/ugtsdti-baseline-training-hardening/design.md` Section 3.3 | `runtime.checkpoint_path` | `ugtsdti/cli/main.py`, `ugtsdti/runtime/checkpoint.py`, `ugtsdti/trainer/trainer.py` | `tests/integration/test_cli.py` |
| `REQ-BTH-007` Best checkpoint tracking | `.kiro/specs/ugtsdti-baseline-training-hardening/design.md` Section 3.4 | `training.loop.best_metric`, `training.loop.best_mode` | `ugtsdti/cli/main.py` | `tests/integration/test_cli.py` |
| `REQ-BTH-008` Early stopping | `.kiro/specs/ugtsdti-baseline-training-hardening/design.md` Section 3.4 | `training.loop.early_stopping.*` | `ugtsdti/config/validate.py`, `ugtsdti/cli/main.py` | `tests/integration/test_cli.py` |
| `REQ-BTH-009` Final evaluation correctness | `.kiro/specs/ugtsdti-baseline-training-hardening/design.md` Section 3.5 | `training.loop.select_checkpoint` | `ugtsdti/cli/main.py`, `ugtsdti/trainer/evaluator.py` | `tests/integration/test_cli.py`, `examples/baseline.py` smoke run |
| `REQ-BTH-010` Baseline workflow correctness | `.kiro/specs/ugtsdti-baseline-training-hardening/design.md` Section 5 | `examples/baseline.py` | `examples/baseline.py` | local smoke run |
| `REQ-BTH-011` Reporting surface | `.kiro/specs/ugtsdti-baseline-training-hardening/design.md` Section 4 | train summary JSON, artifact bundle | `ugtsdti/cli/main.py`, `ugtsdti/runtime/artifacts.py` | `tests/integration/test_cli.py` |
| `REQ-BTH-012` Deterministic surface preservation | `.kiro/specs/ugtsdti-baseline-training-hardening/requirements.md` | explicit checkpoint selection / config-driven loop | `ugtsdti/runtime/identity.py`, `ugtsdti/cli/main.py` | existing runtime/integration tests |
| `REQ-BTH-013` Contract preservation | `.kiro/specs/ugtsdti-baseline-training-hardening/requirements.md` | no direct model bypass | `ugtsdti/trainer/trainer.py`, `ugtsdti/graph/engine.py` | code inspection + integration tests |
| `REQ-BTH-014` Validation | `.kiro/specs/ugtsdti-baseline-training-hardening/design.md` Section 2 | optimizer/scheduler/loop validation | `ugtsdti/config/validate.py` | `tests/integration/test_cli.py` |
| `REQ-BTH-015` Verification | `.kiro/specs/ugtsdti-baseline-training-hardening/tasks.md` | N/A | test suite + smoke run | `tests/integration/test_baseline_reference.py`, `tests/integration/test_cli.py` |

## Baseline Portability Matrix

Matrix này map baseline portability hardening sang config surface, implementation, và tests đang cover.

| Requirement | Design / Docs | Config Surface | Implementation | Validation |
| --- | --- | --- | --- | --- |
| `REQ-BP-001` Baseline deployment profiles are config-driven | `README.md` quick start + configuration | `configs/profiles/baseline_{local,kaggle}.yaml` | `configs/profiles/baseline_local.yaml`, `configs/profiles/baseline_kaggle.yaml` | `tests/integration/test_baseline_portability.py`, `tests/integration/test_orchestration.py` |
| `REQ-BP-002` Profiles extend baseline canonical config | `README.md` configuration section | `extends: ../baseline_reference.yaml` | `configs/profiles/baseline_*.yaml` | `tests/integration/test_baseline_portability.py` |
| `REQ-BP-003` Artifact-backed runtime only | `README.md` data preparation section | `runtime.data_dir`, `data.preprocessing_version`, `data.split_version` | `ugtsdti/cli/main.py`, `ugtsdti/data/loader.py` | `tests/integration/test_baseline_portability.py`, existing CLI/data tests |
| `REQ-BP-004` Baseline-facing data prep entrypoint | `README.md` data preparation section | script arguments | `scripts/prepare_baseline_artifacts.py`, `ugtsdti/data/{acquisition,preprocessing,splitting}.py` | script review + `tests/integration/test_baseline_portability.py` README drift coverage |
| `REQ-BP-005` Smoke and real workflows are explicit | `README.md` quick start | `examples/baseline.py`, `scripts/baseline.sh` | `examples/baseline.py`, `scripts/baseline.sh` | `tests/integration/test_baseline_portability.py` |
| `REQ-BP-006` `wandb` remains optional | `README.md` wandb section | `logging.backend: wandb` override on baseline profiles | `ugtsdti/logging/wandb_logger.py` | `tests/integration/test_baseline_portability.py` |
| `REQ-BP-007` Kaggle path semantics are first-class | `README.md` profile table | `runtime.data_dir`, `runtime.artifacts_dir`, `runtime.checkpoint_dir` | `configs/profiles/baseline_kaggle.yaml` | `tests/integration/test_baseline_portability.py`, `tests/integration/test_orchestration.py` |
| `REQ-BP-008` README/config drift is checked | `README.md` | referenced baseline configs and scripts | `README.md` | `tests/integration/test_baseline_portability.py` |

## Data Bootstrap Matrix

Matrix này map `REQ-DB-*` của data bootstrap layer sang config surface, implementation, và tests đang cover.

| Requirement | Design / Docs | Config Surface | Implementation | Validation |
| --- | --- | --- | --- | --- |
| `REQ-DB-001` Train/eval runtime stays artifact-backed | `.kiro/specs/ugtsdti-data-bootstrap/design.md` | `data.source.*`, `runtime.data_dir` | `ugtsdti/cli/main.py`, `ugtsdti/data/loader.py`, `ugtsdti/data/bootstrap.py` | `tests/integration/test_cli.py`, `tests/quality/test_data.py` |
| `REQ-DB-002` Config-driven data source selection | `.kiro/specs/ugtsdti-data-bootstrap/requirements.md` | `data.source.type` | `ugtsdti/config/{normalize,validate}.py`, `ugtsdti/data/bootstrap.py` | `tests/quality/test_config.py`, `tests/quality/test_data.py` |
| `REQ-DB-003` `auto_prepare` semantics | `.kiro/specs/ugtsdti-data-bootstrap/design.md` | `data.source.auto_prepare` | `ugtsdti/data/bootstrap.py`, `ugtsdti/cli/main.py` | `tests/quality/test_data.py`, `tests/integration/test_cli.py` |
| `REQ-DB-004` Missing artifacts fail closed when prepare disabled | `.kiro/specs/ugtsdti-data-bootstrap/requirements.md` | `data.source.auto_prepare: false` | `ugtsdti/data/bootstrap.py` | `tests/quality/test_data.py`, `tests/integration/test_baseline_portability.py` |
| `REQ-DB-005` CSV bootstrap path | `.kiro/specs/ugtsdti-data-bootstrap/design.md` | `data.source.type: csv`, `data.source.raw_csv` | `ugtsdti/data/bootstrap.py` | `tests/quality/test_data.py`, `tests/integration/test_cli.py` |
| `REQ-DB-006` PyTDC bootstrap path remains optional | `.kiro/specs/ugtsdti-data-bootstrap/design.md` | `data.source.type: pytdc` | `ugtsdti/data/bootstrap.py` | `tests/quality/test_data.py` |
| `REQ-DB-007` One-command baseline happy path | `README.md` real workflow section | `scripts/baseline.sh` + config | `scripts/baseline.sh`, `ugtsdti/cli/main.py` | local smoke run + `tests/integration/test_baseline_portability.py` |
| `REQ-DB-008` Manual prep remains secondary utility | `README.md` data preparation section | `scripts/prepare_baseline_artifacts.py --config` | `scripts/prepare_baseline_artifacts.py` | script help smoke + README drift test |
| `REQ-DB-009` Research-oriented hash semantics preserved | `.kiro/specs/ugtsdti-data-bootstrap/design.md` | bootstrap-only `data.source.*` excluded from hash | `ugtsdti/runtime/identity.py` | `tests/quality/test_runtime.py` |

## Flow 1 Teacher-Student Matrix

Matrix này map `REQ-FLOW1-*` của Flow 1 Teacher-Student spec sang design, config surface, implementation, và tests.

| Requirement | Design / Docs | Config Surface | Implementation | Validation |
| --- | --- | --- | --- | --- |
| `REQ-FLOW1-001` `encoder.seq_bilstm` | `.kiro/specs/ugtsdti-flow1-model/design.md` Section 5.1 | `graph.nodes[*].type: encoder.seq_bilstm` | `ugtsdti/nodes/flow1.py::SeqBiLSTMEncoderRuntime`, `ugtsdti/runtime/defaults.py` | `tests/quality/test_flow1_nodes.py::TestSeqBiLSTMEncoderRuntime` |
| `REQ-FLOW1-002` `encoder.gnn_drug` + missing modality | `.kiro/specs/ugtsdti-flow1-model/design.md` Sections 5.2, 6 | `graph.nodes[*].type: encoder.gnn_drug`, `params.allow_missing_input` | `ugtsdti/nodes/flow1.py::GNNDrugEncoderRuntime`, `ugtsdti/core/context.py` | `tests/quality/test_flow1_nodes.py::TestGNNDrugEncoderRuntime`, `tests/integration/test_flow1_pipeline.py::test_flow1_s2_*` |
| `REQ-FLOW1-003` `head.dense` | `.kiro/specs/ugtsdti-flow1-model/design.md` Section 5.3 | `graph.nodes[*].type: head.dense` | `ugtsdti/nodes/flow1.py::DenseHeadRuntime`, `ugtsdti/runtime/defaults.py` | `tests/quality/test_flow1_nodes.py::TestDenseHeadRuntime` |
| `REQ-FLOW1-004` Registry 3 type keys | `.kiro/specs/ugtsdti-flow1-model/design.md` Section 11 | `encoder.seq_bilstm`, `encoder.gnn_drug`, `head.dense` | `ugtsdti/runtime/defaults.py::build_default_graph_registry` | `tests/quality/test_flow1_nodes.py::TestFlow1Registry` |
| `REQ-FLOW1-005` `fusion.concat` N ≥ 2 | `.kiro/specs/ugtsdti-flow1-model/design.md` Section 5.4 | `graph.nodes[*].type: fusion.concat`, N inputs | `ugtsdti/nodes/baseline.py::ConcatFusionRuntime` | `tests/quality/test_flow1_nodes.py::TestConcatFusionNInput` |
| `REQ-FLOW1-006` Teacher graph topology | `.kiro/specs/ugtsdti-flow1-model/design.md` Section 4.1 | `graph.nodes` teacher branch, `roles.teacher` | `configs/flow1_teacher_student.yaml` | `tests/integration/test_flow1_pipeline.py::test_flow1_s1_full_pipeline_state_keys` |
| `REQ-FLOW1-007` Student graph topology | `.kiro/specs/ugtsdti-flow1-model/design.md` Section 4.2 | `graph.nodes` student branch, `roles.student` | `configs/flow1_teacher_student.yaml` | `tests/integration/test_flow1_pipeline.py::test_flow1_student_pipeline_runs_without_drug_graph` |
| `REQ-FLOW1-008` Execution phase ordering | `.kiro/specs/ugtsdti-flow1-model/design.md` Section 3.2 | `interaction.inputs` = role outputs | `ugtsdti/interaction/engine.py`, `ugtsdti/trainer/trainer.py` | `tests/integration/test_flow1_pipeline.py` |
| `REQ-FLOW1-009` Decision explicit inputs | `.kiro/specs/ugtsdti-flow1-model/design.md` Section 4.4 | `decision.inputs`, `decision.fallback` | `ugtsdti/decision/module.py::SoftBlendingDecisionModule` | `tests/integration/test_flow1_pipeline.py::test_flow1_decision_fallback_no_teacher` |
| `REQ-FLOW1-010` KD logits + numerical stability | `.kiro/specs/ugtsdti-flow1-model/design.md` Section 5.5 | `interaction.kd.params.temperature` | `ugtsdti/interaction/kd.py::binary_logits_to_dist` | `tests/integration/test_flow1_pipeline.py::test_flow1_s1_all_outputs_finite` |
| `REQ-FLOW1-011` Uncertainty confidence proxy | `.kiro/specs/ugtsdti-flow1-model/design.md` Section 4.3 | `interaction.uncertainty.type: uncertainty.confidence_proxy` | `ugtsdti/interaction/uncertainty.py::ConfidenceProxyUncertaintyInteraction` | `tests/integration/test_flow1_pipeline.py::test_flow1_s1_gate_alpha_in_range` |
| `REQ-FLOW1-012` Soft blending decision | `.kiro/specs/ugtsdti-flow1-model/design.md` Section 4.4 | `decision.strategy: soft`, `decision.use_uncertainty` | `ugtsdti/decision/module.py::SoftBlendingDecisionModule` | `tests/integration/test_flow1_pipeline.py::test_flow1_s1_gate_alpha_in_range` |
| `REQ-FLOW1-013` Missing modality S2/S4 | `.kiro/specs/ugtsdti-flow1-model/design.md` Section 6 | `encoder.gnn_drug.params.allow_missing_input: true` | `ugtsdti/nodes/flow1.py::GNNDrugEncoderRuntime`, `ugtsdti/core/context.py` | `tests/integration/test_flow1_pipeline.py::test_flow1_s2_*` |
| `REQ-FLOW1-014` KD warmup lambda_max | `.kiro/specs/ugtsdti-flow1-model/design.md` Section 5.6 | `training.kd.schedule: warmup`, `training.kd.lambda_max` | `ugtsdti/trainer/trainer.py::_scheduled_kd_weight` | `tests/quality/test_flow1_nodes.py::TestScheduledKdWeight`, `tests/integration/test_flow1_pipeline.py::test_flow1_kd_warmup_*` |
| `REQ-FLOW1-015` Teacher freeze mode | `.kiro/specs/ugtsdti-flow1-model/design.md` Section 11 | `training.teacher.freeze: true/false` | `ugtsdti/trainer/trainer.py::_resolve_freeze_policy`, `_apply_freeze_policy` | `tests/integration/test_flow1_pipeline.py::test_flow1_frozen_teacher_*` |
| `REQ-FLOW1-016` Key namespace compliance | `.kiro/specs/ugtsdti-flow1-model/design.md` Section 3.1 | all node/role/interaction/decision output keys | `ugtsdti/core/schema.py`, `ugtsdti/roles/binder.py` | `tests/integration/test_flow1_pipeline.py` |
| `REQ-FLOW1-017` Validation error taxonomy | `.kiro/specs/ugtsdti-flow1-model/design.md` Section 9 | N/A | `ugtsdti/core/errors.py` | `tests/quality/test_flow1_nodes.py` (raises checks) |
| `REQ-FLOW1-018` Config `flow1_teacher_student.yaml` | `.kiro/specs/ugtsdti-flow1-model/design.md` Section 7 | `configs/flow1_teacher_student.yaml` | `configs/flow1_teacher_student.yaml` | `tests/integration/test_flow1_pipeline.py::test_flow1_config_loads_and_validates` |
| `REQ-FLOW1-019` Config `profiles/flow1_local.yaml` | `.kiro/specs/ugtsdti-flow1-model/design.md` Section 8 | `configs/profiles/flow1_local.yaml` | `configs/profiles/flow1_local.yaml` | `tests/integration/test_flow1_pipeline.py::test_flow1_local_profile_resolves_extends` |
| `REQ-FLOW1-020` Full pipeline S1–S4 | `.kiro/specs/ugtsdti-flow1-model/design.md` Sections 4, 6 | full config + scenario eval | `ugtsdti/trainer/trainer.py::PipelineExecutor` | `tests/integration/test_flow1_pipeline.py` |
