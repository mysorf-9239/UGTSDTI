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
