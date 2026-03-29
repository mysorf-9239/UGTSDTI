# Traceability Matrix

## Purpose

Matrix nay map `REQ-HARD-*` cua pha research hardening sang design, config surface, implementation, va tests dang cover.

## Matrix

| Requirement | Design / Docs | Config Surface | Implementation | Validation |
| --- | --- | --- | --- | --- |
| `REQ-HARD-001` Scenario protocol correctness | `.kiro/specs/ugtsdti-research-hardening/design.md` Section 4 | `data.*`, `scenario.*`, split manifests `scenario_partitions.*` | `ugtsdti/data/contracts.py`, `ugtsdti/data/splitting.py`, `ugtsdti/data/loader.py` | `tests/quality/test_data.py`, `tests/pbt/test_data_properties.py` |
| `REQ-HARD-002` Split leakage validation | `.kiro/specs/ugtsdti-research-hardening/design.md` Section 4.4 | split manifest `protocol_report`, partition artifacts | `ugtsdti/data/validate.py` | `tests/quality/test_data.py`, `tests/pbt/test_data_properties.py` |
| `REQ-HARD-003` Runtime interaction contract enforcement | `.kiro/specs/ugtsdti-research-hardening/design.md` Section 5 | `interaction.order`, `interaction.dependencies`, plugin output keys | `ugtsdti/interaction/engine.py`, `ugtsdti/interaction/registry.py`, `ugtsdti/trainer/trainer.py` | `tests/quality/test_interaction.py`, `tests/integration/test_pipeline.py` |
| `REQ-HARD-004` State mutability discipline | `.kiro/specs/ugtsdti-research-hardening/design.md` Section 2.3 | implicit state boundary contract | `ugtsdti/core/state.py` | `tests/quality/test_state.py` |
| `REQ-HARD-005` Identity consistency | `.kiro/specs/ugtsdti-research-hardening/design.md` Section 6 | CLI run identity, logs dir, artifact/checkpoint identity metadata | `ugtsdti/cli/main.py`, `ugtsdti/runtime/identity.py`, `ugtsdti/runtime/artifacts.py`, `ugtsdti/runtime/checkpoint.py` | `tests/integration/test_cli.py`, `tests/quality/test_runtime.py` |
| `REQ-HARD-006` Reproducibility key integration | `.kiro/specs/ugtsdti-research-hardening/design.md` Section 6 | `data.dataset`, `data.preprocessing_version`, `data.split_version`, `runtime.seed` | `ugtsdti/cli/main.py`, `ugtsdti/runtime/identity.py`, `ugtsdti/runtime/checkpoint.py` | `tests/integration/test_cli.py`, `tests/quality/test_runtime.py` |
| `REQ-HARD-007` Metric correctness | `.kiro/specs/ugtsdti-research-hardening/design.md` Section 7 | `metrics.enabled`, `metrics.by_scenario` | `ugtsdti/postprocess/metrics.py` | `tests/quality/test_postprocess.py` |
| `REQ-HARD-008` Uncertainty semantic honesty | `.kiro/specs/ugtsdti-research-hardening/design.md` Section 8 | `interaction.*.type: uncertainty.sample_variance | uncertainty.confidence_proxy` | `ugtsdti/interaction/uncertainty.py`, `ugtsdti/decision/trust.py`, `ugtsdti/config/validate.py` | `tests/quality/test_interaction.py`, `tests/quality/test_decision.py`, `tests/integration/test_pipeline.py` |
| `REQ-HARD-009` Trust / gate semantic clarity | `.kiro/specs/ugtsdti-research-hardening/design.md` Section 9 | `decision.mode`, `decision.strategy`, `training.gate.trainable` | `ugtsdti/config/normalize.py`, `ugtsdti/config/validate.py`, `ugtsdti/decision/module.py`, `ugtsdti/trainer/trainer.py` | `tests/quality/test_config.py`, `tests/integration/test_orchestration.py` |
| `REQ-HARD-010` Training loop minimum research usability | `.kiro/specs/ugtsdti-research-hardening/design.md` Section 10 | `training.loop.*`, `training.optimizer.*`, `runtime.checkpoint_path` | `ugtsdti/cli/main.py`, `ugtsdti/runtime/adapter.py`, `ugtsdti/runtime/defaults.py`, `ugtsdti/trainer/trainer.py`, `ugtsdti/trainer/evaluator.py` | `tests/integration/test_cli.py`, `tests/integration/test_orchestration.py`, `tests/quality/test_runtime.py` |
| `REQ-HARD-011` Custom plugin usability | `.kiro/specs/ugtsdti-research-hardening/design.md` Section 11 | `runtime.plugin_registrars` | `ugtsdti/runtime/plugins.py`, `ugtsdti/cli/main.py`, `ugtsdti/config/validate.py` | `tests/integration/test_cli.py` |
| `REQ-HARD-012` Traceability update | `.kiro/specs/ugtsdti-research-hardening/requirements.md`, `.kiro/specs/ugtsdti-research-hardening/tasks.md` | N/A | `.docs/traceability.md`, spec docs | `tests/integration/test_cli.py`, `tests/quality/test_config.py`, `tests/quality/test_runtime.py` |

## Notes

- Heuristic gate hien duoc chot boi `decision.mode: heuristic`; runtime learned gate chua duoc implement va validator fail-closed khi `training.gate.trainable: true`.
- Train loop hardening hien duoc phan anh boi `training.loop.{epochs, checkpoint_every_epochs, summary_every_steps, eval_every_epochs, eval_partition}`.
- Eval lineage co the nap `runtime.checkpoint_path`; `config_hash` canonicalization bo qua field nay de khong lam lech reproducibility lineage.
- Custom plugin validation hien tai la plugin-aware o CLI path: default registries duoc load truoc, sau do moi apply `runtime.plugin_registrars` neu co.
