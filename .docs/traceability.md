# Traceability Matrix

## 1. Purpose

This matrix maps major requirements to their owning spec, config surface, implementation module, and validation test.

## 2. Matrix

| Requirement | Spec section | Config key | Module | Test |
| --- | --- | --- | --- | --- |
| Fixed pipeline order | `contracts.md` §2.1 | stage structure implicit in runtime | `trainer/trainer.py`, `trainer/evaluator.py` | `tests/integration/test_pipeline_order.py` |
| Teacher-student asymmetry | `architecture.md` §4, `contracts.md` §3 | `roles`, `training.teacher`, `training.student` | `roles/binder.py`, `trainer/trainer.py` | `tests/research/test_teacher_student_asymmetry.py` |
| Write-once state | `contracts.md` §2.5, §5 | n/a | `core/state.py` | `tests/quality/test_state_contract.py` |
| Single producer per key | `contracts.md` §2.4 | `graph.nodes.*` | `graph/builder.py` | `tests/quality/test_graph_builder.py` |
| Explicit dependency access | `contracts.md` §2.3, §5.3 | `graph.nodes.*.inputs` | `nodes/base.py`, `graph/engine.py` | `tests/quality/test_declared_inputs.py` |
| Graph node output naming | `contracts.md` §4.1 | `graph.nodes.*` | `nodes/base.py` | `tests/quality/test_node_keys.py` |
| Role binding to canonical logits | `contracts.md` §7 | `roles.*.outputs` | `roles/binder.py` | `tests/quality/test_roles.py` |
| Interaction graph ordering | `contracts.md` §8.3 | `interaction.order`, `interaction.dependencies` | `config/validate.py`, `interaction/base.py` | `tests/quality/test_interaction_graph.py` |
| KD as interaction and training signal | `contracts.md` §9 | `interaction.kd`, `loss.map.kd` | `interaction/kd.py`, `postprocess/loss.py` | `tests/quality/test_kd_interaction.py` |
| Feature and relation KD extensibility | `contracts.md` §9.3 | `interaction.kd.mode` | `interaction/kd.py` | `tests/research/test_kd_modes.py` |
| Uncertainty as interaction module | `contracts.md` §10 | `interaction.uncertainty` | `interaction/uncertainty.py` | `tests/quality/test_uncertainty_interaction.py` |
| Gate as trust mechanism | `architecture.md` §7, `contracts.md` §11 | `decision` | `decision/gate.py` | `tests/research/test_gate_trust_semantics.py` |
| Alpha bound enforcement | `contracts.md` §11.4 | `decision.type`, `decision.strategy` | `decision/gate.py` | `tests/quality/test_gate_alpha_bounds.py` |
| Teacher freeze dynamics | `contracts.md` §12.1 | `training.teacher.freeze` | `trainer/trainer.py` | `tests/quality/test_teacher_freeze.py` |
| KD schedule dynamics | `contracts.md` §12.3 | `training.kd.schedule` | `trainer/trainer.py` | `tests/research/test_kd_schedule.py` |
| Gate trainability dynamics | `contracts.md` §12.4 | `training.gate.trainable` | `trainer/trainer.py`, `decision/gate.py` | `tests/quality/test_gate_trainability.py` |
| Interaction-to-loss mapping | `contracts.md` §13 | `loss.map` | `postprocess/loss.py` | `tests/quality/test_loss_mapping.py` |
| Scenario-aware KD/gate/loss behavior | `contracts.md` §14 | `scenario.policy` | `interaction/kd.py`, `decision/gate.py`, `postprocess/loss.py` | `tests/research/test_scenario_aware_behavior.py` |
| Diagnostics and observability | `contracts.md` §15, `experiment.md` §10 | `metrics.diagnostics`, `logging.*` | `postprocess/metrics.py`, `graph/engine.py` | `tests/research/test_diagnostics.py` |
| Experiment DSL inheritance | `config_schema.md` §4.1 | `extends` | `config/loader.py`, `config/normalize.py` | `tests/quality/test_config_extends.py` |
| Experiment sweep DSL | `config_schema.md` §4.2 | `sweep` | `config/schema.py`, `config/normalize.py` | `tests/research/test_sweep_dsl.py` |
| Four-scenario reporting | `experiment.md` §4 | `scenario.eval`, `metrics.by_scenario` | `trainer/evaluator.py`, `postprocess/metrics.py` | `tests/research/test_four_scenarios.py` |

## 3. Maintenance Rule

Any new requirement should be added here together with its owning spec section, config surface, target module, and validation test.
