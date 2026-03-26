# Implementation Plan

## 1. Scope

This plan translates the UGTSDTI system specification into an implementation sequence while preserving the fixed research pipeline.

Each phase specifies:

- referenced spec sections;
- module deliverables;
- validation criteria.

## 2. Build Strategy

Implementation order follows architectural dependency rather than convenience:

1. contracts and state system;
2. config system and experiment DSL;
3. graph nodes and graph engine;
4. role binding;
5. interaction graph;
6. decision layer;
7. loss and metrics;
8. trainer, evaluator, and experiment workflow.

## 3. Phase 0: Spec Lock-In

### References

- [README.md](README.md)
- [architecture.md](architecture.md)
- [contracts.md](contracts.md)
- [traceability.md](traceability.md)

### Deliverables

- fixed pipeline agreed and frozen;
- package layout agreed;
- traceability baseline created;
- ADR baseline aligned with pipeline semantics.

### Validation

- stage ownership is unambiguous;
- teacher-student asymmetry is explicit in docs;
- gate is documented as trust mechanism, not generic fusion.

## 4. Phase 1: Core State And Error System

### References

- [contracts.md](contracts.md) sections on state, naming, invariants, and failure taxonomy

### Deliverables

- `core/state.py`
- `core/context.py`
- `core/errors.py`

### Validation

- write-once state enforced;
- single producer checks available;
- deterministic trace hooks available.

## 5. Phase 2: Config System And Experiment DSL

### References

- [config_schema.md](config_schema.md)
- [contracts.md](contracts.md) config consistency rules

### Deliverables

- `config/schema.py`
- `config/loader.py`
- `config/validate.py`
- `config/normalize.py`

### Validation

- `extends` inheritance resolves deterministically;
- `sweep` targets are normalized consistently;
- interaction order and dependencies validate correctly;
- training dynamics config validates against teacher, student, KD, and gate rules.

## 6. Phase 3: Graph Node Framework

### References

- [node_spec.md](node_spec.md)
- [graph_execution.md](graph_execution.md)

### Deliverables

- `nodes/base.py`
- `nodes/registry.py`
- example plugins under `nodes/encoder`, `nodes/fusion`, `nodes/head`

### Validation

- node plugins register deterministically;
- graph outputs follow `<node>.<attr>` naming;
- graph nodes cannot read role or interaction outputs.

## 7. Phase 4: Graph Build, Planning, And Engine

### References

- [graph_execution.md](graph_execution.md)
- [contracts.md](contracts.md) execution rules

### Deliverables

- `graph/builder.py`
- `graph/planner.py`
- `graph/engine.py`

### Validation

- DAG validation works;
- deterministic plan is reproducible;
- execution trace and state inspection work.

## 8. Phase 5: Role Binding

### References

- [architecture.md](architecture.md)
- [contracts.md](contracts.md) role binding contract

### Deliverables

- `roles/binder.py`

### Validation

- teacher and student outputs bind correctly;
- future multi-role aggregation remains possible without changing graph semantics.

## 9. Phase 6: Interaction Graph

### References

- [architecture.md](architecture.md) interaction layer and interaction graph concept
- [contracts.md](contracts.md) interaction, KD, and uncertainty contracts

### Deliverables

- `interaction/base.py`
- `interaction/kd.py`
- `interaction/uncertainty.py`
- interaction execution helper if needed

### Validation

- KD exists as an interaction plugin, not a loss-only shortcut;
- uncertainty plugins consume role outputs only;
- interaction order and dependencies execute correctly;
- interaction diagnostics are available for later analysis.

## 10. Phase 7: Decision Layer

### References

- [architecture.md](architecture.md) gate semantics
- [contracts.md](contracts.md) gate contract

### Deliverables

- `decision/base.py`
- `decision/gate.py`

### Validation

- final `logits` are produced only after the gate stage;
- `gate.alpha` is bounded when present;
- hard/soft and learned/heuristic gate variants fit the same interface;
- fallback behavior is explicit and test-covered.

## 11. Phase 8: Loss And Metrics

### References

- [contracts.md](contracts.md) interaction-to-loss mapping, scenario-aware behavior, diagnostics
- [experiment.md](experiment.md)

### Deliverables

- `postprocess/loss.py`
- `postprocess/metrics.py`

### Validation

- `loss.total` is produced in training mode;
- KD-derived interaction outputs can map into loss explicitly;
- scenario-aware metrics and diagnostics are supported.

## 12. Phase 9: Trainer And Evaluator

### References

- [architecture.md](architecture.md)
- [experiment.md](experiment.md)

### Deliverables

- `trainer/trainer.py`
- `trainer/evaluator.py`

### Validation

- end-to-end pipeline follows the fixed stage order;
- teacher freeze, KD schedule, and gate trainability behave as configured;
- repeated runs with same seed are reproducible.

## 13. Phase 10: Research Workflow Validation

### References

- [experiment.md](experiment.md)
- [traceability.md](traceability.md)
- [adr.md](adr.md)

### Deliverables

- baseline configs;
- ablation configs for KD, uncertainty, gate, and scenario-aware behavior;
- tuning configs using `sweep`.

### Validation

- `S1` to `S4` reporting works;
- branch diagnostics appear in reports;
- interaction and gate ablations remain isolated;
- teacher-student asymmetry remains visible in outputs and analysis.

## 14. Summary

The implementation order exists to preserve architecture fidelity. If implementation pressure suggests collapsing interaction, gate, or asymmetry semantics into generic helpers, the design should be revisited first.
