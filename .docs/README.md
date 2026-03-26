# UGTSDTI System Design Specification

## 1. System Overview

UGTSDTI is a research-grade framework for drug-target interaction prediction built around a fixed execution pipeline and a plugin-based component model.

The framework is designed for experiments that combine:

- a graph-executed computational backbone;
- semantic `teacher` and `student` roles;
- an explicit interaction layer for knowledge distillation and uncertainty estimation;
- a separate decision layer that gates or blends branch outputs;
- reproducible training, evaluation, ablation, and tuning workflows.

The canonical system pipeline is fixed:

```text
Batch
-> Graph (nodes)
-> Role Binding
-> Interaction Layer (KD + Uncertainty)
-> Decision Layer (Gate)
-> Loss + Metrics
```

This pipeline does not change across experiments. Experiments vary by swapping plugins and configuration, not by rewriting the pipeline.

## 2. Design Principles

### 2.1 Fixed pipeline, pluggable internals

The execution stages are fixed. The following components are pluggable:

- encoders;
- fusion modules;
- heads;
- KD strategies;
- uncertainty estimators;
- gating strategies;
- loss composition;
- metric suites.

### 2.2 Determinism

Given the same configuration, seed, and input batch, the framework should produce the same execution plan and the same externally visible outputs.

### 2.3 Configuration-driven assembly

Graphs, roles, interaction modules, decision modules, and post-process behavior are assembled from validated config. Pipeline code should remain stable as experiments evolve.

### 2.4 Research extensibility

The framework should support rapid ablation and hyperparameter exploration without weakening architectural boundaries or reproducibility.

### 2.5 Traceability

Major requirements must be traceable across:

- design sections;
- config surface;
- target module;
- validation tests.

See [traceability.md](traceability.md).

## 3. Document Map

- [architecture.md](architecture.md)
  System structure, package boundaries, and the fixed end-to-end pipeline.

- [contracts.md](contracts.md)
  The single source of truth for invariants, interfaces, naming rules, and cross-layer constraints.

- [config_schema.md](config_schema.md)
  Public configuration API, normalization rules, and validation workflow.

- [node_spec.md](node_spec.md)
  Node lifecycle, categories, plugin rules, and graph-facing implementation expectations.

- [graph_execution.md](graph_execution.md)
  DAG construction, planning, execution, scheduling, tracing, and state inspection for the graph phase.

- [experiment.md](experiment.md)
  Research protocol, split assumptions, evaluation, ablation, reporting, and reproducibility guidance.

- [implementation_plan.md](implementation_plan.md)
  Design-aligned build phases and validation criteria for implementation.

- [traceability.md](traceability.md)
  Requirement to spec, config, module, and test mapping.

- [adr.md](adr.md)
  Architectural decision record for major design choices.

## 4. Recommended Reading Order

### 4.1 Before implementation

1. [architecture.md](architecture.md)
2. [contracts.md](contracts.md)
3. [config_schema.md](config_schema.md)
4. [node_spec.md](node_spec.md)
5. [graph_execution.md](graph_execution.md)
6. [implementation_plan.md](implementation_plan.md)

### 4.2 Before extending plugins

1. [contracts.md](contracts.md)
2. [node_spec.md](node_spec.md)
3. [config_schema.md](config_schema.md)
4. [adr.md](adr.md)

### 4.3 Before running experiments

1. [experiment.md](experiment.md)
2. [contracts.md](contracts.md)
3. [traceability.md](traceability.md)

## 5. How To Use This Spec

### 5.1 For implementation

- Use [architecture.md](architecture.md) to understand the pipeline stages.
- Use [contracts.md](contracts.md) to enforce invariants and interfaces.
- Use [config_schema.md](config_schema.md) to expose the right config surface.
- Use [traceability.md](traceability.md) to identify expected tests.

### 5.2 For extension

- Add new computation through graph nodes.
- Add KD and uncertainty behavior through interaction plugins.
- Add branch selection or blending through gate plugins.
- Update [adr.md](adr.md) if the change modifies a major architectural decision.

### 5.3 For debugging

- Start from [contracts.md](contracts.md) to validate assumptions.
- Use [graph_execution.md](graph_execution.md) for graph-phase trace and state inspection.
- Use [experiment.md](experiment.md) to separate protocol errors from runtime bugs.

## 6. Ownership Model Across Documents

| Concept | Source of truth |
| --- | --- |
| fixed system flow and package boundaries | `architecture.md` |
| invariants, interfaces, and cross-layer rules | `contracts.md` |
| config API and validation workflow | `config_schema.md` |
| graph node behavior | `node_spec.md` |
| graph construction and execution | `graph_execution.md` |
| experiment protocol and reporting | `experiment.md` |

Other files may reference these concepts but must not redefine them.

## 7. Scope Of This Spec

This specification defines:

- how the UGTS-DTI system is assembled;
- how fixed pipeline stages interact;
- how plugins are inserted without changing pipeline code;
- how experiments are configured and evaluated.

This specification does not claim that a particular model or dataset is scientifically optimal. It defines the framework required to test such hypotheses cleanly and reproducibly.
