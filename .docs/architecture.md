# Architecture

## 1. Purpose

This document defines the structural architecture of the UGTSDTI framework as a modality-aware teacher-student research system for drug-target interaction prediction.

It specifies:

- the fixed end-to-end pipeline;
- the original design intuition behind teacher and student roles;
- the architectural meaning of modality asymmetry;
- the role of KD, uncertainty, and gating in the system;
- the scaling path from two-branch to future multi-branch settings.

Normative interface and invariants are defined in [contracts.md](contracts.md). This document defines structure, semantics, and architectural boundaries.

## 2. Research Identity

UGTSDTI is not a generic multimodal ensemble framework. It is a research system built to study:

- teacher versus student behavior;
- knowledge transfer under branch asymmetry;
- uncertainty as a reliability signal;
- trust-based decision making between branches;
- generalization under warm and cold DTI scenarios.

The architecture therefore preserves research semantics explicitly rather than hiding them behind generic fusion abstractions.

## 3. Original Design Flows

The system originates from two canonical flows that must remain visible in the architecture.

### 3.1 Flow 1: reduced-modality student

```text
Teacher = sequence + structure
Student = sequence only
```

This flow represents a student as a reduced-modality version of the teacher.

### 3.2 Flow 2: modality-specific student

```text
Teacher = sequence + structure
Student = structure only
```

This flow represents a student as a modality-specific branch rather than merely a smaller copy.

### 3.3 Architectural implication

The student is not fixed.

```text
Student is a design space.
```

The architecture must support at least the following student families:

- modality-reduced student;
- modality-specific student;
- architecture-reduced student.

This requirement affects configuration, role semantics, KD design, and evaluation.

## 4. Modality Model

The architecture treats modalities as first-class research entities.

Canonical examples include:

- sequence modality;
- structure modality;
- future auxiliary modalities if introduced later.

### 4.1 Teacher modality semantics

The teacher may act as:

- a multi-modal aggregator;
- a privileged branch with richer input coverage;
- a stronger branch with more complete modality context.

### 4.2 Student modality semantics

The student may act as:

- a partial-modality branch;
- a reduced-modality branch;
- a restricted-capacity branch under the same or reduced modality set.

### 4.3 Architectural consequence

Nodes operate on modality-bearing inputs. Branch asymmetry may therefore arise from:

- different node families;
- different modality access;
- different capacity or pretraining state.

KD and gate behavior may need to reason about modality completeness rather than only about logits.

## 5. Canonical End-To-End Pipeline

The canonical pipeline is fixed:

```text
Batch
-> Graph (nodes)
-> Role Binding
-> Interaction (KD + Uncertainty)
-> Decision (Gate)
-> Loss + Metrics
```

This stage order is architectural and must not change across experiments.

### 5.1 Batch

The batch carries:

- raw modality inputs;
- labels;
- scenario metadata;
- optional research metadata required for diagnostics or analysis.

### 5.2 Graph

The graph stage executes computational nodes over modality-aware inputs and produces deterministic named outputs under graph namespaces.

### 5.3 Role Binding

Role binding maps graph outputs into semantic branches such as `teacher` and `student`.

### 5.4 Interaction

The interaction stage computes branch interactions after roles are bound and before final decision is made. It owns:

- KD signal construction;
- uncertainty estimation;
- disagreement and reliability diagnostics;
- future feature-level and relation-level interaction mechanisms.

### 5.5 Decision

The decision stage consumes role outputs and interaction outputs to produce:

- final `logits`;
- `gate.alpha` when defined;
- gate diagnostics that describe branch trust behavior.

### 5.6 Loss + Metrics

The final stage consumes final outputs, labels, and optionally interaction diagnostics to compute:

- hard and KD-aware losses;
- scenario-wise metrics;
- research observability outputs.

## 6. Teacher-Student Asymmetry

Teacher and student are intentionally asymmetric.

### 6.1 Teacher role

The teacher may:

- use richer modality;
- be pretrained;
- be frozen or partially frozen;
- be transductive or otherwise privileged.

The teacher is therefore not merely another peer branch.

### 6.2 Student role

The student is the canonical deployable or inductive branch and should remain trainable in standard research runs.

The student is expected to:

- learn under stronger deployment constraints;
- operate with equal or reduced modality access relative to the teacher;
- remain the main recipient of KD.

### 6.3 Architectural implication

Because the two roles are asymmetric, the architecture must not:

- treat them as exchangeable ensemble members;
- reduce gate semantics to generic mixture logic;
- reduce KD semantics to symmetric branch regularization.

## 7. Interaction Semantics

The interaction stage is responsible for making branch relations explicit rather than implicit.

### 7.1 KD as interaction

KD belongs in the interaction stage because it captures directed knowledge transfer:

```text
teacher -> student
```

KD is both:

- a structural interaction between branches;
- a source of training signal later consumed by the loss stage.

### 7.2 Uncertainty as interaction

Uncertainty also belongs in the interaction stage because it measures branch reliability before the final trust decision is made.

### 7.3 Interaction graph

The interaction stage should be viewed as a small ordered interaction graph rather than a flat unordered block.

For example:

- uncertainty may depend on role outputs only;
- KD may depend on role outputs and optionally uncertainty outputs;
- future interaction modules may depend on earlier interaction diagnostics.

This interaction graph must remain acyclic and deterministic.

## 8. Gate Semantics As Trust Mechanism

The gate is not a generic fusion function.

Architecturally, the gate answers:

> given the current branch outputs, modality completeness, and interaction signals, how much should the system trust each branch?

This trust may reflect:

- uncertainty;
- modality completeness;
- branch reliability under the current scenario;
- learned branch preference under training.

### 8.1 Gate families

The architecture supports:

- hard gating;
- soft gating;
- learned gates;
- heuristic gates.

### 8.2 Two-branch form

For the current canonical case:

```math
y = \\alpha y_{teacher} + (1 - \\alpha) y_{student}
```

where `alpha` represents teacher trust in the soft-gating case.

### 8.3 Future N-branch form

The architecture must scale to an N-branch system in which:

```math
y = \\sum_i \\alpha_i y_i
```

with:

```math
\\alpha_i \\ge 0, \\quad \\sum_i \\alpha_i = 1
```

The two-branch teacher-student setting is therefore a special case of a future trust-allocation mechanism, not a dead-end design.

## 9. Scenario-Aware Architecture

Scenario metadata such as `S1` to `S4` is part of the batch and may affect later stages without changing graph structure.

Scenario may legitimately influence:

- KD weighting or activation;
- uncertainty interpretation;
- gate trust behavior;
- loss weighting;
- evaluation slicing.

Scenario must not:

- change graph topology;
- silently redefine teacher and student identities.

## 10. Diagnostics And Research Observability

The architecture is designed to expose signals that allow paper-level analysis of:

- teacher versus student disagreement;
- gate.alpha distributions;
- uncertainty versus observed error;
- branch reliability under different modality settings;
- KD effectiveness under different scenarios and schedules.

These diagnostics are part of the scientific output of the system, not just implementation-level logging.

## 11. Multi-Branch Future Extension

Although the current canonical setup is one teacher and one student, the architecture should extend to:

- multiple teachers;
- multiple students;
- mixed teacher and student ensembles under the same pipeline.

This extension should be achieved by:

- richer role binding;
- richer interaction graphs;
- vector-valued gates;

without rewriting the main pipeline stages.

## 12. Package Topology

The architecture maps to the following package layout:

```text
ugtsdti/
  core/
    state.py
    context.py
    errors.py

  config/
    schema.py
    loader.py
    validate.py
    normalize.py

  graph/
    builder.py
    planner.py
    engine.py

  nodes/
    base.py
    registry.py
    encoder/
    fusion/
    head/

  roles/
    binder.py

  interaction/
    base.py
    kd.py
    uncertainty.py

  decision/
    base.py
    gate.py

  postprocess/
    loss.py
    metrics.py

  trainer/
    trainer.py
    evaluator.py
```

## 13. Summary

UGTSDTI is a modality-aware, uncertainty-guided teacher-student research architecture. Its purpose is not only to produce predictions but also to preserve and study the asymmetry between branches, the effect of KD, the reliability of uncertainty, and the quality of trust-based decision making.
