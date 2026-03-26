# Contracts

## 1. Scope

This document is the single source of truth for:

- invariants;
- interfaces;
- state rules;
- naming rules;
- cross-layer consistency rules;
- failure categories.

If another document conflicts with this file, this file takes precedence.

## 2. Global Invariants

### 2.1 Fixed pipeline invariant

The architectural stage order is fixed:

```text
Batch -> Graph -> Role Binding -> Interaction -> Decision -> Loss/Metrics
```

No experiment may reorder, merge, or bypass these stages. A stage may contain a no-op implementation, but the stage boundary remains present.

### 2.2 Determinism invariant

For a fixed config, seed, runtime mode, and batch, the framework shall produce the same execution plan and the same externally visible outputs.

### 2.3 Explicit dependency invariant

Every computed key must have a declared producer and every consumer must access it through declared inputs or formally defined stage interfaces.

### 2.4 Single producer invariant

Each key in state has exactly one producer per forward pass.

### 2.5 Write-once invariant

State keys are write-once within a forward pass.

## 3. Teacher-Student Asymmetry Contract

### 3.1 Teacher asymmetry

The teacher branch may:

- use richer modality;
- be pretrained;
- be frozen or partially frozen;
- be transductive or otherwise privileged.

### 3.2 Student asymmetry

The student branch is the canonical deployable or inductive branch and should remain trainable in standard research configurations unless an explicit ablation overrides that expectation.

### 3.3 Student design space

The student is not fixed. The architecture must allow at least:

- modality-reduced students;
- modality-specific students;
- architecture-reduced students.

### 3.4 Directionality

Teacher and student are not interchangeable roles. The canonical KD direction is:

```text
teacher -> student
```

## 4. Modality Contract

### 4.1 Modality semantics

Modalities such as sequence and structure are first-class research entities.

### 4.2 Node relation to modality

Graph nodes operate on modality-bearing inputs and may therefore differ in:

- modality access;
- modality completeness;
- modality aggregation behavior.

### 4.3 KD relation to modality

KD may be:

- intra-modality KD, when teacher and student reason over aligned modality views;
- cross-modality KD, when knowledge transfer crosses modality boundaries and therefore requires alignment assumptions.

## 5. Naming Contract

### 5.1 Graph outputs

Graph node outputs must use:

```text
<node>.<attr>
```

### 5.2 Role outputs

Role outputs must use:

```text
<role>.logits
```

Canonical roles are `teacher` and `student`.

### 5.3 Interaction outputs

Interaction outputs must use stable explicit names. Recommended examples include:

- `kd.teacher_target`
- `kd.student_target`
- `interaction.kd.loss_component`
- `teacher.var`
- `student.var`
- `interaction.disagreement`

### 5.4 Decision outputs

Decision outputs must expose final prediction keys under stable names. Canonical outputs are:

- `logits`
- `gate.alpha` when applicable
- `gate.*` diagnostics

### 5.5 Diagnostic outputs

Structured diagnostics should be exposed under stable names such as:

- `diagnostics.disagreement`
- `diagnostics.gate_alpha`
- `diagnostics.uncertainty_error`

### 5.6 Loss and metrics

- loss outputs must use `loss.*`
- metric outputs must use `metrics.*` or a stable grouped `metrics` object

## 6. State Contract

### 6.1 Definition

`state` is the single communication medium across pipeline stages.

```python
State = Dict[str, Any]
```

### 6.2 Monotonicity

State grows monotonically during a forward pass. New keys may be added, but existing keys may not be overwritten.

### 6.3 Access discipline

A component may read only:

- batch keys;
- keys declared in its input list;
- keys produced by prior stages.

Undeclared state access is forbidden.

### 6.4 Observability discipline

The framework should support state inspection at graph, interaction, and decision boundaries without mutating state semantics.

## 7. Graph Node Contract

### 7.1 Interface

```python
class Node:
    name: str
    inputs: list[str]

    def forward(self, state: dict) -> dict:
        return {"outputs": {...}}
```

### 7.2 Graph constraints

- graph nodes may read batch keys and graph outputs only;
- graph nodes may not read role outputs, interaction outputs, or gate outputs;
- graph nodes must not access undeclared state;
- graph nodes must emit explicitly named outputs.

## 8. Role Binding Contract

### 8.1 Function

Role binding maps graph outputs to semantic branches.

### 8.2 Canonical output rule

Each configured role must expose one canonical `<role>.logits` tensor after binding.

### 8.3 Multi-role future rule

The system may later support multiple teachers or students, but each downstream semantic role must still expose a well-defined canonical output surface after aggregation.

## 9. Interaction Contract

### 9.1 Interface

```python
class Interaction:
    name: str
    inputs: list[str]

    def forward(self, state: dict) -> dict:
        return {"outputs": {...}}
```

### 9.2 Ownership

Interaction modules own cross-branch computation that is neither graph computation nor final gate decision.

### 9.3 Interaction graph contract

The interaction stage must support an ordered interaction graph rather than a flat unordered list.

The interaction configuration may declare:

- explicit execution order;
- explicit interaction dependencies.

### 9.4 Interaction dependency rules

An interaction may depend on:

- role outputs;
- prior interaction outputs.

An interaction may not depend on:

- future interaction outputs;
- final gate outputs;
- implicit undeclared state.

### 9.5 Interaction execution guarantees

The interaction graph must be:

- acyclic;
- dependency-validated before execution;
- deterministic in execution order under the same config.

## 10. KD Contract

### 10.1 Semantic status

KD is both:

- an interaction module that produces structured supervision signals;
- a training signal that contributes to final loss.

KD must not be modeled as a loss term only.

### 10.2 Direction

The canonical direction is teacher to student.

### 10.3 Core parameters

KD must support formalization of:

- temperature, where `temperature > 0`;
- weighting between hard and KD losses;
- optional feature KD;
- optional relation KD.

### 10.4 KD types

KD should distinguish at least:

- intra-modality KD;
- cross-modality KD, which requires an alignment assumption between teacher and student representations.

### 10.5 Output rule

A KD interaction should emit explicit outputs required for later loss composition or analysis, such as softened targets, alignment statistics, or KD loss components.

### 10.6 Mathematical form

The canonical KD-augmented objective is:

```math
L = (1 - \lambda) L_{hard} + \lambda L_{KD}
```

where `lambda` is the KD contribution weight.

## 11. Uncertainty Contract

### 11.1 Semantic status

Uncertainty estimation is an interaction module, not a gate and not a loss.

### 11.2 Targeting rule

Uncertainty may be computed for:

- teacher only;
- student only;
- both branches.

### 11.3 Output rule

Uncertainty outputs must be branch-explicit and stable, for example `teacher.var` and `student.var`.

### 11.4 Generic formalization

Uncertainty may be represented by predictive variance, entropy, or another documented reliability statistic.

## 12. Gate Contract

### 12.1 Interface

```python
class Gate:
    name: str
    inputs: list[str]

    def forward(self, state: dict) -> dict:
        return {
            "outputs": {
                "logits": ...,
                "gate.alpha": ...,
            }
        }
```

### 12.2 Semantic status

The gate is a trust estimator between branches, not a generic blending helper.

### 12.3 Supported semantics

A gate may be:

- hard or soft;
- learned or heuristic;
- scenario-aware or scenario-agnostic.

### 12.4 Two-branch alpha bound

If scalar `gate.alpha` is produced for the two-branch case, it must satisfy:

```text
0 <= alpha <= 1
```

### 12.5 N-branch generalization

For an N-branch future extension, the gate must support a trust vector `alpha` such that:

```math
\alpha_i \ge 0, \quad \sum_i \alpha_i = 1
```

### 12.6 Fallback rule

The gate must define explicit behavior when:

- teacher is absent;
- student is absent;
- uncertainty is unavailable;
- the selected strategy cannot compute its preferred signal.

### 12.7 Mathematical form

For the canonical two-branch soft gate:

```math
y = \alpha y_{teacher} + (1 - \alpha) y_{student}
```

## 13. Training Dynamics Contract

### 13.1 Teacher trainability

Teacher training behavior must be explicit. Supported cases include:

- frozen teacher;
- trainable teacher;
- partially trainable teacher.

### 13.2 Student trainability

The student should remain trainable in canonical research configurations.

### 13.3 KD schedule

KD contribution may be scheduled over training, for example:

- warmup;
- constant;
- staged activation.

This schedule is epoch-aware or step-aware and must be explicit in config.

### 13.4 Gate trainability

Gate trainability must be explicit. A gate may be:

- trainable;
- frozen;
- purely heuristic.

## 14. Interaction-To-Loss Mapping Contract

The mapping from interaction outputs to final loss must be explicit.

Example:

```text
interaction.kd.loss_component -> loss.total
```

A loss configuration may weight or combine interaction-derived loss components, but it may not assume hidden interaction behavior.

## 15. Scenario-Aware Behavior Contract

Scenario may influence:

- KD activation or weighting;
- uncertainty interpretation;
- gate behavior;
- loss weighting;
- metrics slicing.

Scenario may not:

- change graph topology;
- silently swap teacher and student semantics.

## 16. Diagnostics And Observability Contract

The framework should expose diagnostics sufficient to study:

- teacher versus student disagreement;
- gate.alpha distributions;
- uncertainty versus observed error;
- KD effectiveness over time and scenario.

These diagnostics are part of the research contract, not mere logging noise.

## 17. Execution Contract

### 17.1 Stage order

The canonical stage order is:

1. initialize state from batch
2. execute graph
3. bind roles
4. execute interaction graph
5. execute decision layer
6. compute loss and metrics

### 17.2 Validation boundaries

- graph validates dependencies and output collisions;
- roles validate binding outputs;
- interaction validates order, dependencies, and prerequisites;
- gate validates prerequisites and alpha constraints;
- loss validates final composition inputs.

## 18. Config Consistency Contract

The config system must validate:

- roles reference graph outputs;
- KD configuration is compatible with teacher and student availability;
- uncertainty targets are compatible with bound roles;
- decision requirements match available interaction outputs;
- training dynamics are compatible with configured teacher, student, KD, and gate behavior;
- loss composition references valid interaction outputs;
- interaction graphs are acyclic and dependency-valid.

## 19. Failure Taxonomy

Canonical failure categories include:

- missing dependency;
- key collision;
- invalid role binding;
- invalid interaction order;
- invalid interaction configuration;
- invalid gate output;
- invalid alpha range;
- invalid loss mapping;
- uncertainty miscalibration diagnostics;
- numerical instability.
