# Architectural Decision Record

## ADR-001 - Fixed Pipeline With Pluggable Components

### Context

The framework must support research variation without letting experiments rewrite the system structure.

### Decision

Lock the end-to-end pipeline to:

```text
Batch -> Graph -> Role Binding -> Interaction -> Decision -> Loss/Metrics
```

Allow variation only through plugins and configuration inside each stage.

### Alternatives considered

- fully custom experiment pipelines;
- graph-only systems with no explicit semantic layers;
- trainer-defined ad hoc orchestration.

### Consequences

Positive:

- stable reasoning model;
- easier reproducibility;
- clearer debugging and traceability.

Trade-offs:

- architectural changes require spec updates;
- some experiments may feel constrained until a new plugin exists.

## ADR-002 - Graph-Based Compute Instead Of Hand-Wired Imperative Models

### Context

The framework must support branching, shared computation, and reusable intermediate outputs.

### Decision

Use a validated DAG of nodes for the graph stage.

### Consequences

Positive:

- explicit dependencies;
- reusable components;
- deterministic planning.

Trade-offs:

- builder and planner complexity;
- stronger need for naming discipline.

## ADR-003 - Teacher And Student As Semantic Roles

### Context

Teacher and student are semantic branches, not graph-engine primitives.

### Decision

Represent teacher and student through role binding after graph execution.

### Consequences

Positive:

- graph remains generic;
- branch meaning is explicit;
- teacher-only and student-only variants remain clean.

## ADR-004 - KD As Interaction, Not Only As Loss

### Context

KD is a branch interaction and may need to emit more than a single scalar term.

### Decision

Model KD in the interaction layer. Loss composition may consume KD-derived outputs later, but KD itself is not reduced to loss configuration alone.

### Consequences

Positive:

- more faithful research abstraction;
- easier KD ablations and diagnostics.

Trade-offs:

- one extra plugin stage to manage.

## ADR-005 - Uncertainty As Interaction, Gate As Decision

### Context

Uncertainty estimation and final branch decision are related but not identical concerns.

### Decision

Compute uncertainty in the interaction layer and reserve the decision layer for final gating or blending.

### Consequences

Positive:

- cleaner responsibility boundaries;
- easier comparison of uncertainty estimators versus gate strategies.

Trade-offs:

- more explicit inter-stage wiring through state.

## ADR-006 - Contracts Centralized In One Document

### Context

Distributing invariants across many documents causes drift.

### Decision

Keep all normative invariants and interfaces in [contracts.md](contracts.md). Other docs may reference them but do not redefine them.

### Consequences

Positive:

- easier review;
- fewer contradictory rules;
- clearer implementation target.
