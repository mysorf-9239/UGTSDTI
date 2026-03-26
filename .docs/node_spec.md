# Node Specification

## 1. Scope

This document defines graph node behavior only.

It covers:

- node lifecycle;
- node plugin categories;
- declared-input and declared-output expectations;
- graph-facing extension rules.

It does not define role, interaction, or gate behavior. Those belong to [contracts.md](contracts.md) and [architecture.md](architecture.md).

## 2. Node Position In The Pipeline

Nodes belong exclusively to the graph stage:

```text
Batch -> Graph (nodes) -> Role Binding -> Interaction -> Decision -> Loss/Metrics
```

A graph node may produce outputs that later become teacher or student logits, but it does not know that semantic meaning itself.

## 3. Node Interface

```python
class Node:
    name: str
    inputs: list[str]

    def forward(self, state: dict) -> dict:
        return {"outputs": {...}}
```

## 4. Lifecycle

### 4.1 Setup

Optional setup may prepare:

- parameters;
- device placement;
- static resources.

### 4.2 Forward

`forward()` consumes declared graph-phase inputs and emits named outputs.

### 4.3 Teardown

Optional teardown may release resources.

## 5. Node Categories

### 5.1 Encoders

Encoders transform raw modalities into latent representations.

Examples:

- sequence encoders;
- graph encoders;
- teacher encoders;
- lightweight student encoders.

### 5.2 Fusion modules

Fusion nodes combine multiple latent representations into a shared representation.

Examples:

- concat fusion;
- additive fusion;
- gated fusion.

### 5.3 Heads

Heads project latent representations to task-facing outputs such as logits.

Examples:

- linear heads;
- MLP heads.

## 6. Input Rules

- every dependency must be declared in `inputs`;
- undeclared state access is forbidden;
- graph nodes may read batch keys and graph outputs only;
- graph nodes may not read role outputs, interaction outputs, or gate outputs.

## 7. Output Rules

- outputs must use `<node>.<attr>` naming;
- output keys must be stable and predictable;
- output meaning must be local to the graph stage.

Recommended attribute names include:

- `embedding`
- `hidden`
- `logits`
- `weights`

## 8. Missing-Modality Handling

If a node supports missing or optional modality input:

- that behavior must be explicit;
- fallback behavior must be deterministic;
- tests must cover the missing-input path.

## 9. Plugin Registry

Nodes are registered through a registry keyed by config type.

Example:

```python
NodeRegistry.register("encoder.baseline", BaselineEncoder)
NodeRegistry.register("encoder.teacher", TeacherEncoder)
NodeRegistry.register("fusion.concat", ConcatFusion)
NodeRegistry.register("head.linear", LinearHead)
```

## 10. Factory Expectations

Node construction should:

- resolve the registry key;
- inject config params;
- attach node name and declared inputs;
- avoid implicit dependency inference.

## 11. Extension Guidance

When adding a new node plugin:

1. register a unique type key;
2. document expected inputs and outputs;
3. ensure outputs stay in the graph namespace;
4. add unit tests;
5. update [traceability.md](traceability.md) if the plugin class introduces a new requirement.

## 12. Non-Ownership Clarification

Graph nodes do not own:

- KD semantics;
- uncertainty semantics;
- branch gating;
- loss composition.

Those concerns belong to later stages by design.
