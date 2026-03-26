# Graph Execution

## 1. Scope

This document defines the graph phase only:

- graph construction;
- dependency resolution;
- execution planning;
- deterministic execution;
- graph-phase tracing and state inspection.

Interaction and gate execution are outside the graph phase and are not scheduled as graph nodes.

## 2. Graph Phase Boundary

The graph phase starts from initialized batch state and ends before role binding.

Its job is to compute graph outputs, not semantic branch interactions.

## 3. Build Pipeline

### 3.1 Parse graph config

Read `graph.nodes` from validated config and instantiate node plugins from the registry.

### 3.2 Resolve producers and consumers

For each declared input:

- resolve batch-key dependencies;
- resolve prior node output dependencies;
- reject missing producers.

### 3.3 Validate DAG properties

The graph builder must validate:

- unique node names;
- explicit dependencies;
- single producer per key;
- no cycles.

## 4. Planning

### 4.1 Topological plan

The planner produces a deterministic topological order for execution.

### 4.2 Stable ordering

If several valid orders exist, the planner must use a stable tie-breaker such as config insertion order.

### 4.3 Plan trace

The planner should expose a human-readable trace including:

- node order;
- dependency edges;
- produced keys;
- unresolved dependency failures.

## 5. Runtime Execution

Conceptual graph loop:

```python
for node in plan.order:
    validate_inputs(node, state)
    result = node.forward(state)
    validate_outputs(node, result)
    commit_outputs(state, result["outputs"])
```

## 6. Execution Modes

### 6.1 Training mode

Execute all graph nodes required for downstream training.

### 6.2 Evaluation mode

Execute all graph nodes required for downstream evaluation.

### 6.3 Inference mode

Execute the same graph stage; downstream loss and metric stages may differ, but graph-stage semantics remain stable.

## 7. Determinism

The graph engine must guarantee:

- stable node order;
- no hidden dependency reads;
- no duplicate key writes;
- reproducible traces under the same seed and config.

## 8. Debugging Support

The graph phase should support:

- execution trace logging;
- state-key inspection after each node;
- optional snapshot of intermediate state for debugging.

## 9. Memory And Optimization

The graph phase may support:

- lazy execution planning;
- node-level cache;
- early release of no-longer-needed keys.

These optimizations must preserve graph semantics exactly.

## 10. Relationship To Later Stages

After the graph phase completes:

- role binding interprets selected outputs as branches;
- interaction plugins consume role outputs;
- the gate consumes role and interaction outputs;
- loss and metrics consume final decision outputs.

This phase boundary must remain clean.
