# Configuration Schema

## 1. Scope

This document defines the configuration API for the UGTSDTI framework.

It covers:

- top-level experiment declaration;
- modality-aware role configuration;
- graph configuration;
- interaction graph configuration;
- gate configuration;
- training dynamics;
- loss mapping;
- diagnostics schema;
- experiment extension and sweep DSL.

Normative rules are defined in [contracts.md](contracts.md).

## 2. Top-Level Structure

The canonical top-level structure is:

```yaml
version: v1
contract_version: v1

extends: [...]
sweep: {...}

experiment: {...}
data: {...}
scenario: {...}
modalities: {...}
graph: {...}
roles: {...}
interaction: {...}
decision: {...}
training: {...}
loss: {...}
metrics: {...}
diagnostics: {...}
logging: {...}
runtime: {...}
```

Required sections:

- `version`
- `data`
- `scenario`
- `modalities`
- `graph`
- `roles`
- `interaction`
- `decision`
- `training`
- `loss`

## 3. Validation Workflow

Validation proceeds in four stages:

1. required-section validation;
2. schema validation;
3. cross-section validation;
4. normalization and inheritance resolution.

## 4. Experiment DSL

### 4.1 `extends`

`extends` allows a config to inherit from one or more base experiment templates.

Example:

```yaml
extends:
  - defaults/research_base
  - defaults/davis_s1
```

Use cases:

- common runtime defaults;
- dataset-specific presets;
- branch-specific ablation baselines.

### 4.2 `sweep`

`sweep` defines hyperparameter exploration or ablation search space.

Example:

```yaml
sweep:
  method: grid
  parameters:
    interaction.kd.temperature: [2.0, 4.0, 8.0]
    loss.kd_weight: [0.1, 0.3, 0.5]
    decision.strategy: [soft, hard]
```

Use cases:

- tuning KD temperature;
- tuning KD weight;
- tuning uncertainty sample counts;
- comparing gate strategies;
- comparing teacher freeze policies.

## 5. Scenario Section

### 5.1 Structure

```yaml
scenario:
  train: s1
  eval: [s1, s2, s3, s4]
  policy:
    affect_kd: true
    affect_gate: true
    affect_loss: true
```

### 5.2 Semantics

Scenario metadata may influence KD, gate, and loss behavior. It must not alter graph topology.

## 6. Modalities Section

### 6.1 Purpose

The modality section makes the teacher-student modality asymmetry explicit rather than implicit in graph wiring alone.

### 6.2 Structure

```yaml
modalities:
  available: [sequence, structure]
  teacher:
    uses: [sequence, structure]
  student:
    uses: [sequence]
```

### 6.3 Flow support

This section must support at least:

- reduced-modality student flows;
- modality-specific student flows;
- architecture-reduced student flows under the same modality set.

### 6.4 KD relation

If KD crosses modality boundaries, any required alignment assumptions should be reflected either here or in the KD subsection.

## 7. Graph Section

### 7.1 Structure

```yaml
graph:
  nodes:
    student_encoder:
      type: encoder.baseline
      inputs: [drug_seq, protein_seq]
      params: {...}
    teacher_encoder:
      type: encoder.teacher
      inputs: [drug_graph, protein_seq]
      params: {...}
```

### 7.2 Notes

The graph section contains graph-stage computation only.

It must not contain:

- KD configuration;
- uncertainty configuration;
- gate configuration.

## 8. Roles Section

### 8.1 Structure

```yaml
roles:
  teacher:
    outputs: [teacher_head.logits]
    aggregation: first
  student:
    outputs: [student_head.logits]
    aggregation: first
```

### 8.2 Semantics

Roles are semantic and asymmetric. The teacher and student sections exist because branch identity matters to later interaction and gate stages.

### 8.3 Future extension

The roles section should remain extensible to multiple teachers or students without changing the pipeline.

## 9. Interaction Section

The interaction section defines an ordered interaction graph executed after role binding.

### 9.1 Structure

```yaml
interaction:
  order: [uncertainty, kd]
  dependencies:
    kd: [uncertainty]
  kd:
    type: kd.standard
    temperature: 4.0
    mode: logits
    enabled: true
  uncertainty:
    type: uncertainty.mc_dropout
    samples: 10
    targets:
      teacher: true
      student: true
```

### 9.2 `order`

`order` declares the intended execution order of interaction modules.

### 9.3 `dependencies`

`dependencies` declares explicit interaction dependencies.

Example:

- `kd` may depend on uncertainty outputs if uncertainty-aware KD is enabled;
- `uncertainty` normally depends only on role outputs.

### 9.4 KD subsection

Supported concepts include:

- `type`
- `temperature`
- `mode`: logits, feature, relation
- `enabled`
- optional scenario-aware weighting
- optional modality alignment metadata for cross-modality KD

Example:

```yaml
interaction:
  kd:
    type: kd.standard
    temperature: 4.0
    mode: logits
    feature_weight: 0.0
    relation_weight: 0.0
    cross_modality:
      enabled: false
      alignment: none
```

### 9.5 Uncertainty subsection

Supported concepts include:

- estimator type;
- sample count;
- target branches;
- optional scenario-aware policy.

## 10. Decision Section

### 10.1 Structure

```yaml
decision:
  type: gate.uncertainty
  strategy: soft
  trainable: true
  learned: true
  use_uncertainty: true
  fallback:
    no_teacher: student
    no_student: teacher
```

### 10.2 Semantics

The decision section configures trust estimation between teacher and student.

Supported concepts include:

- hard versus soft gating;
- learned versus heuristic gate;
- scenario-aware behavior;
- fallback behavior.

### 10.3 Future extension

For future N-branch systems, the decision section should support vector-valued trust allocation.

## 11. Training Section

The training section owns stage-specific training dynamics.

### 11.1 Structure

```yaml
training:
  teacher:
    freeze: true
  student:
    freeze: false
  kd:
    schedule: warmup
  gate:
    trainable: true
```

### 11.2 Teacher training dynamics

Teacher configuration should explicitly declare whether the teacher is:

- frozen;
- trainable;
- partially trainable through extended config.

### 11.3 Student training dynamics

The student should remain trainable by default.

### 11.4 KD schedule

Supported schedule concepts include:

- `warmup`
- `constant`
- future staged or curriculum schedules via extension.

### 11.5 Gate trainability

The gate may be trainable or heuristic. This must be explicit.

## 12. Loss Section

### 12.1 Structure

```yaml
loss:
  type: composite
  hard_weight: 0.7
  kd_weight: 0.3
  map:
    kd:
      from: interaction.kd.loss_component
      weight: 0.3
```

### 12.2 Interaction-to-loss mapping

Loss must define how interaction outputs contribute to final loss.

Example:

```text
interaction.kd.loss_component -> loss.total
```

### 12.3 Scenario-aware loss

Loss may optionally support scenario-aware weighting.

## 13. Metrics Section

### 13.1 Structure

```yaml
metrics:
  primary: auroc
  enabled: [auroc, auprc, f1]
  by_scenario: true
```

## 14. Diagnostics Section

### 14.1 Purpose

Diagnostics are research outputs, not only logging preferences.

### 14.2 Structure

```yaml
diagnostics:
  disagreement: true
  gate_alpha: true
  uncertainty_error: true
```

### 14.3 Semantics

Typical diagnostics include:

- teacher-student disagreement;
- gate alpha distributions;
- uncertainty versus error statistics.

## 15. Logging Section

### 15.1 Structure

```yaml
logging:
  backend: wandb
  trace_execution: true
  inspect_state: true
  log_gate_distribution: true
```

## 16. Minimal Canonical Example

```yaml
version: v1
contract_version: v1

extends:
  - defaults/research_base

experiment:
  name: ugtsdti_hybrid

data:
  dataset: davis

scenario:
  train: s1
  eval: [s1, s2, s3, s4]
  policy:
    affect_kd: true
    affect_gate: true
    affect_loss: true

modalities:
  available: [sequence, structure]
  teacher:
    uses: [sequence, structure]
  student:
    uses: [sequence]

graph:
  nodes:
    student_encoder:
      type: encoder.baseline
      inputs: [drug_seq, protein_seq]
    teacher_encoder:
      type: encoder.teacher
      inputs: [drug_graph, protein_seq]
    student_head:
      type: head.linear
      inputs: [student_encoder.embedding]
    teacher_head:
      type: head.linear
      inputs: [teacher_encoder.embedding]

roles:
  teacher:
    outputs: [teacher_head.logits]
  student:
    outputs: [student_head.logits]

interaction:
  order: [uncertainty, kd]
  dependencies:
    kd: [uncertainty]
  kd:
    type: kd.standard
    temperature: 4.0
    mode: logits
    enabled: true
  uncertainty:
    type: uncertainty.mc_dropout
    samples: 10
    targets:
      teacher: true
      student: true

decision:
  type: gate.uncertainty
  strategy: soft
  trainable: true
  use_uncertainty: true
  fallback:
    no_teacher: student
    no_student: teacher

training:
  teacher:
    freeze: true
  student:
    freeze: false
  kd:
    schedule: warmup
  gate:
    trainable: true

loss:
  type: composite
  hard_weight: 0.7
  kd_weight: 0.3
  map:
    kd:
      from: interaction.kd.loss_component
      weight: 0.3

metrics:
  primary: auroc
  enabled: [auroc, auprc]
  by_scenario: true

diagnostics:
  disagreement: true
  gate_alpha: true
  uncertainty_error: true

runtime:
  device: cuda
  seed: 42
  deterministic: true
```

## 17. Normalization Rules

Normalization may:

- expand shorthand into canonical stage-aligned sections;
- resolve `extends` inheritance;
- normalize sweep parameter targets;
- inject documented defaults.

Normalization must not:

- move plugins between stages;
- silently remove stage boundaries;
- infer hidden loss mappings;
- symmetrize teacher and student semantics.
