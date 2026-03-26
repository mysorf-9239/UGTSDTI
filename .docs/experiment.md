# Experiment Guideline

## 1. Scope

This document defines the research protocol for UGTSDTI experiments.

It covers:

- how to study teacher-student asymmetry;
- how to evaluate modality-aware student variants;
- how to evaluate KD effectiveness;
- how to evaluate uncertainty reliability;
- how to evaluate gate trust behavior;
- how to conduct failure analysis and tuning.

## 2. Research Questions

The framework should support experiments that answer at least the following questions:

1. when does the teacher help relative to the student;
2. when does KD improve the student;
3. when is uncertainty predictive of reliability;
4. when does the gate make better trust decisions than static rules;
5. how do these behaviors vary across modality settings and scenarios.

## 3. Original Flow Preservation

The experimental protocol must preserve the two original flows that motivated UGTSDTI.

### 3.1 Flow 1: reduced-modality student

```text
Teacher = sequence + structure
Student = sequence only
```

### 3.2 Flow 2: modality-specific student

```text
Teacher = sequence + structure
Student = structure only
```

### 3.3 Experimental implication

The student is a design space rather than a single fixed object. Experiments should therefore make explicit which student family is being studied:

- modality-reduced student;
- modality-specific student;
- architecture-reduced student.

## 4. Scenario Regime

Canonical scenario family:

| Scenario | Drug seen | Target seen | Meaning |
| --- | --- | --- | --- |
| `S1` | yes | yes | warm-start |
| `S2` | no | yes | cold drug |
| `S3` | yes | no | cold target |
| `S4` | no | no | fully cold |

`S4` remains the strongest indicator of genuine inductive behavior.

## 5. Teacher-Student Asymmetry In Evaluation

Teacher and student must be evaluated as asymmetric branches.

### 5.1 Teacher evaluation lens

Teacher analysis should consider:

- performance under richer modality;
- behavior when pretrained or frozen;
- reliability under warm and cold regimes.

### 5.2 Student evaluation lens

Student analysis should consider:

- deployable performance;
- inductive generalization;
- benefit received from teacher interaction;
- sensitivity to modality reduction.

## 6. Mathematical Formalization

### 6.1 Gate

For the canonical two-branch soft gate:

```math
y = \alpha \cdot y_{teacher} + (1 - \alpha) \cdot y_{student}
```

### 6.2 KD-augmented loss

The canonical objective is:

```math
L = (1 - \lambda) L_{hard} + \lambda L_{KD}
```

where `lambda` controls KD contribution.

### 6.3 Uncertainty

Uncertainty may be represented by predictive variance, entropy, or another documented reliability statistic depending on the chosen interaction plugin.

## 7. Modality-Aware Evaluation

Experiments should explicitly track modality configuration because modality is part of the research question, not only an implementation detail.

Questions to answer include:

- does the teacher benefit mainly from modality completeness or from model capacity;
- does the student degrade gracefully under modality reduction;
- does cross-modality KD help when teacher and student do not share the same modality coverage.

## 8. KD Evaluation Protocol

KD should be evaluated both as interaction and as training signal.

Questions to answer:

- does KD improve student performance;
- does KD help uniformly or only in certain scenarios;
- do logits-only, feature-KD, and relation-KD variants behave differently;
- does intra-modality KD differ materially from cross-modality KD;
- does temperature tuning materially change results.

Recommended reporting:

- student with no KD;
- student with logits KD;
- student with feature or relation KD when available;
- KD-related component curves over training.

## 9. Uncertainty Evaluation Protocol

Uncertainty should not be evaluated only by presence or absence.

Questions to answer:

- does uncertainty correlate with error;
- is teacher uncertainty more informative than student uncertainty;
- does uncertainty improve KD behavior or gate behavior;
- how does uncertainty quality vary across `S1` to `S4`.

Recommended outputs:

- uncertainty versus error correlation;
- per-scenario uncertainty distributions;
- calibration-oriented plots when feasible.

## 10. Gate Evaluation Protocol

The gate must be interpreted as a trust mechanism.

Questions to answer:

- does the gate trust teacher more in warm regimes and student more in colder regimes;
- does hard gating differ materially from soft gating;
- do learned gates outperform heuristic gates;
- how often do fallback paths trigger;
- does modality completeness affect gate preference.

Recommended diagnostics:

- `gate.alpha` distribution;
- per-scenario gate statistics;
- teacher-student disagreement conditioned on gate choice.

## 11. Training Dynamics Evaluation

Training dynamics should be evaluated explicitly when changing:

- teacher freeze status;
- KD schedule;
- gate trainability.

Recommended comparisons:

- frozen teacher versus trainable teacher;
- KD warmup versus constant schedule;
- heuristic gate versus trainable gate.

## 12. Scenario-Aware Evaluation

Scenario-aware behavior should be evaluated separately for:

- KD weighting or activation;
- gate policy;
- loss weighting.

Compare:

- scenario-aware versus scenario-agnostic KD;
- scenario-aware versus scenario-agnostic gates;
- scenario-aware versus scenario-agnostic losses.

## 13. Diagnostics Schema

The framework should expose structured research diagnostics such as:

```yaml
diagnostics:
  disagreement:
  gate_alpha:
  uncertainty_error:
```

These diagnostics are part of the scientific output of the system and should be preserved in experiment artifacts when feasible.

## 14. Core Ablations

Core ablation families should include:

### 14.1 KD ablation

Disable or replace the KD interaction plugin.

### 14.2 Uncertainty ablation

Disable or replace the uncertainty interaction plugin.

### 14.3 Gate ablation

Compare hard versus soft, heuristic versus learned, and uncertainty-aware versus static gates.

### 14.4 Teacher ablation

Remove the teacher branch and retain student-only execution.

### 14.5 Scenario-aware ablation

Compare scenario-aware versus scenario-agnostic KD, gate, and loss policies.

## 15. Failure Analysis Protocol

Failure analysis is a required research activity rather than an optional debugging aid.

### 15.1 KD failure

Investigate:

- no improvement in student performance;
- unstable KD loss;
- collapse under cross-modality KD;
- mismatch between KD signal and scenario difficulty.

### 15.2 Gate failure

Investigate:

- nearly constant `gate.alpha`;
- gate preference that contradicts branch reliability;
- excessive fallback usage;
- over-reliance on the teacher in cold regimes.

### 15.3 Uncertainty miscalibration

Investigate:

- weak uncertainty-error correlation;
- branch uncertainty ranking inconsistent with observed error;
- scenario-specific calibration failure.

## 16. Experiment DSL Usage

The experiment DSL should support:

- `extends` for reusable baselines;
- `sweep` for tuning and ablation search.

Recommended sweep targets include:

- KD temperature;
- KD weight;
- uncertainty sample count;
- gate strategy;
- teacher freeze policy;
- student modality assignment.

## 17. Reporting Standard

Recommended main result table:

```text
Model | Student Type | Modalities | S1 | S2 | S3 | S4 | Avg
```

Recommended diagnostic table:

```text
Variant | KD | Uncertainty | Gate | Teacher | Student | Alpha | S4 | Avg
```

## 18. Multi-Role Future Extension

Future experiments may introduce:

- multiple teachers;
- multiple students;
- role aggregation studies.

Such experiments should preserve the same pipeline and make aggregation explicit in both config and reporting.

## 19. Reproducibility Checklist

Record for every experiment:

- config snapshot after inheritance resolution;
- dataset split artifact;
- seed;
- code revision;
- metrics;
- diagnostics;
- optional checkpoints and traces.

## 20. Interpretation Rule

Results should be interpreted stage by stage:

- graph quality explains representation quality;
- KD explains branch-to-branch teaching quality;
- uncertainty explains reliability estimation quality;
- gate explains trust decision quality;
- final loss and metrics explain optimization and task performance.

This staged interpretation is necessary to preserve the scientific meaning of UGTSDTI rather than reducing it to a single black-box score.
