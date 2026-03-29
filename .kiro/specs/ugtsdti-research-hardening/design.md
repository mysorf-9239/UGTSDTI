# Tài Liệu Thiết Kế: UGTSDTI Research Hardening

## 1. Mục đích

Tài liệu này chuyển hóa `requirements.md` của spec hardening thành thiết kế triển khai cho pha hoàn thiện tiếp theo.

Trọng tâm:

- sửa đúng **semantics nghiên cứu**, không chỉ sửa bug cục bộ;
- giữ nguyên fixed pipeline hiện tại;
- harden runtime contracts để framework fail-closed;
- nâng execution flow từ “framework skeleton” lên “research-capable runtime”.

---

## 2. Nguyên Tắc Thiết Kế

### 2.1 Không phá kiến trúc stage hiện có

Pipeline vẫn là:

```text
Batch -> Graph -> Role Binding -> Interaction -> Decision -> Loss + Metrics
```

Không đưa split logic, metric semantics, uncertainty semantics vào sai stage.

### 2.2 Hardening theo chiều dọc

Mỗi vấn đề phải được xử lý xuyên suốt:

- config surface;
- runtime implementation;
- artifact / logging / trace;
- test and traceability.

### 2.3 Fail-closed

Các semantic mismatch mới phát hiện phải **raise lỗi sớm**, không fallback ngầm.

---

## 3. Design Delta So Với Framework Spec Trước

Spec cũ đã dựng được:

- pipeline foundation;
- graph/interaction/decision runtime cơ bản;
- CLI / artifact / checkpoint foundation.

Spec mới thêm 5 lớp hardening:

1. protocol-aware data splitting;
2. runtime contract enforcement cho interaction;
3. identity / reproducibility consistency;
4. metrics & uncertainty correctness;
5. train-loop & gate semantic tightening.

---

## 4. Scenario Protocol Redesign

### 4.1 Split model mới

`S1–S4` không còn là các bucket độc lập “cùng một tập records được lọc theo threshold”.

Thay vào đó, cần mô hình:

```text
dataset records
-> partition planner
-> train / val / test assignment
-> scenario-specific manifests
```

### 4.2 Canonical semantics

- `S1`: warm split, pair-level partition là đủ.
- `S2`: evaluation drugs không xuất hiện trong train.
- `S3`: evaluation targets không xuất hiện trong train.
- `S4`: evaluation drugs và evaluation targets đều không xuất hiện trong train.

### 4.3 Artifact shape đề xuất

Mỗi split artifact nên có:

- `train.jsonl`
- `val.jsonl`
- `test.jsonl`
- `manifest.json`
- `protocol_report.json`

`protocol_report.json` ghi:

- train/test drug overlap
- train/test target overlap
- train/test pair overlap
- counts theo partition
- scenario semantics

### 4.4 Validator

`DataValidator` cần thêm:

- `validate_protocol(manifest, protocol_report)`
- leakage checks theo từng scenario
- fail message chỉ rõ offending ids

---

## 5. Interaction Runtime Contract Enforcement

### 5.1 Vấn đề hiện tại

Interaction planner đã biết output keys theo `InteractionPluginSpec`, nhưng runtime executor chưa kiểm outputs thực tế.

### 5.2 Thiết kế mới

Thêm `InteractionEngine` hoặc nâng `_run_interactions()` hiện tại để:

- materialize declared inputs;
- run runtime;
- validate returned keys so với `interaction_plan.produced_keys[name]`;
- reject undeclared keys;
- reject missing expected keys;
- commit qua `StateWriter`.

### 5.3 Rule

Nếu interaction module bị disable thì expected output set là rỗng.

Nếu expected output set không rỗng mà runtime trả thiếu, phải fail ngay ở stage `interaction`, không đẩy lỗi sang decision/loss.

---

## 6. Identity & Reproducibility Unification

### 6.1 Execution identity

`run_cli()` phải tạo **một** `ExperimentIdentity` duy nhất rồi truyền xuống mọi handler.

Không được build lại identity trong `_run_train()` hoặc `_run_eval()`.

### 6.2 Reproducibility metadata

Tại runtime preparation, sau khi data artifacts đã được resolve, build:

```text
reproducibility_key =
  hash(config_hash, dataset_version, preprocessing_version, split_version, seed)
```

### 6.3 Propagation targets

Metadata này phải đi vào:

- checkpoint bundle
- artifact bundle
- logs
- CLI summary line hoặc emitted summary payload

### 6.4 Compatibility checks

`CheckpointIO.load()` nên có expected fields cho:

- `config_hash`
- `dataset`
- `dataset_version`
- `split_version`
- `reproducibility_key`

---

## 7. Metrics Redesign

### 7.1 Metric backend

Không nên giữ metric formulas “hand-written approximate” cho những metric đã có định nghĩa chuẩn.

Thiết kế đề xuất:

- giữ wrapper `MetricsReporter`
- nhưng backend metric dùng:
  - `sklearn.metrics` cho CPU/offline-eval path, hoặc
  - một implementation internal được test so khớp với sklearn fixtures

### 7.2 Contracts

`MetricsReporter` phải:

- chuẩn hóa logits -> probability ở một chỗ duy nhất;
- validate shape labels/logits;
- compute overall metrics;
- compute per-scenario metrics trên subset rõ ràng;
- emit diagnostics tách khỏi core metrics.

### 7.3 Required tests

Phải có fixture tests so khớp với:

- `f1_score`
- `roc_auc_score`
- `average_precision_score`

cho các case:

- balanced labels
- all-positive / all-negative edge case
- tiny batch
- scenario subset

---

## 8. Uncertainty Semantics

### 8.1 Tách 2 loại uncertainty

Spec mới nên tách rõ 2 mode:

- `uncertainty.sample_variance`
  - yêu cầu nhiều stochastic samples
- `uncertainty.confidence_proxy`
  - dùng deterministic logits-derived proxy

### 8.2 Runtime behavior

Nếu config chọn `sample_variance`:

- runtime phải yêu cầu input có sample dimension hoặc runtime sampling contract riêng.

Nếu config chọn `confidence_proxy`:

- docs, output semantics, diagnostics phải ghi rõ đây là proxy, không phải epistemic uncertainty thật.

### 8.3 Decision integration

`TrustEstimator` phải biết uncertainty source type để diagnostics không misleading.

---

## 9. Decision / Gate Semantics

### 9.1 Tách heuristic gate và learned gate

Hiện tại implementation chỉ có heuristic trust-based decisions.

Thiết kế mới nên phân biệt:

- `decision.strategy: identity | soft | hard`
- `decision.mode: heuristic | learned`

Hoặc tương đương:

- `decision.type: identity | gate.heuristic | gate.learned`

### 9.2 Validation rule

Nếu `training.gate.trainable: true` nhưng decision không expose trainable gate runtime:

- validator phải reject;
- hoặc runtime phải coerce fail-closed.

Public config surface hiện được chốt là:

```yaml
decision:
  type: gate.uncertainty
  mode: heuristic
  strategy: soft
  use_uncertainty: true
training:
  gate:
    trainable: false
```

`decision.mode: learned` và `training.gate.trainable: true` hiện phải fail-closed cho tới khi có learned gate runtime thật.

### 9.3 Minimal learned gate design

Nếu làm learned gate ở pha này, shape tối thiểu:

- input: `teacher.logits`, `student.logits`, optional `teacher.var`, `student.var`, `interaction.disagreement`
- output: `gate.alpha`
- parameters: small MLP / linear head

Nếu chưa làm learned gate ở pha này:

- phải thu hẹp config surface để không claim support giả.

---

## 10. Training Loop Hardening

### 10.1 Mục tiêu

`train` command phải là một train loop tối thiểu có ích cho research.

### 10.2 Required runtime features

- persistent runtime parameters qua nhiều steps
- explicit optimizer lifecycle
- checkpoint cadence
- logging cadence
- optional validation pass per epoch

### 10.3 Suggested shape

```text
prepare runtime
-> build loaders
-> for epoch:
     for batch:
       trainer.step(...)
     optional evaluator.evaluate(...)
     checkpoint save
     summary log
```

Public surface tối thiểu được chốt là:

```yaml
training:
  optimizer:
    lr: 0.01
  loop:
    epochs: 1
    checkpoint_every_epochs: 1
    summary_every_steps: 1
    eval_every_epochs: 0
    eval_partition: val
runtime:
  checkpoint_path: null
```

- `training.loop.*` điều khiển epoch loop, checkpoint cadence, summary cadence, và optional eval cadence.
- `runtime.checkpoint_path` là operational input cho eval / lineage path, không được làm thay đổi semantic `config_hash`.

### 10.4 Scope limitation

Không bắt buộc distributed training, AMP phức tạp, gradient accumulation, early stopping nâng cao trong pha này.

---

## 11. Plugin Registration Design

### 11.1 Current baseline

Đã có `runtime.plugin_registrars`.

### 11.2 Hardening goal

Config validation nên có khả năng validate custom plugins sau khi registrars được load.

### 11.3 Suggested flow

```text
load raw config
-> instantiate default registries
-> apply plugin registrars (if any)
-> run plugin-aware validation against loaded registries
-> build plans
```

Điều này tốt hơn validate trước rồi mới plugin-load.

---

## 12. Test Strategy Cho Pha Hardening

### 12.1 Unit

- split protocol leakage tests
- interaction runtime output contract tests
- run identity consistency tests
- metric reference tests
- uncertainty mode validation tests

### 12.2 Integration

- end-to-end `train` với checkpoint + artifact + logs cùng `run_id`
- eval theo từng scenario từ real split manifest
- custom plugin registration path có validation-aware runtime

### 12.3 Property-based

- split non-overlap properties
- reproducibility tuple stability
- contract fail-closed properties cho interaction outputs

---

## 13. Out Of Scope

Không làm trong spec này:

- multi-node distributed runtime
- online data download trong train path
- UI dashboard / experiment tracker nâng cao
- serving / API inference production

---

## 14. Expected Outcome

Kết thúc spec này, UGTSDTI phải chuyển từ:

```text
architecturally-correct framework skeleton
```

sang:

```text
research-usable framework with auditable experiment semantics
```
