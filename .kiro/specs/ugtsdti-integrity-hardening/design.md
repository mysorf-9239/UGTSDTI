# Tài Liệu Thiết Kế: UGTSDTI Integrity Hardening

## 1. Mục đích

Tài liệu này chuyển `requirements.md` của spec integrity hardening thành thiết kế triển khai.

Pha này không nhằm mở rộng capability nghiên cứu lớn, mà nhằm:

- làm runtime behavior nhất quán với những gì framework claim;
- loại bỏ các semantic traps còn sót lại;
- siết contract giữa config, planner, runtime, checkpoint, và reporting;
- nâng integrity của framework lên mức có thể audit và tin cậy hơn.

---

## 2. Nguyên Tắc Thiết Kế

### 2.1 Không thêm abstraction thừa

Chỉ thêm abstraction mới khi nó giải quyết trực tiếp một integrity gap thực tế.

### 2.2 Fail-closed trước, tiện dụng sau

Nếu runtime chưa chắc đúng, nó phải raise lỗi sớm thay vì fallback ngầm.

### 2.3 Semantic stability quan trọng hơn canonical prettiness

Một config “dễ hash” nhưng làm đổi behavior là sai. Hash stability không được đánh đổi bằng runtime semantics.

### 2.4 Validator / planner / runtime phải nói cùng một ngôn ngữ

Critical semantics không được tồn tại hai phiên bản khác nhau ở validator và runtime.

---

## 3. Design Delta So Với Spec Hardening Trước

Spec trước đã sửa:

- protocol `S1–S4`;
- metric semantics;
- uncertainty naming;
- identity propagation;
- plugin-aware validation cơ bản.

Spec mới xử lý 6 delta còn lại:

1. semantic-safe normalization;
2. state/read immutability discipline;
3. runtime lifecycle and restore integrity;
4. scenario coverage hardening;
5. reproducibility hash cleanup;
6. validator/runtime semantic unification.

---

## 4. Semantic-Safe Normalization

### 4.1 Vấn đề

Normalizer hiện đang canonicalize một số list fields theo cách có thể làm đổi runtime behavior.

Điểm nguy hiểm nhất là `graph.nodes[*].inputs`.

### 4.2 Thiết kế mới

Tách rõ hai loại fields:

- **order-sensitive**
- **order-insensitive**

Rules:

- `graph.nodes[*].inputs`: preserve order
- `graph.nodes[*].output_attrs`: preserve declaration order hoặc canonicalize chỉ khi runtime/spec khẳng định order-insensitive
- `interaction.order`: preserve order
- `roles[*].outputs`: preserve order
- `scenario.eval`: có thể sort vì đây là set-like research request surface
- `modalities.available`: có thể sort nếu chỉ dùng như set

### 4.3 Hashing strategy

`hash_config()` không được dựa vào “normalize rồi sort mọi list”.

Thay vào đó:

- keep semantic order cho order-sensitive fields;
- canonicalize chỉ fields set-like;
- dùng một canonicalization pass riêng cho hashing nếu cần, thay vì reuse blanket normalizer assumptions.

---

## 5. State Read Immutability Discipline

### 5.1 Vấn đề

Commit-time isolation đã có, nhưng read-time mutation leak vẫn tồn tại nếu `State.get()` trả internal references.

### 5.2 Thiết kế mới

Có 2 hướng hợp lệ:

1. `State.get()` trả isolated values cho mutable runtime objects;
2. giữ `State.get()` lightweight nhưng thêm explicit `get_readonly()` / `snapshot_deep()` cho critical boundaries và runtime executor bắt buộc dùng các API an toàn ở chỗ nhạy cảm.

Khuyến nghị:

- giữ `State` lightweight cho graph runtime hot path;
- thêm policy theo namespace:
  - graph internals có thể dùng raw values;
  - post-decision / postprocess / artifact / checkpoint boundaries dùng isolated values.

### 5.3 Boundary policy

Các keys sau nên được commit/read theo isolated semantics:

- `teacher.logits`
- `student.logits`
- `logits`
- `gate.alpha`
- `loss.*`
- `metrics.*`
- `diagnostics.*`

---

## 6. Graph Runtime Lifecycle Contract

### 6.1 Vấn đề

Graph engine đang reuse runtimes qua nhiều runs nếu definition giống nhau. Điều này chỉ hợp lệ nếu runtime statefulness được coi là explicit contract.

### 6.2 Thiết kế mới

Chọn một trong hai model:

- **Model A: stateless runtime cache**
  - runtime cache chỉ hợp lệ cho runtimes được khai báo stateless
- **Model B: persistent runtime instances**
  - runtime statefulness là feature; train/eval semantics và determinism docs phải nói rõ

Vì framework hiện đã có trainable graph heads, hướng thực dụng hơn là **Model B**.

### 6.3 Required consequences

Nếu dùng Model B:

- docs phải nói rõ runtime instances persist theo executor lifecycle;
- checkpoint/model-state load phải coi graph runtime state là first-class;
- determinism claim phải ghi rõ “for a fresh executor lifecycle or controlled runtime state”.

---

## 7. Interaction Input Contract Enforcement

### 7.1 Vấn đề

Planner đã validate inputs, nhưng runtime executor còn bỏ qua declared input bị thiếu.

### 7.2 Thiết kế mới

`InteractionEngine` phải materialize inputs theo kiểu graph engine:

```text
for key in declared_inputs:
  if missing -> raise
  else materialize
```

Không dùng `if state.has(key)` để skip ngầm.

### 7.3 Contract symmetry

Interaction stage nên có symmetry với graph stage:

- declared input missing -> fail
- undeclared output -> fail
- missing expected output -> fail
- state commit qua `StateWriter`

---

## 8. Checkpoint Restore Integrity

### 8.1 Vấn đề

`load_model_state()` hiện skip silently nếu runtime không có `load_state_dict`.

### 8.2 Thiết kế mới

Restore report phải explicit:

- `restored_nodes`
- `skipped_nodes`
- `missing_nodes`

Default integrity mode:

- nếu checkpoint có state cho node mà runtime không restore được -> raise
- nếu checkpoint thiếu state cho node được đánh dấu trainable/persistent -> raise hoặc warning fail-closed theo policy

### 8.3 Runtime contract

Graph runtime muốn tham gia training/repro lineage cần expose:

- `parameters()`
- `state_dict()`
- `load_state_dict()`

Nếu chỉ inference/stateless helper runtime, có thể không cần.

---

## 9. Reproducibility Hash Cleanup

### 9.1 Vấn đề

`config_hash` hiện còn dính operational runtime fields.

### 9.2 Thiết kế mới

Tách canonical config cho reproducibility thành:

- **research-semantic config**
- **execution-local config**

Research-semantic config giữ:

- graph / roles / interaction / decision / training semantics
- data dataset/preprocessing/split identifiers
- scenario selection
- seed

Execution-local config loại bỏ:

- checkpoint path
- artifact/log dirs
- local data root
- profile-local paths
- equivalent fields chỉ ảnh hưởng nơi lưu/chạy, không ảnh hưởng experiment meaning

### 9.3 Acceptance logic

Hai configs chỉ khác local paths/profile storage mà giống thí nghiệm thì phải cùng `config_hash`.

---

## 10. Scenario Coverage Integrity

### 10.1 Vấn đề

Runtime có thể eval requested scenarios nhưng silently thiếu một scenario nếu batch list rỗng cho scenario đó.

### 10.2 Thiết kế mới

`DataLoaderFactory` hoặc CLI eval path phải trả thêm:

- `materialized_scenarios`
- `missing_scenarios`

Default hardening mode:

- `missing_scenarios != []` -> raise explicit error

Optional permissive mode:

- cho phép tiếp tục
- summary payload phải chứa `missing_scenarios`
- metrics report phải explicit không phải omission ngầm

---

## 11. Validator and Runtime Semantic Unification

### 11.1 Vấn đề

Hiện vẫn còn vài chỗ validator và runtime không cùng semantics.

### 11.2 Thiết kế mới

Các bề mặt cần đồng bộ:

- decision fallback semantics
- plugin registrar injection path
- canonical config hashing rules
- interaction input/runtime rules
- role aggregation ambiguity policy

### 11.3 One-surface rule

Nếu runtime support một default behavior quan trọng, validator phải biết.
Nếu validator cấm một behavior, runtime không nên silently support nó như fallback ngầm.

---

## 12. Role Aggregation Hardening

### 12.1 Vấn đề

`aggregation: first` với nhiều outputs hiện che cấu hình sai.

### 12.2 Thiết kế mới

Mặc định hardening:

- nếu `first` và số outputs > 1 -> raise

Nếu cần behavior này cho ablation/debug:

- phải có flag explicit như `allow_ambiguous_first: true`

---

## 13. Artifact Cadence Semantics

### 13.1 Vấn đề

Artifact bundle đang bị rewrite quá thường xuyên trong training path.

### 13.2 Thiết kế mới

Tách 3 loại outputs:

- scalar logs
- checkpoint snapshots
- canonical artifact bundle

Suggested policy:

- scalar logs: theo step cadence
- checkpoint: theo epoch/cadence
- canonical artifact bundle: final eval hoặc final checkpoint
- interim bundle nếu có phải đặt namespace riêng như `snapshots/`

---

## 14. Test Strategy

Spec này cần thêm tests ở 4 lớp:

- unit: normalization, state, interaction, restore
- property: semantic-preserving normalization, path-insensitive hashing
- integration: CLI eval missing-scenario fail-closed, checkpoint restore integrity
- regression: order-sensitive node behavior không bị normalizer đổi

---

## 15. Kết Quả Mong Muốn

Sau pha này:

- framework không còn các silent semantic traps lớn;
- audit determinism/reproducibility claims sẽ bớt “conditional” hơn;
- kết quả train/eval/checkpoint/artifact đáng tin hơn cho nghiên cứu;
- spec/runtime/config/docs nói cùng một ngôn ngữ cho các đường critical.
