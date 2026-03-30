# Tài Liệu Yêu Cầu: UGTSDTI Integrity Hardening

## 1. Mục đích

Tài liệu này định nghĩa **yêu cầu hệ thống cho pha integrity hardening** tiếp theo của UGTSDTI.

Framework hiện tại đã:

- chạy được end-to-end;
- có protocol `S1–S4` tốt hơn trước;
- có metric semantics chuẩn hơn;
- có CLI, checkpoint, artifacts, train/eval loop cơ bản.

Tuy nhiên audit hệ thống vẫn phát hiện các vấn đề còn lại ở mức **contract integrity**, **runtime honesty**, và **research traceability**.

Spec này nhằm:

- loại bỏ các chỗ còn **silent semantic drift**;
- buộc runtime **fail-closed** ở các đường quan trọng;
- làm claim **deterministic / reproducible / research-ready** trở nên defensible hơn;
- hoàn thiện framework thành một research runtime đáng tin cậy hơn, không chỉ “chạy được”.

---

## 2. Phạm vi

Spec này tập trung vào 8 cụm vấn đề:

1. config normalization không được đổi semantics runtime;
2. state read surface không được làm rò mutation ngầm;
3. graph runtime lifecycle và cache semantics phải rõ ràng;
4. interaction runtime phải enforce input contract đầy đủ;
5. checkpoint restore phải fail-closed nếu restore không hoàn chỉnh;
6. reproducibility hash phải chỉ phản ánh research semantics;
7. scenario coverage phải fail-closed khi eval không đủ requested scenarios;
8. public config/runtime surface phải đồng nhất giữa validator và runtime.

Spec này **không** yêu cầu:

- đổi stage order;
- thêm learned gate thực sự;
- thêm distributed training;
- thêm serving/inference production stack.

---

## 3. Tiền Đề

Framework hiện tại đã có:

- fixed pipeline `Batch -> Graph -> Role Binding -> Interaction -> Decision -> Loss + Metrics`;
- graph planner / engine;
- split artifacts và leakage validator;
- CLI train/eval/validate/sweep;
- checkpoints, artifacts, logging;
- hardening spec trước đó cho protocol / metrics / uncertainty / identity / plugin flow.

Spec này chỉ xử lý những vấn đề còn sót lại sau audit toàn repo.

---

## 4. Mục Tiêu Chất Lượng

Kết thúc spec này, framework phải đạt được:

- **semantic-stable**: normalization không được làm đổi meaning của model graph;
- **mutation-safe**: state surfaces không dễ bị phá invariant bằng in-place mutation;
- **restore-safe**: checkpoint load không được thành công giả;
- **coverage-safe**: eval scenarios requested phải được materialize đầy đủ hoặc fail rõ;
- **repro-semantically-clean**: reproducibility key không bị nhiễu bởi runtime fields thuần vận hành;
- **contract-closed**: planner/runtime/validator dùng cùng semantics cho critical paths.

---

## 5. Yêu Cầu Hệ Thống

### REQ-INTG-001: Normalization Must Preserve Runtime Semantics

Config normalization **MUST NOT** thay đổi semantics runtime của graph hoặc interaction.

Acceptance criteria:

- Normalizer **MUST NOT** reorder bất kỳ field nào mà thứ tự có thể mang semantic runtime.
- Nếu framework cần canonical hashing, canonicalization **MUST** chỉ áp dụng lên fields được chứng minh order-insensitive.
- Node input order **MUST** được giữ nguyên như user khai báo trừ khi plugin spec khai báo rõ là order-insensitive.
- Test suite **MUST** có case phát hiện normalization làm đổi behavior của node order-sensitive.

### REQ-INTG-002: State Read Surfaces Must Respect Immutability Discipline

Framework **MUST** giảm tối đa khả năng mutation sau khi giá trị đã được commit vào state.

Acceptance criteria:

- State public read surface **MUST** có policy rõ cho mutable values.
- Boundary-sensitive values như `teacher.logits`, `student.logits`, `logits`, `gate.alpha`, `loss.*`, `metrics.*` **MUST** không bị mutate ngầm qua public read surface.
- Nếu framework không thể freeze toàn bộ objects, nó **MUST** document rõ phần nào được trả theo read-only / cloned semantics.
- Test suite **MUST** có case bắt mutation-after-read nếu invariant bị phá.

### REQ-INTG-003: Graph Runtime Lifecycle Must Be Explicit

Framework **MUST** làm rõ contract cho runtime caching và runtime statefulness trong graph stage.

Acceptance criteria:

- Runtime cache semantics **MUST** được xác định rõ là stateless-only hoặc stateful-with-lifecycle.
- Nếu runtime được reuse qua nhiều forward passes, design/docs/tests **MUST** phản ánh điều đó.
- Nếu determinism claim được giữ, framework **MUST** chỉ ra conditions mà runtime reuse vẫn hợp lệ.
- Stateful runtime behavior **MUST NOT** trở thành hidden contract không được validate/document.

### REQ-INTG-004: Interaction Runtime Must Enforce Declared Inputs

Interaction runtime **MUST** fail nếu declared input không hiện diện ở runtime.

Acceptance criteria:

- Interaction executor **MUST NOT** silently drop missing declared inputs.
- Missing interaction input **MUST** raise explicit error tại interaction stage.
- Interaction input contract **MUST** được enforce ở runtime tương tự graph stage.
- Unit/integration tests **MUST** cover missing-input fail-closed behavior.

### REQ-INTG-005: Checkpoint Restore Must Be Fail-Closed

Checkpoint load **MUST NOT** thành công nếu runtime state không được restore đầy đủ theo contract.

Acceptance criteria:

- Nếu checkpoint có state cho một graph runtime nhưng runtime đó không hỗ trợ restore, framework **MUST** fail rõ.
- Nếu checkpoint references unknown node names, framework **MUST** fail rõ.
- Restore report **SHOULD** ghi rõ node nào đã restore thành công.
- Evaluation từ checkpoint **MUST NOT** silently tiếp tục với partially restored model state.

### REQ-INTG-006: Reproducibility Hash Must Reflect Research Semantics Only

`config_hash` và `reproducibility_key` **MUST** được build từ fields phản ánh research semantics, không phải runtime logistics thuần vận hành.

Acceptance criteria:

- Canonical config hash **MUST** bỏ qua các runtime fields thuần vận hành như checkpoint path, artifact path, log path, local data dir, và equivalent fields khác không đổi semantics nghiên cứu.
- Nếu một runtime field có ảnh hưởng semantics thật, field đó **MUST** được giữ lại và documented rõ.
- Hai runs tương đương về thí nghiệm nhưng khác local paths **MUST** có cùng `config_hash`.
- Test suite **MUST** có fixtures xác nhận hash stability dưới path/profile-only changes.

### REQ-INTG-007: Scenario Coverage Must Be Explicit and Complete

Evaluation runtime **MUST** fail-closed hoặc report rõ khi requested scenarios không được materialize đầy đủ.

Acceptance criteria:

- Nếu user yêu cầu eval `["s1", "s2", "s3", "s4"]`, framework **MUST** biết scenario nào thực sự có dữ liệu.
- Framework **MUST NOT** silently bỏ mất requested scenario khỏi report.
- Theo default hardening mode, requested scenario không có samples **MUST** raise explicit error.
- Nếu có chế độ permissive, output summary **MUST** ghi rõ missing scenarios thay vì im lặng.

### REQ-INTG-008: Validator and Runtime Must Share One Canonical Semantics

Public config surface **MUST** nhất quán giữa validator, normalizer, planner, và runtime.

Acceptance criteria:

- Validator injection path **MUST** không bị bỏ qua trong CLI/runtime.
- Decision fallback semantics **MUST** giống nhau giữa validator và runtime.
- Plugin-aware validation **MUST** dùng cùng registry semantics như runtime.
- Public config/documentation **MUST** không claim behavior rộng hơn runtime thực có.

### REQ-INTG-009: Role Aggregation Must Not Fail Silently

Role binding **MUST NOT** silently bỏ qua cấu hình có khả năng che lỗi semantic.

Acceptance criteria:

- Aggregation mode `first` với nhiều outputs **SHOULD** bị reject hoặc warning fail-closed theo config policy.
- Role binding **MUST** có validation rõ cho shape/type compatibility của aggregation modes.
- Test suite **MUST** có cases cho multi-output `first` ambiguity.

### REQ-INTG-010: Artifact Emission Must Respect Runtime Cadence Semantics

Artifact bundle emission **SHOULD** phản ánh lifecycle rõ ràng thay vì rewrite mơ hồ ở mọi step.

Acceptance criteria:

- Framework **SHOULD** support artifact cadence tách khỏi scalar logging cadence.
- Final artifact bundle **MUST** được phân biệt rõ với interim snapshots nếu cả hai cùng tồn tại.
- Docs/tests **SHOULD** chỉ rõ artifact nào là canonical run artifact.

### REQ-INTG-011: Traceability Update

Spec integrity hardening **MUST** được trace tới tasks, tests, và collateral docs.

Acceptance criteria:

- Mỗi `REQ-INTG-*` **MUST** có owner trong `tasks.md`.
- `traceability.md` **SHOULD** được cập nhật khi tasks hoàn tất.
- Test additions **MUST** map được tới requirement IDs của spec này.

---

## 6. Định Nghĩa Done Cho Pha Integrity Hardening

Pha integrity hardening được coi là hoàn tất khi:

- normalization không còn đổi semantics runtime;
- state/read surfaces không còn làm lộ mutation bug dễ dàng;
- interaction input contract và checkpoint restore đều fail-closed;
- `config_hash` / `reproducibility_key` phản ánh đúng semantics nghiên cứu;
- eval scenarios requested không còn bị mất ngầm;
- audit lại repo không còn blocker mức P0 liên quan tới determinism, restore integrity, và research traceability.
