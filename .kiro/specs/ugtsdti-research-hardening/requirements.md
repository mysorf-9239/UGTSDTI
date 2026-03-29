# Tài Liệu Yêu Cầu: UGTSDTI Research Hardening

## 1. Mục đích

Tài liệu này định nghĩa **yêu cầu hệ thống cho giai đoạn hardening** của UGTSDTI sau khi framework nền đã được dựng xong.

Mục tiêu của giai đoạn này không còn là “dựng pipeline chạy được”, mà là:

- làm framework **đúng nghĩa research-ready**;
- khóa lại các chỗ đang **lệch semantics khoa học**;
- làm rõ các contract runtime còn đang **chỉ đúng một phần**;
- nâng chất lượng kết quả để có thể dùng cho **thí nghiệm nghiêm túc**.

Tài liệu này là spec mới cho pha tiếp theo. Nó **không thay thế** `ugtsdti-framework/requirements.md`, mà **mở rộng** nó theo hướng hardening.

---

## 2. Phạm vi

Spec này tập trung vào 6 cụm vấn đề:

1. experimental protocol cho `S1–S4`;
2. runtime contract enforcement cho interaction / decision / artifacts;
3. reproducibility và run identity consistency;
4. metric correctness và reporting correctness;
5. uncertainty semantics và trust semantics;
6. training/runtime behavior đủ mạnh cho research loop thực tế.

Spec này **không** yêu cầu:

- đổi kiến trúc stage order;
- thay đổi package layout lớn;
- thêm multi-teacher hoặc multi-student;
- production serving hoặc distributed training phức tạp.

---

## 3. Tiền Đề

Framework hiện tại đã có các thành phần nền sau:

- fixed pipeline `Batch -> Graph -> Role Binding -> Interaction -> Decision -> Loss + Metrics`;
- graph planner / engine;
- role binding;
- interaction foundation;
- decision modules cơ bản;
- CLI train/eval/validate/sweep;
- artifacts / checkpoint / logging foundation.

Các vấn đề đã được xử lý ở pha trước hoặc ở nhánh `main` **không phải trọng tâm của spec này**, trừ khi phát hiện regression.

---

## 4. Mục Tiêu Chất Lượng

Kết thúc spec này, framework phải đạt được:

- **protocol-valid**: S1–S4 phản ánh đúng cold-start semantics;
- **contract-safe**: runtime không chấp nhận silent contract violations;
- **metric-correct**: metrics khớp định nghĩa chuẩn;
- **traceable**: run id, artifact bundle, checkpoint, logs cùng chỉ về một execution identity;
- **experiment-usable**: train/eval loop đủ để chạy thí nghiệm lặp lại được.

---

## 5. Yêu Cầu Hệ Thống

### REQ-HARD-001: Scenario Protocol Correctness

Framework **MUST** triển khai `S1–S4` theo đúng semantics nghiên cứu cold-start.

Acceptance criteria:

- `S1` **MUST** là warm-start protocol.
- `S2` **MUST** enforce cold-drug separation giữa train và evaluation partitions.
- `S3` **MUST** enforce cold-target separation giữa train và evaluation partitions.
- `S4` **MUST** enforce đồng thời cold-drug và cold-target separation giữa train và evaluation partitions.
- Một sample **MUST NOT** được đồng thời thuộc nhiều evaluation protocols nếu điều đó phá semantics cold-start.
- Split artifacts **MUST** lưu metadata đủ để audit overlap / leakage.

### REQ-HARD-002: Split Leakage Validation

Framework **MUST** validate leakage invariants cho split artifacts.

Acceptance criteria:

- Validator **MUST** phát hiện overlap pair leakage.
- Với `S2`, validator **MUST** phát hiện overlap drug identity giữa train và eval.
- Với `S3`, validator **MUST** phát hiện overlap target identity giữa train và eval.
- Với `S4`, validator **MUST** phát hiện overlap cả drug lẫn target identity giữa train và eval.
- Validation failure **MUST** raise explicit error với scenario và offending ids.

### REQ-HARD-003: Runtime Interaction Contract Enforcement

Interaction runtime **MUST** bị validate đối với static contract đã được planner chấp thuận.

Acceptance criteria:

- Runtime output keys **MUST NOT** vượt ra ngoài declared output keys của interaction plugin spec.
- Runtime **MUST** fail nếu thiếu output quan trọng đã được planning chấp thuận.
- Planner contract và runtime contract **MUST** dùng cùng key namespace.
- Interaction outputs **MUST** giữ single-producer semantics như graph outputs.

### REQ-HARD-004: State Mutability Discipline

Framework **MUST** giảm thiểu khả năng silent mutation sau khi state đã commit.

Acceptance criteria:

- Internal documentation và runtime boundary logic **MUST** chỉ rõ object nào có thể mutable và object nào phải treated as immutable.
- Với các boundary quan trọng như `role logits`, `final logits`, `gate.alpha`, `loss.*`, framework **SHOULD** commit detached/isolated values khi phù hợp.
- Test suite **MUST** có case phát hiện mutation-after-commit nếu semantics bị phá.

### REQ-HARD-005: Identity Consistency

Mỗi CLI execution **MUST** có đúng một execution identity xuyên suốt.

Acceptance criteria:

- `run_id` in ra CLI, logs dir, checkpoint path, artifact bundle **MUST** trùng nhau trong cùng execution.
- Checkpoint metadata và artifact metadata **MUST** dùng cùng `run_id`.
- `config_hash` trong logs/checkpoint/artifact **MUST** giống nhau trong cùng execution.

### REQ-HARD-006: Reproducibility Key Integration

Framework **MUST** dùng reproducibility tuple như first-class runtime metadata.

Acceptance criteria:

- Runtime **MUST** materialize `reproducibility_key` từ `(config_hash, dataset_version, preprocessing_version, split_version, seed)`.
- Artifact bundle và checkpoint metadata **MUST** chứa key này.
- Nếu resume/eval từ artifact không khớp reproducibility tuple kỳ vọng, framework **MUST** raise explicit compatibility error.

### REQ-HARD-007: Metric Correctness

Framework **MUST** tính metrics theo định nghĩa chuẩn và audit được.

Acceptance criteria:

- `f1`, `auroc`, `auprc` **MUST** khớp định nghĩa chuẩn cho binary classification.
- Metric implementation **MUST** có test đối chiếu với implementation tham chiếu hoặc known-good fixtures.
- Scenario-wise metrics **MUST** chỉ dùng đúng subset samples của từng scenario.
- Metric reporting **MUST NOT** silently coerce invalid label shapes mà không validate.

### REQ-HARD-008: Uncertainty Semantic Honesty

Framework **MUST NOT** gắn nhãn một uncertainty method bằng tên mạnh hơn khả năng thật của nó.

Acceptance criteria:

- Nếu method được đặt tên `mc_dropout`, implementation **MUST** dựa trên sampled forward passes hoặc stochastic ensemble semantics.
- Nếu implementation chỉ dùng logit-derived confidence proxy, type key / naming / docs **MUST** phản ánh điều đó.
- Decision layer **MUST** biết uncertainty source đang là sampled uncertainty hay heuristic confidence proxy.

### REQ-HARD-009: Trust / Gate Semantic Clarity

Gate semantics **MUST** nhất quán giữa config, training surface, và runtime behavior.

Acceptance criteria:

- Nếu config cho phép `training.gate.trainable`, runtime **MUST** thật sự có gate parameters trainable; nếu không, validator **MUST** reject hoặc warning fail-closed.
- Decision strategies heuristic và learned **MUST** được phân biệt rõ trong config surface.
- Gate diagnostics **MUST** phản ánh đúng source trust signals được dùng.

### REQ-HARD-010: Training Loop Minimum Research Usability

CLI `train` **MUST** vượt mức smoke-path và đủ cho một thí nghiệm lặp lại được.

Acceptance criteria:

- Train loop **MUST** hỗ trợ nhiều step liên tiếp với parameter persistence.
- Runtime **MUST** support checkpoint save ít nhất theo epoch hoặc explicit cadence.
- Logging **MUST** ghi được scalar loss/metrics với step index nhất quán.
- CLI **MUST** có đường rõ ràng để validate / train / eval trên cùng artifact lineage.

### REQ-HARD-011: Custom Plugin Usability

Public CLI/runtime **MUST** hỗ trợ custom plugin registration mà không cần sửa framework code.

Acceptance criteria:

- Custom graph/interactions plugins **MUST** đăng ký được qua config-declared registrars hoặc plugin manifest.
- Config validation **SHOULD** hiểu custom plugin specs đủ để không phải fallback vào heuristic string rules nếu plugin đã được load.
- CLI failures vì missing plugin **MUST** chỉ rõ type key / registrar path gây lỗi.

### REQ-HARD-012: Traceability Update

Spec hardening **MUST** được trace vào design, tasks, implementation owner, và tests.

Acceptance criteria:

- Mỗi REQ-HARD-* **MUST** có task owner trong `tasks.md`.
- `traceability.md` **SHOULD** được cập nhật sau khi task tương ứng hoàn tất.
- Các test mới **MUST** map được tới requirement IDs của spec này.

---

## 6. Định Nghĩa Done Cho Pha Hardening

Pha hardening được coi là hoàn tất khi:

- protocol S1–S4 không còn leakage semantics;
- interaction/runtime contracts fail-closed;
- metrics và uncertainty semantics không còn misleading;
- run identity và reproducibility metadata nhất quán end-to-end;
- CLI train/eval đủ để chạy thí nghiệm lặp lại được;
- audit toàn repo không còn blocker mức P0 đối với research readiness.
