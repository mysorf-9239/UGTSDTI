# Kế Hoạch Triển Khai: UGTSDTI Baseline Model

## Tổng Quan

Triển khai theo thứ tự:

1. khóa baseline spec vào đúng runtime/config surface hiện tại
2. thêm baseline node runtimes và registry wiring
3. ship canonical baseline config
4. chứng minh forward/loss/metrics/training/checkpoint path đều chạy được
5. sync traceability và collateral docs

Có 4 checkpoint:

1. Registry và node runtimes baseline đã tồn tại, build graph được
2. Baseline forward pass materialize đủ keys chuẩn
3. Baseline train được và checkpoint round-trip được
4. Baseline canonical config + tests + docs hoàn tất

## Tasks

- [x] 1. Audit baseline fit với framework hiện tại
  - [x] 1.1 Rà lại graph registry, trainer, validator, metrics, và config surface hiện tại
    - Chốt các chỗ baseline phải reuse thay vì sửa framework
    - _Requirements: REQ-BL-001, REQ-BL-009, REQ-BL-014_
  - [x] 1.2 Chốt canonical control path của baseline
    - `interaction.noop`
    - `decision.identity`
    - `loss.hard`
    - _Requirements: REQ-BL-001, REQ-BL-002, REQ-BL-009_
  - [x] 1.3 Chốt data assumptions cho test fixtures và config canonical
    - `drug_seq`, `protein_seq`, `labels`, `scenario`
    - _Requirements: REQ-BL-009, REQ-BL-012_

- [x] 2. Harden registry duplicate semantics cho baseline plugins
  - [x] 2.1 Sửa `NodeRegistry.register()` để duplicate `type_key` fail-fast
    - _Requirements: REQ-BL-008_
  - [x] 2.2 Viết regression test cho duplicate registration
    - _Requirements: REQ-BL-008_

- [x] 3. Implement `encoder.simple_drug`
  - [x] 3.1 Thêm runtime với embedding -> mean pooling -> projection
    - _Requirements: REQ-BL-004_
  - [x] 3.2 Expose `parameters()`, `state_dict()`, `load_state_dict()`
    - _Requirements: REQ-BL-011, REQ-BL-013_
  - [x] 3.3 Viết unit tests cho output shape, key name, state round-trip
    - _Requirements: REQ-BL-004, REQ-BL-013_

- [x] 4. Implement `encoder.cnn_protein`
  - [x] 4.1 Thêm runtime với embedding -> Conv1d -> ReLU -> global pooling
    - _Requirements: REQ-BL-005_
  - [x] 4.2 Hỗ trợ config params tối thiểu: `vocab_size`, `embedding_dim`, `hidden_dim`, `kernel_size`, `pooling`
    - _Requirements: REQ-BL-005_
  - [x] 4.3 Viết unit tests cho output shape, pooling mode, state round-trip
    - _Requirements: REQ-BL-005, REQ-BL-013_

- [x] 5. Implement `fusion.concat`
  - [x] 5.1 Thêm runtime concat hai embeddings theo feature axis
    - _Requirements: REQ-BL-006_
  - [x] 5.2 Hỗ trợ optional projection sau concat
    - _Requirements: REQ-BL-006_
  - [x] 5.3 Viết unit tests cho batch mismatch và projected output shape
    - _Requirements: REQ-BL-006_

- [x] 6. Implement `head.mlp`
  - [x] 6.1 Thêm runtime `Linear -> ReLU -> Linear`
    - _Requirements: REQ-BL-007_
  - [x] 6.2 Bảo đảm output là raw logits, không sigmoid
    - _Requirements: REQ-BL-007_
  - [x] 6.3 Viết unit tests cho output shape và checkpoint/state round-trip
    - _Requirements: REQ-BL-007, REQ-BL-013_

- [x] 7. Register baseline node types vào default graph registry
  - [x] 7.1 Thêm 4 type keys mới vào `build_default_graph_registry()`
    - _Requirements: REQ-BL-008_
  - [x] 7.2 Giữ compatibility với legacy type keys đang tồn tại
    - _Requirements: REQ-BL-008, REQ-BL-014_
  - [x] 7.3 Viết tests cho shipped registry surface
    - _Requirements: REQ-BL-008_

- [x] 8. Checkpoint 1 — Baseline nodes build được trong framework
  - Verify:
    - registry biết 4 type keys baseline mới
    - duplicate registration fail-fast
    - baseline graph plan build được và không cycle
  - _Requirements: REQ-BL-003, REQ-BL-008_

- [x] 9. Ship canonical baseline config
  - [x] 9.1 Tạo config baseline reference mới
    - `student` only
    - `interaction.noop`
    - `decision.identity`
    - _Requirements: REQ-BL-002, REQ-BL-009_
  - [x] 9.2 Đảm bảo config pass validator/normalizer/planner
    - _Requirements: REQ-BL-009_
  - [x] 9.3 Nếu cần, thêm profile hoặc fixture config tối thiểu cho tests
    - _Requirements: REQ-BL-009_

- [x] 10. Prove forward + loss + metrics path
  - [x] 10.1 Thêm integration test cho baseline `PipelineExecutor.run_batch()`
    - assert `head.logits`, `student.logits`, `logits`
    - assert `loss.total`
    - assert `metrics.auroc` / `metrics.auprc`
    - _Requirements: REQ-BL-010, REQ-BL-012_
  - [x] 10.2 Kiểm graph trace và state boundary summaries
    - _Requirements: REQ-BL-001, REQ-BL-010_

- [x] 11. Prove training path thực sự learnable
  - [x] 11.1 Tạo learnable synthetic fixture dataset
    - _Requirements: REQ-BL-011_
  - [x] 11.2 Chạy nhiều `Trainer.step()` với optimizer thật
    - _Requirements: REQ-BL-011_
  - [x] 11.3 Assert loss giảm rõ
    - _Requirements: REQ-BL-011_
  - [x] 11.4 Nếu ổn, assert metric tốt hơn random trên fixture
    - ví dụ `AUROC > 0.5`
    - _Requirements: REQ-BL-012_

- [x] 12. Prove model state / checkpoint lineage
  - [x] 12.1 Viết test cho `PipelineExecutor.model_state()` trên baseline graph
    - _Requirements: REQ-BL-013_
  - [x] 12.2 Viết test cho `load_model_state()` round-trip với executor mới
    - _Requirements: REQ-BL-013_
  - [x] 12.3 Nếu phù hợp, thêm checkpoint save/load smoke cho baseline trainer path
    - _Requirements: REQ-BL-013_

- [x] 13. Checkpoint 2 — Baseline forward path complete
  - Verify:
    - canonical config pass validator/planner
    - `logits`, `loss.total`, `metrics.*` materialize đúng
    - role binding / decision identity hoạt động đúng
  - _Requirements: REQ-BL-001, REQ-BL-002, REQ-BL-009, REQ-BL-010_

- [x] 14. Checkpoint 3 — Baseline trainable and restorable
  - Verify:
    - loss giảm trên fixture learnable
    - baseline parameters đi qua optimizer path
    - model_state / checkpoint round-trip restore được
  - _Requirements: REQ-BL-011, REQ-BL-012, REQ-BL-013_

- [x] 15. Sync traceability và collateral docs
  - [x] 15.1 Sync `.docs/traceability.md` với `REQ-BL-*`
    - _Requirements: REQ-BL-015_
  - [x] 15.2 Nếu public config surface mới được ship, sync docs liên quan
    - _Requirements: REQ-BL-015_

- [x] 16. Checkpoint 4 — Baseline reference complete
  - Verify:
    - baseline nodes đã shipped trong registry mặc định
    - canonical baseline config chạy end-to-end
    - training path learnable
    - metrics/logging/reporting đúng surface hiện tại
    - baseline trở thành reference model hợp lệ cho teacher/student work về sau
  - _Requirements: REQ-BL-001 đến REQ-BL-015_

- [x]* 17. Optional CLI smoke for baseline reference
  - [x] 17.1 Thêm fixture-driven CLI train/eval smoke cho baseline config nếu data fixture phù hợp
    - _Requirements: REQ-BL-009, REQ-BL-012_
  - [x] 17.2 Ghi rõ trong docs nếu smoke này chỉ chạy trong CI subset hoặc local profile
    - _Requirements: REQ-BL-015_
