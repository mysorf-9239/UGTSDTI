# Kế Hoạch Triển Khai: UGTSDTI Research Hardening

## Tổng Quan

Triển khai theo thứ tự:

1. khóa protocol đúng cho `S1–S4`
2. siết runtime contracts còn fail-open
3. đồng bộ identity / reproducibility
4. sửa metrics / uncertainty / gate semantics
5. nâng train loop lên mức usable cho research
6. cập nhật traceability + test suites

Có 4 checkpoint:

1. Protocol-correct data artifacts cho `S1–S4`
2. Runtime contracts fail-closed
3. Metric / uncertainty / identity semantics đúng
4. Train/eval loop đủ cho research runs lặp lại được

## Tasks

- [x] 1. Hardening data protocol cho S1–S4
  - [x] 1.1 Thiết kế lại `DataSplitter` để tạo partition-aware artifacts
    - Thay model “mỗi scenario là một file độc lập” bằng model có `train/val/test`
    - Encode semantics rõ cho `S1`, `S2`, `S3`, `S4`
    - _Requirements: REQ-HARD-001_
  - [x] 1.2 Mở rộng `SplitManifest` và protocol report metadata
    - Thêm metadata đủ để audit overlap/leakage
    - Bổ sung `protocol_report.json` hoặc equivalent manifest section
    - _Requirements: REQ-HARD-001, REQ-HARD-002_
  - [x] 1.3 Mở rộng `DataValidator` cho leakage validation
    - Validate pair overlap, drug overlap, target overlap theo scenario
    - Raise explicit protocol errors với offending ids
    - _Requirements: REQ-HARD-002_
  - [x] 1.4 Cập nhật `DataLoaderFactory` và CLI data loading path theo split model mới
    - Dùng chung loader logic cho CLI/runtime
    - Không duplicate data loading semantics
    - _Requirements: REQ-HARD-001, REQ-HARD-010_
  - [x] 1.5 Viết unit tests cho protocol correctness
    - Test `S2` cold-drug
    - Test `S3` cold-target
    - Test `S4` fully-cold
    - Test overlap detection fail-closed
    - _Requirements: REQ-HARD-001, REQ-HARD-002_
  - [x]* 1.6 Viết property tests cho split non-overlap invariants
    - **Property 1: `S2` train/eval không overlap drug ids**
    - **Property 2: `S3` train/eval không overlap target ids**
    - **Property 3: `S4` train/eval không overlap cả drug lẫn target ids**
    - _Requirements: REQ-HARD-002_

- [x] 2. Checkpoint 1 — Protocol-correct data artifacts
  - Verify split artifacts encode đúng semantics `S1–S4`
  - Verify validator reject leakage rõ ràng
  - Verify loader/runtime đọc được artifacts mới
  - _Requirements: REQ-HARD-001, REQ-HARD-002_

- [x] 3. Siết interaction runtime contracts
  - [x] 3.1 Thêm runtime output validation cho interaction stage
    - Reject undeclared keys
    - Reject missing expected keys
    - Commit chỉ khi contract hợp lệ
    - _Requirements: REQ-HARD-003_
  - [x] 3.2 Refactor `_run_interactions()` trong `PipelineExecutor`
    - Tách thành `InteractionEngine` riêng hoặc helper có validation rõ
    - Reuse `interaction_plan.produced_keys`
    - _Requirements: REQ-HARD-003_
  - [x] 3.3 Viết unit tests cho interaction contract enforcement
    - Runtime trả thiếu key
    - Runtime trả thừa key
    - Runtime disabled path đúng contract
    - _Requirements: REQ-HARD-003_

- [x] 4. Hardening state mutability discipline
  - [x] 4.1 Audit các boundary outputs nhạy cảm
    - `teacher.logits`, `student.logits`, `logits`, `gate.alpha`, `loss.*`, `metrics.*`
    - _Requirements: REQ-HARD-004_
  - [x] 4.2 Bổ sung isolation/detach policy ở nơi phù hợp
    - Clone/detach tensors tại boundary nếu cần
    - Document rõ objects nào được coi là immutable contract surface
    - _Requirements: REQ-HARD-004_
  - [x] 4.3 Viết tests phát hiện mutation-after-commit
    - _Requirements: REQ-HARD-004_

- [x] 5. Đồng bộ execution identity và reproducibility
  - [x] 5.1 Tạo đúng một `ExperimentIdentity` cho mỗi CLI execution
    - Truyền identity xuyên suốt từ `run_cli()` vào train/eval handlers
    - _Requirements: REQ-HARD-005_
  - [x] 5.2 Tích hợp `reproducibility_key` vào runtime flow
    - Build từ `(config_hash, dataset_version, preprocessing_version, split_version, seed)`
    - _Requirements: REQ-HARD-006_
  - [x] 5.3 Ghi identity/repro metadata vào checkpoint + artifact + logs
    - _Requirements: REQ-HARD-005, REQ-HARD-006_
  - [x] 5.4 Mở rộng compatibility validation ở `CheckpointIO`
    - _Requirements: REQ-HARD-006_
  - [x] 5.5 Viết integration tests cho identity consistency
    - CLI stdout run_id == checkpoint run_id == artifact run_id
    - _Requirements: REQ-HARD-005, REQ-HARD-006_

- [x] 6. Checkpoint 2 — Runtime contracts fail-closed
  - Verify interaction runtime không còn silent contract violations
  - Verify state boundary quan trọng không bị mutate ngầm qua test
  - Verify run identity và reproducibility metadata nhất quán
  - _Requirements: REQ-HARD-003, REQ-HARD-004, REQ-HARD-005, REQ-HARD-006_

- [x] 7. Sửa metric semantics
  - [x] 7.1 Thay backend metric bằng implementation chuẩn hoặc implementation đối chiếu chuẩn
    - _Requirements: REQ-HARD-007_
  - [x] 7.2 Chuẩn hóa label/logit shape validation trong `MetricsReporter`
    - _Requirements: REQ-HARD-007_
  - [x] 7.3 Viết reference tests cho `f1`, `auroc`, `auprc`
    - So khớp known-good fixtures
    - _Requirements: REQ-HARD-007_
  - [x] 7.4 Viết tests cho scenario-wise metrics correctness
    - _Requirements: REQ-HARD-007_

- [ ] 8. Sửa uncertainty semantics
  - [ ] 8.1 Tách uncertainty modes
    - `uncertainty.sample_variance`
    - `uncertainty.confidence_proxy`
    - _Requirements: REQ-HARD-008_
  - [ ] 8.2 Cập nhật validator + planner + docs cho uncertainty modes mới
    - _Requirements: REQ-HARD-008_
  - [ ] 8.3 Cập nhật `TrustEstimator` để biết uncertainty source type
    - _Requirements: REQ-HARD-008, REQ-HARD-009_
  - [ ] 8.4 Viết tests cho uncertainty mode semantics
    - _Requirements: REQ-HARD-008_

- [ ] 9. Siết gate semantics
  - [ ] 9.1 Tách config surface giữa heuristic gate và learned gate
    - _Requirements: REQ-HARD-009_
  - [ ] 9.2 Nếu chưa có learned gate thực, reject `training.gate.trainable: true`
    - Hoặc implement learned gate tối thiểu
    - _Requirements: REQ-HARD-009_
  - [ ] 9.3 Viết tests cho gate semantic consistency
    - _Requirements: REQ-HARD-009_

- [ ] 10. Checkpoint 3 — Metric / uncertainty / identity semantics đúng
  - Verify metrics khớp fixtures chuẩn
  - Verify uncertainty naming không misleading
  - Verify gate surface không claim support giả
  - _Requirements: REQ-HARD-005, REQ-HARD-006, REQ-HARD-007, REQ-HARD-008, REQ-HARD-009_

- [ ] 11. Nâng train loop lên mức research-usable
  - [ ] 11.1 Refactor CLI `train` thành epoch-aware loop
    - _Requirements: REQ-HARD-010_
  - [ ] 11.2 Thêm checkpoint cadence và summary logging cadence
    - _Requirements: REQ-HARD-010_
  - [ ] 11.3 Thêm optional eval cadence sau epoch
    - _Requirements: REQ-HARD-010_
  - [ ] 11.4 Đồng bộ data loading path giữa CLI và data subsystem
    - _Requirements: REQ-HARD-010_
  - [ ] 11.5 Viết integration tests cho train/eval lineage
    - train -> checkpoint -> eval -> artifact chain
    - _Requirements: REQ-HARD-010_

- [ ] 12. Hardening custom plugin usability
  - [ ] 12.1 Nâng plugin registration flow cho validator-aware runtime
    - Load plugin registrars trước plugin-aware validation khi cần
    - _Requirements: REQ-HARD-011_
  - [ ] 12.2 Viết integration tests cho custom plugin path
    - Missing registrar
    - Invalid registrar
    - Valid custom node/interaction plugin
    - _Requirements: REQ-HARD-011_

- [ ] 13. Cập nhật traceability và spec collateral
  - [ ] 13.1 Cập nhật `.docs/traceability.md` với REQ-HARD-* mới
    - _Requirements: REQ-HARD-012_
  - [ ] 13.2 Cập nhật design/docs references nếu implementation đổi public surface
    - _Requirements: REQ-HARD-012_

- [ ] 14. Checkpoint 4 — Research-ready hardening baseline
  - Verify:
    - S1–S4 protocol đúng semantics
    - interaction contracts fail-closed
    - run identity / repro metadata nhất quán
    - metrics đúng định nghĩa chuẩn
    - train loop đủ cho research iteration
  - Đảm bảo tất cả tests pass, hỏi user nếu có vấn đề.
  - _Requirements: REQ-HARD-001 đến REQ-HARD-012_
