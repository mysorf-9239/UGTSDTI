# Kế Hoạch Triển Khai: UGTSDTI Integrity Hardening

## Tổng Quan

Triển khai theo thứ tự:

1. khóa semantic stability của config normalization
2. siết state/read và interaction contracts
3. khóa restore integrity + reproducibility hashing
4. siết scenario coverage và validator/runtime semantic alignment
5. hoàn thiện role/artifact integrity và traceability

Có 4 checkpoint:

1. Normalization và state surfaces không còn silent semantic drift
2. Runtime contracts fail-closed ở interaction + checkpoint restore
3. Reproducibility / scenario coverage semantics sạch
4. Public surface đồng nhất và audit-ready

## Tasks

- [x] 1. Hardening config normalization semantics
  - [x] 1.1 Audit và phân loại các fields order-sensitive vs order-insensitive
    - Graph inputs, interaction order, role outputs, scenario eval, modalities available
    - _Requirements: REQ-INTG-001_
  - [x] 1.2 Sửa `ConfigNormalizer` để preserve order cho fields mang semantic runtime
    - Không sort `graph.nodes[*].inputs`
    - Không canonicalize bừa các list có thể đổi behavior
    - _Requirements: REQ-INTG-001_
  - [x] 1.3 Tách canonicalization-for-hash khỏi normalization runtime nếu cần
    - _Requirements: REQ-INTG-001, REQ-INTG-006_
  - [x] 1.4 Viết regression tests cho node order-sensitive behavior
    - Normalize không được đổi output semantics của runtime order-sensitive
    - _Requirements: REQ-INTG-001_
  - [x]* 1.5 Viết property tests cho semantic-preserving normalization
    - **Property 1: normalize không đổi input order của graph nodes**
    - **Property 2: path/profile-only changes không đổi semantic runtime config**
    - _Requirements: REQ-INTG-001, REQ-INTG-006_

- [x] 2. Hardening state read immutability
  - [x] 2.1 Thiết kế read policy cho `State`
    - Raw vs isolated reads
    - Boundary-sensitive namespaces
    - _Requirements: REQ-INTG-002_
  - [x] 2.2 Triển khai API/read path an toàn cho boundary-sensitive values
    - `get_isolated()` hoặc equivalent executor-side safe path
    - _Requirements: REQ-INTG-002_
  - [x] 2.3 Audit pipeline stages đang đọc trực tiếp state values
    - Decision, metrics, loss, artifacts, evaluator, trainer
    - _Requirements: REQ-INTG-002_
  - [x] 2.4 Viết tests cho mutation-after-read leaks
    - Tensor
    - nested dict/list
    - decision/postprocess boundary values
    - _Requirements: REQ-INTG-002_

- [x] 3. Checkpoint 1 — No more silent semantic drift
  - Verify normalization không đổi runtime meaning
  - Verify state boundary reads không còn leak mutation dễ dàng
  - _Requirements: REQ-INTG-001, REQ-INTG-002_

- [x] 4. Siết graph runtime lifecycle semantics
  - [x] 4.1 Chốt runtime lifecycle contract trong code/docs
    - persistent runtime theo executor hay stateless-only
    - _Requirements: REQ-INTG-003_
  - [x] 4.2 Nếu giữ persistent runtime, đồng bộ docs/tests cho determinism assumptions
    - _Requirements: REQ-INTG-003_
  - [x] 4.3 Viết regression tests cho runtime reuse semantics
    - Same executor
    - fresh executor
    - checkpoint-loaded executor
    - _Requirements: REQ-INTG-003_

- [x] 5. Siết interaction runtime input contracts
  - [x] 5.1 Sửa `InteractionEngine` để materialize declared inputs fail-closed
    - Missing input -> explicit error
    - _Requirements: REQ-INTG-004_
  - [x] 5.2 Đồng bộ interaction runtime behavior với planner contract
    - _Requirements: REQ-INTG-004, REQ-INTG-008_
  - [x] 5.3 Viết unit/integration tests cho missing declared input
    - _Requirements: REQ-INTG-004_

- [x] 6. Hardening checkpoint restore integrity
  - [x] 6.1 Mở rộng `PipelineExecutor.load_model_state()` thành fail-closed restore path
    - Unknown node -> fail
    - Known node nhưng runtime không restore được -> fail
    - _Requirements: REQ-INTG-005_
  - [x] 6.2 Thiết kế restore report hoặc explicit restore summary
    - restored / skipped / failed nodes
    - _Requirements: REQ-INTG-005_
  - [x] 6.3 Đồng bộ eval/resume path trong CLI với restore integrity
    - _Requirements: REQ-INTG-005_
  - [x] 6.4 Viết tests cho partial restore, unsupported restore, unknown node state
    - _Requirements: REQ-INTG-005_

- [ ] 7. Checkpoint 2 — Runtime contracts fail-closed
  - Verify interaction runtime không còn drop missing inputs
  - Verify checkpoint restore không còn success giả
  - Verify runtime lifecycle semantics được test/document rõ
  - _Requirements: REQ-INTG-003, REQ-INTG-004, REQ-INTG-005_

- [ ] 8. Dọn reproducibility hashing
  - [ ] 8.1 Xác định research-semantic vs execution-local runtime fields
    - _Requirements: REQ-INTG-006_
  - [ ] 8.2 Sửa canonical hash logic để bỏ operational-only fields
    - checkpoint path
    - artifact/log dirs
    - local data dir
    - equivalent profile-local paths
    - _Requirements: REQ-INTG-006_
  - [ ] 8.3 Viết tests cho path-insensitive config hash
    - _Requirements: REQ-INTG-006_

- [ ] 9. Siết scenario coverage semantics
  - [ ] 9.1 Mở rộng loader/eval path để trả materialized vs missing scenarios
    - _Requirements: REQ-INTG-007_
  - [ ] 9.2 Mặc định fail-closed khi requested scenario không có samples
    - _Requirements: REQ-INTG-007_
  - [ ] 9.3 Nếu hỗ trợ permissive mode, output phải explicit missing scenarios
    - _Requirements: REQ-INTG-007_
  - [ ] 9.4 Viết integration tests cho multi-scenario eval coverage
    - _Requirements: REQ-INTG-007_

- [ ] 10. Đồng bộ validator và runtime semantics
  - [ ] 10.1 Sửa CLI validation path để không bỏ qua validator injection
    - _Requirements: REQ-INTG-008_
  - [ ] 10.2 Đồng bộ decision fallback semantics giữa validator và runtime
    - _Requirements: REQ-INTG-008_
  - [ ] 10.3 Rà lại plugin-aware validation với runtime registries
    - _Requirements: REQ-INTG-008_
  - [ ] 10.4 Viết regression tests cho validator/runtime agreement
    - _Requirements: REQ-INTG-008_

- [ ] 11. Hardening role aggregation semantics
  - [ ] 11.1 Chốt policy cho `aggregation: first` với nhiều outputs
    - reject mặc định hoặc explicit allow flag
    - _Requirements: REQ-INTG-009_
  - [ ] 11.2 Thêm validation path tương ứng
    - _Requirements: REQ-INTG-009_
  - [ ] 11.3 Viết tests cho ambiguous `first` aggregation
    - _Requirements: REQ-INTG-009_

- [ ] 12. Làm rõ artifact cadence semantics
  - [ ] 12.1 Tách canonical artifact bundle khỏi step-level snapshots
    - _Requirements: REQ-INTG-010_
  - [ ] 12.2 Đồng bộ docs/tests cho artifact lifecycle
    - _Requirements: REQ-INTG-010_
  - [ ]* 12.3 Nếu cần, thêm config surface cho artifact cadence
    - _Requirements: REQ-INTG-010_

- [ ] 13. Cập nhật traceability và collateral
  - [ ] 13.1 Sync `.docs/traceability.md` với `REQ-INTG-*`
    - _Requirements: REQ-INTG-011_
  - [ ] 13.2 Sync design/docs nếu public surface đổi
    - _Requirements: REQ-INTG-011_

- [ ] 14. Checkpoint 3 — Reproducibility / coverage semantics sạch
  - Verify config hash không bị path-only drift
  - Verify requested scenarios không còn biến mất ngầm
  - Verify validator/runtime share cùng semantics ở các đường critical
  - _Requirements: REQ-INTG-006, REQ-INTG-007, REQ-INTG-008_

- [ ] 15. Checkpoint 4 — Integrity-hardening baseline complete
  - Verify:
    - normalization không đổi semantics runtime
    - state/read surfaces không leak mutation lớn
    - interaction + restore fail-closed
    - reproducibility hash phản ánh research semantics
    - requested scenarios được cover đầy đủ hoặc fail rõ
    - validator/runtime/config/docs đồng nhất ở các đường critical
  - Đảm bảo test suites pass và audit không còn blocker P0 của spec này.
  - _Requirements: REQ-INTG-001 đến REQ-INTG-011_
