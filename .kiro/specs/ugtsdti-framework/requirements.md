# Tài Liệu Yêu Cầu: UGTSDTI Framework

## 1. Mục đích

Tài liệu này định nghĩa **yêu cầu hệ thống** cho UGTSDTI như một framework nghiên cứu cho bài toán dự đoán tương tác thuốc-đích (DTI).

Mục tiêu của tài liệu là:

- khóa các bất biến kiến trúc của framework;
- xác định các hành vi bắt buộc có thể kiểm chứng được;
- tách requirement khỏi implementation detail không cần thiết;
- cung cấp nền tảng ổn định để cập nhật `design.md`, `traceability.md`, và codebase.

Tài liệu này là tài liệu **chuẩn nền** cho giai đoạn refinement tiếp theo. Nếu có mâu thuẫn giữa requirement cũ và requirement mới trong file này, **file này là nguồn đúng**.

---

## 2. Phạm vi

Tài liệu này áp dụng cho phiên bản `v1` của framework UGTSDTI với các thành phần:

- pipeline thực thi cố định;
- graph stage dựa trên node plugins;
- role binding cho `teacher` và `student`;
- interaction layer cho KD và uncertainty;
- decision layer cho final prediction;
- training, evaluation, reporting, reproducibility, và data lifecycle.

Tài liệu này **không** khẳng định một mô hình cụ thể là tối ưu về mặt khoa học. Nó chỉ định framework cần có để kiểm thử các giả thuyết nghiên cứu một cách sạch, tái tạo được, và có thể audit.

---

## 3. Tài Liệu Tham Chiếu

### 3.1 Local references

- `.docs/README.md`
- `.docs/architecture.md`
- `.docs/contracts.md`
- `.docs/config_schema.md`
- `.docs/node_spec.md`
- `.docs/graph_execution.md`
- `.docs/experiment.md`
- `.docs/implementation_plan.md`

### 3.2 External references used to calibrate this document

Các nguồn dưới đây được dùng như **non-normative guidance** để làm sạch chất lượng requirement, không phải để thay thế requirement của dự án:

- RFC 2119: key words for requirement levels
  https://datatracker.ietf.org/doc/html/rfc2119
- ISO/IEC/IEEE 29148:2018 requirements engineering standard overview
  https://www.iso.org/standard/72089.html
- NASA Appendix C: How to Write a Good Requirement
  https://www.nasa.gov/reference/appendix-c-how-to-write-a-good-requirement/
- NASA Technical Requirements Definition
  https://www.nasa.gov/reference/4-2-technical-requirements-definition/
- NASA Requirements Management / traceability guidance
  https://www.nasa.gov/reference/6-2-requirements-management/
- PyTorch reproducibility notes
  https://docs.pytorch.org/docs/stable/notes/randomness.html
- PyTorch deterministic algorithms reference
  https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
- TensorFlow Datasets versioning guide
  https://www.tensorflow.org/datasets/datasets_versioning
- DVC data/model versioning overview
  https://dvc.org/doc/use-cases/versioning-data-and-models

---

## 4. Quy Ước Ngôn Ngữ

Các từ khóa `MUST`, `MUST NOT`, `SHALL`, `SHALL NOT`, `SHOULD`, `SHOULD NOT`, `MAY` trong tài liệu này được hiểu theo tinh thần của RFC 2119.

Quy tắc áp dụng:

- `MUST` / `SHALL`: yêu cầu bắt buộc.
- `MUST NOT` / `SHALL NOT`: điều bị cấm.
- `SHOULD`: yêu cầu mạnh, chỉ được bỏ qua nếu có lý do kỹ thuật rõ ràng.
- `MAY`: tùy chọn hợp lệ.

---

## 5. Tiêu Chí Chất Lượng Của Requirement

Mỗi requirement trong tài liệu này phải thỏa cả bốn tiêu chí sau:

- **Necessary**: có lý do tồn tại gắn với mục tiêu framework.
- **Unambiguous**: không dùng các từ mơ hồ kiểu "good", "fast", "robust" nếu không có điều kiện kiểm chứng.
- **Verifiable**: có thể kiểm bằng test, inspection, analysis hoặc runtime validation.
- **Traceable**: có thể nối tới thiết kế, implementation module, và test.

Các requirement chỉ mô tả:

- hành vi hệ thống quan sát được;
- interface công khai;
- contract giữa các stage;
- constraint kỹ thuật cần bảo toàn.

Các chi tiết như tên file nội bộ, class layout, hoặc decomposition cụ thể **không phải là requirement**, trừ khi đó là public surface bắt buộc.

---

## 6. Thuật Ngữ

- **Batch**: payload đầu vào cho một forward pass.
- **State**: cấu trúc giao tiếp duy nhất giữa các stage trong một forward pass.
- **Graph stage**: DAG các node tính toán trên batch keys và graph outputs.
- **Role binding**: ánh xạ graph outputs sang các role ngữ nghĩa như `teacher` và `student`.
- **Interaction**: stage tính toán các quan hệ liên-branch như KD, uncertainty, disagreement.
- **Decision**: stage tạo final `logits` từ role outputs và interaction outputs.
- **Teacher**: nhánh đặc quyền, có thể giàu modality hơn, pretrained hơn, hoặc transductive hơn.
- **Student**: nhánh deployable/inductive và là đối tượng chính của KD.
- **Scenario**:
  - `S1`: warm-start
  - `S2`: cold drug
  - `S3`: cold target
  - `S4`: fully cold

---

## 7. System Context

UGTSDTI là một framework nghiên cứu cho các câu hỏi:

- khi nào teacher giúp student;
- khi nào KD cải thiện student;
- khi nào uncertainty là tín hiệu tin cậy;
- khi nào gate đưa ra trust decision tốt hơn rule tĩnh;
- các hành vi này thay đổi thế nào theo modality và scenario.

Pipeline kiến trúc chuẩn là:

```text
Batch -> Graph -> Role Binding -> Interaction -> Decision -> Loss + Metrics
```

Pipeline này là bất biến của framework.

---

## 8. Yêu Cầu Hệ Thống

### REQ-ARCH-001: Fixed Pipeline Invariant

Framework **MUST** thực thi pipeline theo thứ tự cố định:

```text
Batch -> Graph -> Role Binding -> Interaction -> Decision -> Loss + Metrics
```

Acceptance criteria:

- Không thí nghiệm nào được reorder, merge, hoặc bypass stage boundary.
- Nếu một stage không có logic nghiên cứu cụ thể, framework **MUST** dùng no-op hoặc identity implementation thay vì bỏ qua stage.
- Training, evaluation, và inference **MUST** giữ nguyên stage order; chỉ output hậu kỳ mới được phép khác.

### REQ-ARCH-002: Teacher-Student Asymmetry

Framework **MUST** bảo toàn semantics bất đối xứng của `teacher` và `student`.

Acceptance criteria:

- `teacher` và `student` **MUST NOT** bị coi là hai ensemble members đối xứng.
- `teacher` **MAY** dùng modality phong phú hơn, pretrained hơn, frozen hơn, hoặc transductive hơn.
- `student` **SHOULD** là nhánh deployable/inductive mặc định và **SHOULD** trainable trong canonical research runs.
- Framework **MUST** hỗ trợ ít nhất ba họ student:
  - modality-reduced
  - modality-specific
  - architecture-reduced

### REQ-ARCH-003: Stage Contracts

Mỗi stage **MUST** có contract input/output rõ ràng.

Acceptance criteria:

- Graph stage chỉ được đọc batch keys và graph outputs đã khai báo.
- Role binding chỉ được đọc graph outputs.
- Interaction stage chỉ được đọc role outputs và prior interaction outputs.
- Decision stage chỉ được đọc role outputs và interaction outputs.
- Loss/Metrics stage chỉ được đọc final decision outputs, labels, scenario metadata, và các interaction outputs được loss map hoặc diagnostics config tham chiếu tường minh.

### REQ-ARCH-004: Stable Naming

Framework **MUST** dùng naming convention ổn định giữa các stage.

Acceptance criteria:

- Graph outputs dùng dạng `<node>.<attr>`.
- Role outputs dùng dạng `<role>.logits`.
- Interaction outputs dùng tên explicit, ví dụ `teacher.var`, `student.var`, `interaction.kd.loss_component`.
- Decision outputs dùng `logits`, `gate.alpha`, và `gate.*` diagnostics nếu có.
- Loss outputs dùng `loss.*`; metric outputs dùng `metrics.*`; diagnostics outputs dùng `diagnostics.*`.

---

### REQ-STATE-001: Single State Medium

Framework **MUST** dùng một `State` duy nhất làm medium giao tiếp giữa các stage trong một forward pass.

Acceptance criteria:

- State là cấu trúc key-value duy nhất đi qua pipeline.
- Components **MUST NOT** trao đổi dữ liệu qua hidden globals hoặc side-channel.
- Stage executor là nơi duy nhất được phép commit outputs vào State.

### REQ-STATE-002: Write-Once, Single-Producer, Monotonicity

State **MUST** tuân thủ ba invariant:

- write-once
- single producer per key
- monotonic growth

Acceptance criteria:

- Ghi đè một key đã tồn tại trong cùng forward pass **MUST** raise explicit error.
- Mỗi key **MUST** có đúng một producer.
- State **MUST NOT** xóa hoặc mutate key đã commit trong cùng forward pass.

### REQ-STATE-003: Declared Access Discipline

Components **MUST** chỉ đọc keys đã được phép theo contract của stage và declared inputs của chính chúng.

Acceptance criteria:

- Graph nodes đọc ngoài `inputs` của chúng **MUST** bị từ chối.
- Interaction modules đọc future outputs hoặc decision outputs **MUST** bị từ chối.
- Runtime validator **MUST** báo lỗi chỉ rõ stage, component, và key vi phạm.

### REQ-STATE-004: State Schema Validation

Framework **MUST** hỗ trợ validate State tại stage boundaries.

Acceptance criteria:

- Validator **MUST** kiểm tra presence của required keys.
- Với tensor keys, validator **MUST** kiểm tra shape compatibility, device consistency, và dtype consistency khi applicable.
- Role binding **MUST** validate shape compatibility của outputs cần downstream blend hoặc compare.
- Validation failure **MUST** raise explicit schema/contract error.

---

### REQ-CONF-001: Canonical Config Surface

Framework **MUST** hỗ trợ YAML config với top-level sections chuẩn sau:

```yaml
version
data
scenario
modalities
graph
roles
interaction
decision
training
loss
```

Các section sau là optional nhưng được khuyến nghị:

```yaml
experiment
extends
sweep
metrics
diagnostics
logging
runtime
```

Acceptance criteria:

- Thiếu required section **MUST** fail validation.
- Config surface **MUST** đủ để cấu hình architecture, ablation, training, và evaluation mà không sửa pipeline code.

### REQ-CONF-002: Validation Workflow

Config system **MUST** validate theo thứ tự logic sau:

1. required-section validation
2. schema validation
3. cross-section validation
4. normalization and inheritance resolution

Acceptance criteria:

- Validation **MUST** fail fast với lỗi mô tả rõ vi phạm.
- Cross-section validation **MUST** kiểm tra tối thiểu:
  - roles tham chiếu graph outputs hợp lệ
  - interaction dependencies hợp lệ và acyclic
  - decision prerequisites khớp outputs sẵn có
  - loss mappings tham chiếu keys hợp lệ
  - modality compatibility khớp configured nodes

### REQ-CONF-003: Inheritance, Sweep, and Normalization

Framework **MUST** hỗ trợ `extends`, `sweep`, và normalization nhất quán.

Acceptance criteria:

- `extends` **MUST** resolve deterministically.
- `sweep` **MUST** hỗ trợ normalized parameter targets.
- Normalization **MUST** là idempotent.
- Parse -> normalize -> serialize -> parse **SHOULD** giữ semantic equivalence.
- Normalization **MUST NOT** tự ý di chuyển component giữa các stage hoặc tự suy luận hidden loss mappings.

---

### REQ-DATA-001: Data Lifecycle Separation

Framework **MUST** tách data acquisition/materialization khỏi train/eval/inference runtime.

Acceptance criteria:

- Train/eval/infer **MUST NOT** phụ thuộc network call hoặc pyTDC live API.
- Nếu dùng pyTDC, pyTDC **MUST** chỉ được dùng ở acquisition/materialization phase.
- Runtime **MUST** đọc từ materialized local artifacts.

### REQ-DATA-002: Dataset Versioning and Split Manifest

Framework **MUST** version hóa dữ liệu theo các thành phần tách biệt:

- `dataset_version`
- `preprocessing_version`
- `split_version`

Acceptance criteria:

- Mỗi experiment **MUST** lưu đủ ba version components.
- Split artifacts **MUST** đi kèm manifest chỉ ra dataset, preprocessing, seed, và scenario definitions.
- Reuse split cũ khi raw dataset thay đổi **MUST NOT** xảy ra âm thầm.

### REQ-DATA-003: Batch Contract

Framework **MUST** dùng batch contract rõ ràng nhưng **không được** làm cứng modality beyond configured need.

Acceptance criteria:

- Required common batch keys:
  - `labels`
  - `scenario`
- Modality-specific batch keys như `drug_seq`, `protein_seq`, `drug_graph`, `drug_id`, `protein_id` chỉ là **conditionally required** nếu có node hoặc validator downstream cần chúng.
- Batch contract **MUST** hỗ trợ cả reduced-modality và modality-specific student flows.

### REQ-DATA-004: Data Validation Hooks

Framework **MUST** có validation hooks trước training và tại runtime.

Acceptance criteria:

- Trước training, framework **MUST** validate data artifacts và split manifest.
- Trong debug mode, framework **MUST** validate batch contract ở mỗi batch.
- Trong normal mode, framework **SHOULD** validate ít nhất batch đầu tiên của mỗi epoch hoặc loader session.
- NaN/Inf hoặc missing required keys trong batch **MUST** raise explicit batch/data error.

---

### REQ-GRAPH-001: Node Plugins and Static Capabilities

Graph stage **MUST** được xây trên node plugins có declared inputs và static capability metadata.

Acceptance criteria:

- Mỗi node plugin **MUST** khai báo:
  - type key
  - declared inputs
  - output spec đủ để dry-run validation
  - modality/capability metadata đủ cho config validation
- Plugin capability metadata **MUST** đủ để phát hiện modality mismatch trước runtime.
- Dry-run validation **MUST NOT** yêu cầu load model weights hoặc external resources.

### REQ-GRAPH-002: Deterministic DAG Build, Plan, Execute

Graph subsystem **MUST** build và thực thi DAG một cách deterministic.

Acceptance criteria:

- Graph builder **MUST** validate:
  - unique node names
  - explicit dependencies
  - single producer per key
  - no cycles
- Graph planner **MUST** tạo topological order deterministic với stable tie-break.
- Graph engine **MUST** validate inputs trước forward, validate outputs sau forward, rồi commit outputs theo State contract.

### REQ-GRAPH-003: Graph Debuggability

Graph subsystem **MUST** hỗ trợ trace và dry-run inspection.

Acceptance criteria:

- Framework **MUST** có human-readable execution trace.
- Debug mode **MUST** log node order, inputs consumed, outputs produced, và state keys tại stage boundaries.
- Dry-run build **MUST** có thể validate DAG và report unresolved dependencies mà không chạy forward pass thật.

---

### REQ-ROLE-001: Canonical Role Binding

Role binding **MUST** ánh xạ graph outputs sang semantic roles theo config.

Acceptance criteria:

- Mỗi role được cấu hình **MUST** expose đúng một canonical `<role>.logits`.
- Role binding **MUST** fail rõ ràng nếu target graph outputs không tồn tại hoặc không hợp lệ về shape/schema.
- V1 **MUST** hỗ trợ `teacher` và `student`.
- Future multi-role support **MAY** được chuẩn bị, nhưng V1 **MUST NOT** phá semantics hai-role hiện tại.

---

### REQ-INT-001: Ordered, Acyclic Interaction Graph

Interaction stage **MUST** là một ordered acyclic graph, không phải flat unordered block.

Acceptance criteria:

- Interaction config **MUST** hỗ trợ execution order và explicit dependencies.
- Framework **MUST** validate interaction graph acyclic trước runtime.
- Interaction modules **MUST** chỉ đọc role outputs và prior interaction outputs.
- Framework **MUST** maintain producer map hoặc cơ chế tương đương để enforce single producer trong interaction stage.

### REQ-INT-002: Knowledge Distillation as Interaction

KD **MUST** được model như interaction module, không chỉ là một loss term.

Acceptance criteria:

- KD module **MUST** nhận `teacher -> student` là hướng mặc định.
- KD config **MUST** hỗ trợ:
  - `temperature > 0`
  - `mode` trong `{logits, feature, relation}`
  - enable/disable
  - weighting được map tường minh vào loss
- Nếu cross-modality KD được bật, framework **MUST** yêu cầu alignment assumptions được khai báo tường minh.
- KD execution **MUST** emit explicit outputs cần cho loss hoặc analysis, ví dụ KD component loss và targets/alignment diagnostics.

### REQ-INT-003: Uncertainty as Interaction

Uncertainty estimation **MUST** thuộc interaction stage và **MUST NOT** bị đồng nhất với gate hoặc loss.

Acceptance criteria:

- Uncertainty module **MUST** hỗ trợ teacher-only, student-only, hoặc both.
- Outputs **MUST** branch-explicit, ví dụ `teacher.var`, `student.var`.
- Uncertainty outputs **MUST** finite và non-negative.
- Decision stage **MAY** dùng uncertainty nhưng không sở hữu việc tính uncertainty.

---

### REQ-DEC-001: Decision Stage External Contract

Decision stage **MUST** là nơi duy nhất sinh final prediction `logits`.

Acceptance criteria:

- Final `logits` **MUST** chỉ được tạo ở decision stage.
- Decision stage **MAY** emit `gate.alpha` và `gate.*` diagnostics khi applicable.
- Nội bộ decision stage **MAY** được decomposition thành trust estimator + decision policy, nhưng decomposition này là **non-normative** ở level requirement.
- Student-only baseline **MUST** vẫn đi qua decision stage bằng identity/no-op decision module.

### REQ-DEC-002: Trust, Fallback, and Alpha Constraints

Decision logic **MUST** coi gate như trust mechanism giữa branches.

Acceptance criteria:

- Với canonical two-branch soft blending:
  - `y = alpha * y_teacher + (1 - alpha) * y_student`
- Nếu scalar `gate.alpha` được emit, framework **MUST** enforce:
  - finite
  - `0 <= alpha <= 1`
- Decision config **MUST** định nghĩa explicit fallback behavior khi:
  - teacher absent
  - student absent
  - uncertainty unavailable
  - preferred strategy cannot compute

---

### REQ-POST-001: Explicit Loss Composition

Loss composition **MUST** là explicit và traceable.

Acceptance criteria:

- Loss config **MUST** chỉ rõ interaction outputs nào map vào `loss.total`.
- Framework **MUST NOT** giả định hidden interaction behavior.
- Canonical KD-augmented objective:
  - `L = (1 - lambda) * L_hard + lambda * L_KD`
  **MUST** được biểu diễn qua config và implementation một cách tường minh.

### REQ-POST-002: Metrics and Diagnostics

Framework **MUST** coi diagnostics là research outputs, không chỉ là logging noise.

Acceptance criteria:

- Metrics **MUST** hỗ trợ tối thiểu:
  - AUROC
  - AUPRC
  - F1
- Framework **MUST** hỗ trợ diagnostics cho:
  - branch disagreement
  - gate alpha behavior
  - uncertainty versus observed error
- Khi `metrics.by_scenario: true`, reporting **MUST** giữ tách biệt theo scenario.

---

### REQ-TRAIN-001: Training Dynamics

Training system **MUST** kiểm soát tường minh trainability và schedule của các component chính.

Acceptance criteria:

- Teacher freeze/trainable state **MUST** được cấu hình explicit.
- Student **SHOULD** trainable theo mặc định trong canonical runs.
- KD schedule **MUST** hỗ trợ ít nhất:
  - `warmup`
  - `constant`
- Decision/gate trainability **MUST** được cấu hình explicit nếu decision implementation có tham số học được.

### REQ-EVAL-001: Scenario-Aware Evaluation

Evaluator **MUST** hỗ trợ evaluation trên các scenarios được chỉ định trong config.

Acceptance criteria:

- V1 **MUST** hỗ trợ `S1` đến `S4`.
- Reporting **MUST** giữ semantics bất đối xứng của teacher/student.
- Nếu `S4` có trong evaluation set, reporting **SHOULD** làm nổi bật `S4` như chỉ báo mạnh nhất cho inductive behavior.
- Result schema **MUST** chứa per-scenario metrics và aggregated metrics.

### REQ-ABL-001: Ablation and Sweep Support

Framework **MUST** hỗ trợ ablation qua config thay vì sửa pipeline code.

Acceptance criteria:

- Có thể disable hoặc thay KD, uncertainty, decision strategy, teacher branch qua config.
- `extends` **MUST** cho phép tạo ablation configs từ baseline.
- `sweep` **MUST** cho phép explore ít nhất:
  - KD temperature
  - KD weight
  - uncertainty sample count
  - decision strategy
  - teacher freeze policy
  - student modality assignment

---

### REQ-REPRO-001: Determinism and Reproducibility Controls

Framework **MUST** hỗ trợ deterministic execution mode ở mức tốt nhất có thể trên platform cụ thể.

Acceptance criteria:

- Runtime **MUST** hỗ trợ `seed` và `deterministic` config.
- Framework **MUST** set và log relevant randomness controls.
- Nếu deterministic mode bật, framework **MUST** chấp nhận trade-off hiệu năng để tăng reproducibility.
- Documentation và runtime **MUST** nói rõ rằng cross-platform hoặc cross-release reproducibility tuyệt đối có thể không bảo đảm, phù hợp với guidance của PyTorch.

### REQ-REPRO-002: Experiment Identity and Reproducibility Bundle

Mỗi run **MUST** có identity và artifact bundle rõ ràng.

Acceptance criteria:

- Mỗi run **MUST** có:
  - unique run id
  - config hash từ normalized config
  - git commit nếu available
  - timestamp
- Artifact bundle **MUST** lưu tối thiểu:
  - normalized config snapshot
  - metrics/result schema
  - diagnostics
  - split/version metadata
  - checkpoints nếu enabled
- Full reproducibility key **SHOULD** bao gồm:
  - `config_hash`
  - `dataset_version`
  - `preprocessing_version`
  - `split_version`
  - `seed`

### REQ-OPS-001: Logging Abstraction

Framework **MUST** tách logging backend khỏi pipeline core.

Acceptance criteria:

- Pipeline core **MUST NOT** phụ thuộc trực tiếp vào `wandb`.
- Logging **MUST** có interface trừu tượng cho metrics, config, artifacts, và lifecycle finish.
- Framework **MUST** có offline-capable logger.
- Nếu backend được cấu hình nhưng unavailable, framework **MUST** fallback rõ ràng hoặc fail rõ ràng theo policy cấu hình.

### REQ-OPS-002: CLI Surface

Framework **MUST** expose CLI đủ để train, eval, validate, và sweep.

Acceptance criteria:

- CLI **MUST** hỗ trợ tối thiểu các commands:
  - `train`
  - `eval`
  - `validate`
  - `sweep`
- CLI **MUST** dùng pipeline: load -> validate -> normalize trước khi dispatch command.
- `validate` **MUST** có chế độ không khởi tạo model hay external resources.
- CLI **SHOULD** in experiment identity khi bắt đầu run.

### REQ-OPS-003: Runtime and Environment Adaptation

Framework **SHOULD** hỗ trợ runtime adaptation theo platform nhưng không làm thay đổi semantics thí nghiệm.

Acceptance criteria:

- Runtime **MAY** điều chỉnh `num_workers`, `pin_memory`, output/checkpoint dirs theo platform.
- Platform adaptation **MUST NOT** thay đổi graph semantics, stage order, role semantics, hoặc loss semantics.
- Framework **SHOULD** cung cấp environment specs cho local/dev/cpu/gpu/kaggle hoặc tương đương.

### REQ-QUAL-001: Validation Boundaries and Failure Taxonomy

Framework **MUST** báo lỗi theo failure taxonomy rõ ràng tại đúng stage boundary.

Acceptance criteria:

- Ít nhất phải có error categories cho:
  - missing dependency
  - key collision
  - invalid role binding
  - invalid interaction graph/order
  - invalid decision output
  - invalid alpha range
  - invalid loss mapping
  - batch/data contract violation
  - state schema violation
  - numerical instability
  - checkpoint corruption
- Error message **MUST** chỉ rõ stage, component, và key/config gây lỗi khi có thể.

### REQ-QUAL-002: Numerical, Device, and Precision Safety

Framework **MUST** phát hiện numerical/device inconsistencies đủ sớm để debug được.

Acceptance criteria:

- Final logits, KD outputs, uncertainty outputs, và total loss **MUST** được validate finite khi strict validation bật.
- Device consistency **MUST** được validate tại stage boundaries hoặc trước compute nhạy cảm.
- Runtime **MAY** hỗ trợ `fp32` và mixed precision, nhưng precision mode **MUST** được cấu hình explicit.

### REQ-QUAL-003: Checkpoint and Resume

Framework **MUST** hỗ trợ checkpoint/resume đủ để phục hồi training state một cách nhất quán.

Acceptance criteria:

- Checkpoint **MUST** chứa:
  - model state
  - optimizer state
  - scheduler state nếu có
  - RNG state
  - epoch/step
  - experiment identity
  - normalized config snapshot
  - dataset/split metadata cần thiết
- Save **SHOULD** dùng atomic write hoặc cơ chế tương đương để tránh corrupt artifact.
- Resume **MUST** validate integrity trước khi restore.

### REQ-QUAL-004: Performance and Concurrency Constraints

Framework **MUST** giữ complexity và concurrency model đủ đơn giản để audit.

Acceptance criteria:

- DAG validation **SHOULD** là `O(V + E)`.
- Runtime graph execution **SHOULD** tránh dependency resolution quadratic.
- State **MUST NOT** bị deep-copied toàn bộ giữa mọi node như default behavior.
- Individual nodes/modules **MUST NOT** mutate State trực tiếp.
- Multi-worker và multi-process execution **MUST** không dựa vào shared mutable State.

---

## 9. Minimal Runnable Baseline For V1

Framework **MUST** hỗ trợ một baseline tối thiểu để verify pipeline hoạt động mà không cần toàn bộ UGTS stack.

Baseline tối thiểu gồm:

- student branch duy nhất;
- graph stage với ít nhất một encoder node và một head node;
- role binding cho `student.logits`;
- interaction stage ở dạng no-op;
- decision stage ở dạng identity/no-op nhưng vẫn giữ stage boundary;
- hard loss only;
- evaluation tối thiểu trên `S1`.

Acceptance criteria:

- Forward pass hoàn chỉnh **MUST** chạy được mà không cần teacher, KD, hoặc uncertainty.
- Final `logits` **MUST** vẫn được tạo tại decision stage.
- Baseline config **MUST** có thể mở rộng dần lên full pipeline bằng cách thêm config sections/components mà không phải rewrite pipeline hoặc phá naming contracts.

---

## 10. Out Of Scope Cho Requirement V1

Các mục sau **không bắt buộc** trong V1, nhưng architecture **SHOULD** để ngỏ đường mở rộng:

- nhiều teachers hoặc nhiều students trong cùng run;
- decision allocation vector đầy đủ cho N branches;
- parallel execution trong interaction stage;
- online data acquisition trong training runtime;
- hard guarantees reproducibility giữa khác hardware, khác PyTorch release, hoặc khác CUDA stack.

---

## 11. Tác Động Tới Design Và Traceability

Vì tài liệu này đã được làm sạch lại theo hướng requirement engineering, các tài liệu sau **cần được sync lại ở bước tiếp theo**:

- `design.md`
- `traceability.md`
- các checkpoint implementation cũ nếu chúng giả định bypass stage hoặc API nội bộ quá cụ thể

Đặc biệt cần sync lại ba điểm:

- decision stage external contract;
- batch/data contract theo modality-conditional inputs;
- minimal baseline phải giữ đủ stage boundaries.

---

## 12. Checklist Review Cho Requirement Mới

Trước khi freeze version tiếp theo của tài liệu này, reviewer **SHOULD** xác nhận:

- mỗi requirement có thể verify được;
- không còn requirement nào vừa là requirement vừa là design detail;
- không còn mâu thuẫn giữa baseline tối thiểu và fixed pipeline invariant;
- không còn mâu thuẫn giữa modality asymmetry và batch/data contract;
- decision stage đã có external contract duy nhất, không còn hai API đối chọi;
- traceability matrix có thể map từng requirement sang design và test dự kiến.
