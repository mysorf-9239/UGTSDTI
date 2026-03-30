# Tài Liệu Yêu Cầu: UGTSDTI Baseline Model

## 1. Mục đích

Tài liệu này định nghĩa **yêu cầu hệ thống cho baseline model đầu tiên** chạy trên framework UGTSDTI hiện tại.

Baseline này là model tham chiếu để:

- kiểm chứng pipeline `Batch -> Graph -> Role Binding -> Interaction -> Decision -> Loss + Metrics`;
- chứng minh graph/node system hiện tại có thể train một model DTI thật, không chỉ placeholder runtimes;
- làm mốc so sánh cho các model `teacher`, `student`, `KD`, `uncertainty`, và `gate` về sau.

Baseline này phải **tối giản nhưng hoàn chỉnh**:

- single-branch;
- supervised;
- không teacher;
- không KD;
- không uncertainty;
- không learned decision logic.

---

## 2. Phạm vi

Spec này bao phủ:

1. baseline node plugins cho drug encoder, protein encoder, fusion, và head;
2. registry wiring cho các type keys baseline mới;
3. config baseline canonical chạy được với framework hiện tại;
4. training/evaluation validation cho baseline end-to-end;
5. test suite chứng minh baseline train được và produce metrics hợp lệ.

Spec này **không** bao phủ:

- teacher model thực sự;
- KD hoặc uncertainty;
- graph neural network hoặc transformer encoders nâng cao;
- benchmark chất lượng SOTA;
- data preprocessing mới ngoài những gì framework hiện có đã hỗ trợ.

---

## 3. Tiền Đề

Framework hiện tại đã có:

- graph planner / graph engine;
- state contract write-once;
- role binding;
- interaction stage với `noop`;
- decision identity path;
- loss composer, metrics reporter, trainer, evaluator, checkpoint, artifacts.

Spec baseline này **phải tận dụng những thứ đó** chứ không được bypass trainer/pipeline.

---

## 4. Mục Tiêu Chất Lượng

Kết thúc spec này, baseline phải đạt:

- **fully integrated**: chỉ dùng graph nodes + current executor flow;
- **trainable**: loss giảm trên fixture training có tín hiệu học được;
- **auditable**: outputs và state keys đúng naming contract;
- **config-driven**: build được hoàn toàn từ YAML config;
- **scenario-evaluable**: chạy eval qua `S1–S4` khi split artifacts có sẵn.

---

## 5. Yêu Cầu Hệ Thống

### REQ-BL-001: Baseline Must Respect Fixed Pipeline

Baseline model **MUST** chạy qua pipeline chuẩn của framework:

```text
Batch -> Graph -> Role Binding -> Interaction -> Decision -> Loss + Metrics
```

Acceptance criteria:

- Trainer **MUST NOT** gọi model trực tiếp ngoài graph/node system.
- Baseline **MUST** dùng `interaction.noop` hoặc equivalent no-op interaction thay vì bỏ qua interaction stage.
- Baseline **MUST** dùng decision identity path thay vì bypass final `logits` emission.

### REQ-BL-002: Baseline Must Be Single-Branch Student-Only

Baseline canonical run **MUST** là single-branch với một role `student`.

Acceptance criteria:

- Config baseline canonical **MUST NOT** yêu cầu `teacher`.
- `roles.student.outputs` **MUST** bind tới output logits của head node.
- Decision identity **MUST** lấy source từ `student.logits`.

### REQ-BL-003: Baseline Graph Topology

Baseline canonical graph **MUST** có topology:

```text
drug_encoder -> protein_encoder -> fusion -> head
```

Operational interpretation:

- `drug_encoder` và `protein_encoder` là hai node độc lập đọc batch inputs;
- `fusion` đọc embeddings của hai encoders;
- `head` đọc fusion embedding và sinh logits.

Acceptance criteria:

- Graph planner **MUST** resolve được tất cả dependencies.
- Baseline graph **MUST** không có cycle.
- Node outputs **MUST** follow `<node>.<attr>` naming.

### REQ-BL-004: Simple Drug Encoder Node

Framework **MUST** cung cấp node type `encoder.simple_drug`.

Acceptance criteria:

- Declared input **MUST** là `drug_seq`.
- Runtime **MUST** gồm:
  - token embedding;
  - sequence pooling;
  - linear projection tới `hidden_dim`.
- Runtime **MUST** output đúng một attr: `embedding`.
- Output shape **MUST** là `(B, D)`.

### REQ-BL-005: CNN Protein Encoder Node

Framework **MUST** cung cấp node type `encoder.cnn_protein`.

Acceptance criteria:

- Declared input **MUST** là `protein_seq`.
- Runtime **MUST** gồm:
  - token embedding;
  - `Conv1d`;
  - nonlinearity;
  - global pooling.
- Runtime **MUST** output đúng một attr: `embedding`.
- Output shape **MUST** là `(B, D)`.

### REQ-BL-006: Concat Fusion Node

Framework **MUST** cung cấp node type `fusion.concat`.

Acceptance criteria:

- Node **MUST** đọc đúng hai declared embeddings từ graph config.
- Runtime **MUST** concatenate theo feature dimension.
- Runtime **MAY** có projection layer sau concatenate.
- Runtime **MUST** output đúng một attr: `embedding`.

### REQ-BL-007: MLP Head Node

Framework **MUST** cung cấp node type `head.mlp`.

Acceptance criteria:

- Node **MUST** đọc fusion embedding.
- Runtime **MUST** implement MLP hai lớp:
  - `Linear -> ReLU -> Linear`
- Runtime **MUST** output attr `logits`.
- Runtime **MUST NOT** apply sigmoid trước khi loss vì hard loss path dùng BCE-with-logits semantics.

### REQ-BL-008: Registry Integration Must Be Deterministic and Fail-Closed

Baseline node types **MUST** được register vào graph registry mặc định của framework.

Acceptance criteria:

- Các type keys sau **MUST** có trong shipped registry:
  - `encoder.simple_drug`
  - `encoder.cnn_protein`
  - `fusion.concat`
  - `head.mlp`
- Registry lookup **MUST** deterministic.
- Duplicate registration cho cùng `type_key` **MUST** fail rõ thay vì overwrite silently.
- Việc thêm baseline nodes **MUST NOT** làm hỏng các legacy type keys đang còn được tests/config cũ dùng.

### REQ-BL-009: Canonical Baseline Config

Repo **MUST** ship một baseline config canonical chạy được với framework hiện tại.

Acceptance criteria:

- Config **MUST** có đủ sections required bởi validator hiện tại:
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
- Baseline config canonical **MUST** dùng:
  - `interaction.order: [noop]`
  - `decision.type: identity`
  - `decision.strategy: identity`
  - `decision.source_key: student.logits`
- Metrics config **SHOULD** enable ít nhất `auroc` và `auprc`.

### REQ-BL-010: Forward Pass Must Materialize Canonical Keys

Một forward pass baseline **MUST** materialize đủ các keys chính.

Acceptance criteria:

- State sau graph stage **MUST** có:
  - `drug_encoder.embedding`
  - `protein_encoder.embedding`
  - `fusion.embedding`
  - `head.logits`
- State sau role binding **MUST** có `student.logits`.
- State sau decision **MUST** có `logits`.
- State sau postprocess **MUST** có `loss.total` và các `metrics.*` đã enable.

### REQ-BL-011: Training Must Be Learnable

Baseline **MUST** train được trên ít nhất một fixture dataset có tín hiệu học được.

Acceptance criteria:

- Có integration test hoặc equivalent verification chứng minh loss giảm qua nhiều training steps.
- Training path **MUST** đi qua `Trainer.step()` / CLI train path hiện có, không dùng ad-hoc optimization loop bypass framework.
- Baseline nodes có parameters trainable **MUST** expose `parameters()`, `state_dict()`, và `load_state_dict()`.

### REQ-BL-012: Evaluation Must Support Scenario Reporting

Baseline **MUST** evaluate được qua scenario-based reporting của framework.

Acceptance criteria:

- `MetricsReporter` **MUST** produce scenario-wise metrics khi `metrics.by_scenario: true`.
- Baseline canonical tests **MUST** cover ít nhất một run có `scenario` metadata trong batch.
- Baseline evaluation path **SHOULD** run được trên materialized split artifacts `S1–S4`.

### REQ-BL-013: Baseline Must Produce Reproducible Runtime State

Baseline trainable runtimes **MUST** tham gia đầy đủ vào checkpoint/artifact lineage.

Acceptance criteria:

- `PipelineExecutor.model_state()` **MUST** collect được baseline node states.
- `PipelineExecutor.load_model_state()` **MUST** restore được baseline node states.
- Có test chứng minh baseline runtime state round-trip qua checkpoint/model_state không nổ.

### REQ-BL-014: Minimalism Over Over-Engineering

Baseline **MUST** giữ minimal surface cần thiết, không thêm complexity research chưa dùng tới.

Acceptance criteria:

- Không thêm teacher branch giả chỉ để “đồng bộ” với future architecture.
- Không thêm uncertainty, gate, hoặc KD vào baseline canonical config.
- Không thêm abstraction layer mới nếu NodeRuntime hiện tại đã đủ để biểu diễn baseline.

### REQ-BL-015: Traceability Update

Spec baseline **MUST** được trace tới tasks, tests, config, và runtime registry updates.

Acceptance criteria:

- Mỗi `REQ-BL-*` **MUST** có owner trong `tasks.md`.
- Baseline config file và baseline tests **MUST** map được về spec này.
- Nếu public config surface mới được thêm, docs liên quan **SHOULD** được sync.

---

## 6. Định Nghĩa Done Cho Baseline

Baseline được coi là hoàn tất khi:

- graph baseline build và run end-to-end không lỗi;
- state materialize đủ `logits`, `loss.total`, `metrics.*`;
- có ít nhất một test chứng minh loss giảm trên fixture có learnable signal;
- registry mặc định biết các baseline node types mới;
- checkpoint/model_state round-trip chạy được cho baseline nodes;
- baseline config canonical có thể được dùng làm mốc cho các model `teacher/student` phức tạp hơn về sau.
