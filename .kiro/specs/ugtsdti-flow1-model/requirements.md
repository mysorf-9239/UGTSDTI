# Requirements: UGTSDTI Flow 1 — Teacher-Student Model (Contract-Compliant)

## Quy ước

- `MUST`: bắt buộc, verifiable bằng test hoặc inspection.
- `MUST NOT`: cấm tuyệt đối.
- `SHOULD`: khuyến nghị mạnh.
- Mọi requirement đều traceable tới contracts.md và có acceptance criteria kiểm chứng được.

---

## REQ-FLOW1-001: Node `encoder.seq_bilstm` — Sequence Encoder

**Mô tả**: Framework MUST cung cấp graph node type `encoder.seq_bilstm` encode sequence (drug hoặc protein) qua 2 CNN1D blocks + BiLSTM. Cùng type key được dùng cho cả drug và protein encoder, phân biệt bởi `vocab_size` trong params.

**Acceptance Criteria**:

1. Node khai báo đúng một input key trong `inputs` list (ví dụ `drug_seq` hoặc `protein_seq`). Node MUST đọc input bằng `inputs[self.declared_input_key]`, KHÔNG dùng `first_value(inputs)` hay dict-order-dependent access.
2. Architecture: `Embedding(vocab_size, embedding_dim) → Conv1d(block1) → ReLU → Conv1d(block2) → ReLU → BiLSTM(bidirectional=True) → mean pooling → Linear projection`.
3. BiLSTM: `nn.LSTM(hidden_dim, lstm_dim, bidirectional=True, batch_first=True)` — output `(B, L, 2*lstm_dim)` trước pooling.
4. Output key là `embedding`, shape `(B, proj_dim)`, tất cả values finite.
5. Params với defaults: `vocab_size=64`, `embedding_dim=32`, `hidden_dim=64`, `kernel_size_1=3`, `kernel_size_2=5`, `lstm_dim=64`, `proj_dim=64`.
6. Gradient flow qua toàn bộ node kể cả BiLSTM — `.backward()` không raise.
7. `state_dict()` / `load_state_dict()` round-trip: weights khớp sau restore.
8. Input rank != 2 MUST raise `InvalidConfigError` với message rõ stage và key.
9. Token ids MUST được clamp vào `[0, vocab_size)` trước khi vào embedding.

---

## REQ-FLOW1-002: Node `encoder.gnn_drug` — GNN Drug Structure Encoder

**Mô tả**: Framework MUST cung cấp graph node type `encoder.gnn_drug` encode drug molecular graph qua 2 GCN-style blocks. MUST NOT dùng `torch_geometric` hoặc sparse ops ngoài PyTorch core.

**Acceptance Criteria**:

1. Node khai báo input key `drug_graph` trong `inputs` list. Node MUST đọc `inputs["drug_graph"]`.
2. `drug_graph` là dict với keys `adj` shape `(B, N, N)` float và `node_feat` shape `(B, N, F)` float.
3. Architecture: `Linear(F→H) → bmm(adj_norm, h) → ReLU [block1] → Linear(H→H) → bmm(adj_norm, h) → ReLU [block2] → mean pooling over N → Linear projection`.
4. Khi `normalize_adj=True`: `adj_norm = adj / deg.clamp_min(1.0)` với `deg = adj.sum(dim=-1, keepdim=True)` — D^{-1} row-normalization.
5. Output key là `embedding`, shape `(B, proj_dim)`, tất cả values finite.
6. Params với defaults: `node_feat_dim=9`, `hidden_dim=64`, `proj_dim=64`, `normalize_adj=True`.
7. **Missing modality handling (S2, S4)**: Khi `allow_missing_input=True` và `drug_graph` không có trong batch, node MUST trả về zero embedding `torch.zeros(B, proj_dim)` — deterministic, không crash. Batch size `B` được suy ra từ context hoặc một batch key khác.
8. Gradient flow qua toàn bộ node.
9. `state_dict()` / `load_state_dict()` round-trip.
10. `adj` rank != 3 hoặc `node_feat` rank != 3 MUST raise `InvalidConfigError`.

---

## REQ-FLOW1-003: Node `head.dense` — Dense MLP Head

**Mô tả**: Framework MUST cung cấp graph node type `head.dense` là 2-layer Dense MLP head, dùng cho cả teacher và student.

**Acceptance Criteria**:

1. Node khai báo đúng một input key trong `inputs` list. Node MUST đọc input bằng `inputs[self.declared_input_key]`.
2. Architecture: `Linear(D→hidden_dim) → ReLU → Linear(hidden_dim→output_dim)`.
3. Output key là `logits`, shape `(B, output_dim)`, tất cả values finite. MUST NOT có sigmoid.
4. Params với defaults: `hidden_dim=64`, `output_dim=1`, `input_dim=None` (dùng `nn.LazyLinear` nếu None).
5. Gradient flow qua toàn bộ node.
6. `state_dict()` / `load_state_dict()` round-trip. Khi dùng `LazyLinear` và chưa forward, `state_dict()` MUST trả về empty dict (không serialize uninitialized weights).
7. Input rank != 2 MUST raise `InvalidConfigError`.

---

## REQ-FLOW1-004: Registry — 3 type keys mới

**Mô tả**: `build_default_graph_registry()` MUST đăng ký 3 type keys mới của Flow 1.

**Acceptance Criteria**:

1. `encoder.seq_bilstm` đăng ký với `output_attrs=["embedding"]`.
2. `encoder.gnn_drug` đăng ký với `output_attrs=["embedding"]`, `input_kinds=["drug_graph"]`.
3. `head.dense` đăng ký với `output_attrs=["logits"]`, `input_kinds=["embedding"]`.
4. Duplicate registration của bất kỳ type key nào MUST raise `InvalidConfigError` (fail-fast, không overwrite silently).
5. Các type keys cũ (`encoder.simple_drug`, `encoder.cnn_protein`, `fusion.concat`, `head.mlp`, v.v.) MUST vẫn hoạt động.

---

## REQ-FLOW1-005: `fusion.concat` hỗ trợ N ≥ 2 inputs

**Mô tả**: `ConcatFusionRuntime` MUST hỗ trợ N ≥ 2 embedding inputs.

**Acceptance Criteria**:

1. `fusion.concat` với 3 inputs (teacher 3-way fusion) hoạt động đúng.
2. `fusion.concat` với 2 inputs (student 2-way fusion) vẫn hoạt động đúng (backward compatible).
3. Output shape: `(B, sum(D_i))` trước projection, `(B, project_dim)` sau projection.
4. Batch dimension mismatch giữa bất kỳ 2 inputs MUST raise `InvalidConfigError` với message rõ.
5. Rank != 2 của bất kỳ input nào MUST raise `InvalidConfigError`.
6. Inputs MUST được đọc theo thứ tự khai báo trong `inputs` list của node config — KHÔNG phụ thuộc dict insertion order.

---

## REQ-FLOW1-006: Teacher graph topology

**Mô tả**: Config MUST cho phép dựng teacher graph với 2 `encoder.seq_bilstm` + 1 `encoder.gnn_drug` + 3-way `fusion.concat` + `head.dense`.

**Acceptance Criteria**:

1. Graph plan build thành công từ config với nodes: `t_drug_enc`, `t_prot_enc`, `t_graph_enc`, `teacher_fusion`, `teacher_head`.
2. `teacher_fusion` nhận 3 inputs: `t_drug_enc.embedding`, `t_prot_enc.embedding`, `t_graph_enc.embedding` — theo thứ tự khai báo.
3. `teacher_head` produce `teacher_head.logits` shape `(B, 1)`.
4. Role binding map `teacher_head.logits` → `teacher.logits` (key namespace: `<role>.logits`).
5. Forward pass với batch có `drug_seq`, `drug_graph`, `protein_seq` không raise.
6. Tất cả output keys tuân theo `<node_name>.<attribute>` naming convention.

---

## REQ-FLOW1-007: Student graph topology

**Mô tả**: Config MUST cho phép dựng student graph với 2 `encoder.seq_bilstm` + 2-way `fusion.concat` + `head.dense`. Student MUST NOT có GNN encoder. Student MUST có fusion layer.

**Acceptance Criteria**:

1. Graph plan build thành công từ config với nodes: `s_drug_enc`, `s_prot_enc`, `student_fusion`, `student_head`.
2. `student_fusion` nhận 2 inputs: `s_drug_enc.embedding`, `s_prot_enc.embedding`.
3. `student_head` produce `student_head.logits` shape `(B, 1)`.
4. Role binding map `student_head.logits` → `student.logits`.
5. Forward pass MUST NOT require `drug_graph` trong batch.
6. Student MUST NOT có `encoder.gnn_drug` node.
7. Student MUST có `fusion.concat` node (không skip fusion).

---

## REQ-FLOW1-008: Execution phase ordering — Interaction reads role outputs only

**Mô tả**: Interaction stage MUST chỉ đọc role outputs (`teacher.logits`, `student.logits`) và prior interaction outputs. MUST NOT đọc graph outputs trực tiếp.

**Acceptance Criteria**:

1. KD interaction khai báo `inputs: [teacher.logits, student.logits]` — đây là role outputs, không phải graph outputs.
2. Uncertainty interaction khai báo `inputs: [teacher.logits, student.logits]`.
3. `InteractionEngine._materialize_inputs()` MUST raise `MissingDependencyError` nếu bất kỳ declared input nào không có trong state tại thời điểm interaction chạy.
4. Interaction MUST chạy sau role binding — pipeline order: graph → role_binding → interaction.
5. Interaction MUST NOT khai báo input là graph-namespace keys như `teacher_head.logits` hay `t_drug_enc.embedding`.

---

## REQ-FLOW1-009: Decision module — explicit inputs

**Mô tả**: Decision module MUST có explicit declared inputs. MUST NOT đọc state implicitly.

**Acceptance Criteria**:

1. Config `decision` section MUST khai báo inputs: `[teacher.logits, student.logits, student.var]`.
2. `SoftBlendingDecisionModule` MUST đọc `teacher.logits`, `student.logits` từ state qua declared keys.
3. `SoftBlendingDecisionModule` MUST đọc `student.var` từ state để tính `gate.alpha`.
4. Nếu `student.var` không có trong state, decision MUST fallback theo `fallback.no_uncertainty` config — không crash.
5. Decision MUST produce `logits` và `gate.alpha` — cả hai phải finite.
6. `gate.alpha ∈ [0, 1]` cho mọi batch.

---

## REQ-FLOW1-010: KD interaction — logits-based với temperature và numerical stability

**Mô tả**: KD interaction MUST hoạt động đúng với `temperature=4.0` và MUST đảm bảo numerical stability.

**Acceptance Criteria**:

1. `kd.standard` với `mode: logits`, `temperature: 4.0` produce `interaction.kd.loss_component` non-negative.
2. Softened probabilities MUST được clamp: `p = clamp(p, 1e-6, 1 - 1e-6)` trước KL divergence để tránh NaN/Inf.
3. `kd.teacher_target` và `kd.student_target` là softened probability distributions, sum to 1 per sample.
4. KD loss MUST scale bởi `T²` theo công thức Hinton.
5. Cả `teacher.logits` và `student.logits` MUST có mặt trong state trước khi KD chạy (enforced bởi InteractionEngine).
6. Output keys: `interaction.kd.loss_component`, `kd.teacher_target`, `kd.student_target` — tuân theo naming contract.

---

## REQ-FLOW1-011: Uncertainty interaction — confidence proxy

**Mô tả**: Uncertainty interaction MUST compute confidence proxy cho cả teacher và student branches.

**Acceptance Criteria**:

1. `uncertainty.confidence_proxy` produce `teacher.var` và `student.var`.
2. Computation: `var = sigmoid(logits) * (1 - sigmoid(logits))` — values trong `[0, 0.25]`.
3. Non-finite logits input MUST raise `InvalidInteractionGraphError`.
4. Output keys tuân theo naming contract: `teacher.var`, `student.var`.

---

## REQ-FLOW1-012: Soft blending decision — uncertainty-gated alpha

**Mô tả**: Decision module MUST blend teacher và student logits dựa trên uncertainty-derived alpha.

**Acceptance Criteria**:

1. `SoftBlendingDecisionModule` produce `logits` và `gate.alpha`.
2. `gate.alpha ∈ [0, 1]` cho mọi batch — enforced bằng clamp hoặc sigmoid.
3. `logits = α × teacher.logits + (1-α) × student.logits`.
4. Fallback khi thiếu teacher: dùng `student.logits` trực tiếp.
5. Fallback khi thiếu student: dùng `teacher.logits` trực tiếp.
6. Fallback khi thiếu `student.var`: dùng `alpha=0.5` (equal blending) hoặc theo `fallback.no_uncertainty` config.

---

## REQ-FLOW1-013: Missing modality handling — S2, S3, S4 scenarios

**Mô tả**: Framework MUST xử lý missing modality một cách deterministic, không crash.

**Acceptance Criteria**:

1. Khi `drug_graph` không có trong batch và `t_graph_enc` có `allow_missing_input: true`, node MUST trả về zero embedding `torch.zeros(B, proj_dim)` — deterministic.
2. Batch size `B` MUST được suy ra từ một batch key khác (ví dụ `drug_seq.shape[0]`).
3. Zero embedding MUST có cùng shape và dtype như normal embedding.
4. Pipeline MUST chạy thành công cho S2 (cold drug — không có drug_graph), S3 (cold target), S4 (fully cold).
5. Student pipeline MUST chạy thành công mà không cần `drug_graph` trong batch (student không có GNN node).
6. Missing modality behavior MUST được document trong node config params.

---

## REQ-FLOW1-014: KD warmup schedule — lambda_max từ training.kd

**Mô tả**: Training pipeline MUST hỗ trợ KD warmup schedule. `_scheduled_kd_weight()` MUST đọc `lambda_max` từ `training.kd`, không phải từ `loss.map.kd.weight`.

**Acceptance Criteria**:

1. Config hỗ trợ `training.kd.schedule: warmup`, `training.kd.warmup_steps`, `training.kd.lambda_max`.
2. `_scheduled_kd_weight()` khi `schedule == "warmup"`: đọc `lambda_max` từ `cfg["training"]["kd"]["lambda_max"]`.
3. Formula: `λ(step) = lambda_max × min(1.0, (step + 1) / warmup_steps)`.
4. Tại step 0: `λ ≈ lambda_max / warmup_steps` (gần 0 khi warmup_steps lớn).
5. Tại step `warmup_steps - 1`: `λ == lambda_max`.
6. Lambda monotonically non-decreasing trong warmup period.
7. Sau warmup: lambda cố định tại `lambda_max`.
8. Khi `schedule == "constant"`: fallback về `loss.map.kd.weight` như hiện tại.

---

## REQ-FLOW1-015: Teacher freeze mode

**Mô tả**: Training pipeline MUST hỗ trợ frozen teacher mode.

**Acceptance Criteria**:

1. Khi `training.teacher.freeze: true`, teacher graph node params MUST NOT được đưa vào optimizer.
2. Sau một training step với freeze=true, teacher params MUST không thay đổi (verified bằng tensor equality).
3. Student params MUST được update bình thường.
4. Khi `training.teacher.freeze: false` (joint training), cả teacher và student params đều được update.

---

## REQ-FLOW1-016: Key namespace compliance

**Mô tả**: Tất cả output keys MUST tuân theo naming convention của contracts.md.

**Acceptance Criteria**:

1. Graph node outputs: `<node_name>.<attribute>` — ví dụ `t_drug_enc.embedding`, `teacher_head.logits`.
2. Role outputs: `<role>.logits` — ví dụ `teacher.logits`, `student.logits`.
3. Interaction outputs: stable explicit names — `interaction.kd.loss_component`, `kd.teacher_target`, `kd.student_target`, `teacher.var`, `student.var`.
4. Decision outputs: `logits`, `gate.alpha`.
5. Loss outputs: `loss.*` — ví dụ `loss.total`, `loss.hard`, `loss.kd`.
6. Metric outputs: `metrics.*` — ví dụ `metrics.auroc`, `metrics.auprc`.
7. MUST NOT có key nào vi phạm namespace (ví dụ interaction output dùng graph namespace).

---

## REQ-FLOW1-017: Validation rules — error taxonomy

**Mô tả**: Framework MUST raise lỗi đúng type tại đúng stage boundary.

**Acceptance Criteria**:

| Case | Error type | Stage |
|---|---|---|
| Missing declared input key | `MissingDependencyError` | graph / interaction |
| Duplicate state key | `KeyCollisionError` | state |
| Shape mismatch trong fusion | `InvalidConfigError` | graph |
| Invalid decision input (non-finite) | `InvalidDecisionOutputError` | decision |
| Interaction reads undeclared key | `MissingDependencyError` | interaction |
| Node input rank sai | `InvalidConfigError` | graph |
| Unregistered type key | `MissingDependencyError` | graph |

---

## REQ-FLOW1-018: Config `flow1_teacher_student.yaml` — contract-compliant

**Mô tả**: Config MUST đầy đủ, không có implicit behavior, pass validation.

**Acceptance Criteria**:

1. File `configs/flow1_teacher_student.yaml` tồn tại và load được.
2. Config có `contract_version: v1` và `runtime.seed: 42`.
3. Config pass validation (tất cả required sections có mặt).
4. Teacher graph: 2 `encoder.seq_bilstm` + 1 `encoder.gnn_drug` (với `allow_missing_input: true`) + 3-way `fusion.concat` + `head.dense`.
5. Student graph: 2 `encoder.seq_bilstm` + 2-way `fusion.concat` + `head.dense`.
6. Interaction modules khai báo explicit `inputs` list.
7. Decision section khai báo explicit `inputs` list.
8. `training.kd.lambda_max: 0.5`, `training.kd.warmup_steps: 100`.

---

## REQ-FLOW1-019: Config `profiles/flow1_local.yaml`

**Mô tả**: Repo MUST ship local profile cho Flow 1.

**Acceptance Criteria**:

1. File `configs/profiles/flow1_local.yaml` tồn tại và load được.
2. Extends `flow1_teacher_student.yaml` qua `extends` mechanism.
3. Override `runtime.device: cpu`, `runtime.seed: 42`, `training.loop.epochs: 5`.
4. Config sau khi resolve extends pass validation.

---

## REQ-FLOW1-020: Full pipeline integration — S1 đến S4

**Mô tả**: Full Flow 1 pipeline MUST chạy end-to-end cho tất cả scenarios S1–S4.

**Acceptance Criteria**:

1. S1 (warm-start): batch có `drug_seq`, `drug_graph`, `protein_seq`, `labels` — full teacher + student pipeline chạy thành công.
2. S2/S3/S4 (cold-start): batch không có `drug_graph` — student pipeline chạy thành công, teacher dùng zero embedding cho GNN branch.
3. State sau run_batch MUST có: `teacher.logits`, `student.logits`, `logits`, `gate.alpha`, `interaction.kd.loss_component`, `teacher.var`, `student.var`, `loss.total`.
4. `metrics.auroc` và `metrics.auprc` được compute by scenario.
5. Tất cả output tensors finite.
6. Deterministic: cùng batch + seed → cùng output.
