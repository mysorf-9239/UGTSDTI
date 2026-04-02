# Tasks: UGTSDTI Flow 1 — Contract-Compliant Teacher-Student

## Phase 1: Node Runtimes (`ugtsdti/nodes/flow1.py`)

- [ ] 1.1 Tạo `ugtsdti/nodes/flow1.py` với `SeqBiLSTMEncoderRuntime` (`encoder.seq_bilstm`)
  - [x] 1.1.1 Class `SeqBiLSTMEncoderRuntime(_TorchRuntime)` với params: `vocab_size`, `embedding_dim`, `hidden_dim`, `kernel_size_1`, `kernel_size_2`, `lstm_dim`, `proj_dim`
  - [x] 1.1.2 Architecture: Embedding → Conv1d(block1) → ReLU → Conv1d(block2) → ReLU → BiLSTM(bidirectional=True) → mean pooling → Linear projection
  - [x] 1.1.3 `nn.LSTM(hidden_dim, lstm_dim, bidirectional=True, batch_first=True)`
  - [x] 1.1.4 Input: đọc bằng `inputs[self.inputs[0]]` — KHÔNG dùng `first_value(inputs)` hay dict-order-dependent access
  - [x] 1.1.5 Token ids clamp vào `[0, vocab_size)` trước embedding
  - [x] 1.1.6 Validate input rank == 2, raise `InvalidConfigError` với stage và key rõ ràng
  - [x] 1.1.7 Reuse `_to_token_ids` từ `baseline.py`

- [x] 1.2 Implement `GNNDrugEncoderRuntime` (`encoder.gnn_drug`) trong `flow1.py`
  - [x] 1.2.1 Architecture: Linear(F→H) → bmm(adj_norm, h) → ReLU [block1] → Linear(H→H) → bmm(adj_norm, h) → ReLU [block2] → mean pooling → projection
  - [x] 1.2.2 Input: đọc `inputs["drug_graph"]` — dict với keys `adj` `(B,N,N)` và `node_feat` `(B,N,F)`
  - [x] 1.2.3 D^{-1} row-normalization khi `normalize_adj=True`: `adj / adj.sum(dim=-1, keepdim=True).clamp_min(1.0)`
  - [x] 1.2.4 **Missing modality**: khi `allow_missing_input=True` và `drug_graph` không có hoặc None, trả về `torch.zeros(B, proj_dim)` — B suy ra từ context hoặc batch key khác
  - [x] 1.2.5 Params: `node_feat_dim=9`, `hidden_dim=64`, `proj_dim=64`, `normalize_adj=True`, `allow_missing_input=False`
  - [x] 1.2.6 Không import `torch_geometric`

- [ ] 1.3 Implement `DenseHeadRuntime` (`head.dense`) trong `flow1.py`
  - [x] 1.3.1 Architecture: Linear(D→hidden_dim) → ReLU → Linear(hidden_dim→output_dim)
  - [x] 1.3.2 Input: đọc bằng `inputs[self.inputs[0]]`
  - [x] 1.3.3 Output key `logits`, KHÔNG có sigmoid
  - [x] 1.3.4 Params: `hidden_dim=64`, `output_dim=1`, `input_dim=None` (LazyLinear nếu None)
  - [x] 1.3.5 Validate input rank == 2

## Phase 2: Registry & Fusion Fix

- [x] 2.1 Đăng ký 3 type keys mới trong `ugtsdti/runtime/defaults.py`
  - [x] 2.1.1 Import `SeqBiLSTMEncoderRuntime`, `GNNDrugEncoderRuntime`, `DenseHeadRuntime` từ `flow1.py`
  - [x] 2.1.2 Register `encoder.seq_bilstm` với `output_attrs=["embedding"]`
  - [x] 2.1.3 Register `encoder.gnn_drug` với `output_attrs=["embedding"]`, `input_kinds=["drug_graph"]`
  - [x] 2.1.4 Register `head.dense` với `output_attrs=["logits"]`, `input_kinds=["embedding"]`

- [x] 2.2 Fix `ConcatFusionRuntime` trong `ugtsdti/nodes/baseline.py`
  - [x] 2.2.1 Thay constraint `len(tensors) == 2` bằng `len(tensors) >= 2`
  - [x] 2.2.2 Đọc inputs theo thứ tự khai báo: `tensors = [inputs[k] for k in self.inputs]` — KHÔNG dùng `inputs.values()`
  - [x] 2.2.3 Validate rank == 2 và batch dim match cho tất cả N tensors
  - [x] 2.2.4 Đảm bảo N == 2 vẫn backward compatible

## Phase 3: Bug Fix — lambda_max gap

- [x] 3.1 Fix `_scheduled_kd_weight()` trong `ugtsdti/trainer/trainer.py`
  - [x] 3.1.1 Khi `schedule == "warmup"`: đọc `lambda_max` từ `cfg["training"]["kd"]["lambda_max"]`
  - [x] 3.1.2 Formula: `lambda_max * min(1.0, (step_idx + 1) / warmup_steps)`
  - [x] 3.1.3 Khi `schedule == "constant"`: fallback về `loss.map.kd.weight` như hiện tại
  - [x] 3.1.4 Raise `InvalidConfigError` cho schedule không hợp lệ

## Phase 4: Config Files

- [x] 4.1 Tạo `configs/flow1_teacher_student.yaml`
  - [x] 4.1.1 Thêm `contract_version: v1` và `runtime.seed: 42`
  - [x] 4.1.2 Teacher graph: `t_drug_enc` (seq_bilstm, vocab=64), `t_prot_enc` (seq_bilstm, vocab=32), `t_graph_enc` (gnn_drug, allow_missing_input=true), `teacher_fusion` (3-way concat), `teacher_head` (head.dense)
  - [x] 4.1.3 Student graph: `s_drug_enc` (seq_bilstm, vocab=64), `s_prot_enc` (seq_bilstm, vocab=32), `student_fusion` (2-way concat), `student_head` (head.dense)
  - [x] 4.1.4 Interaction: khai báo explicit `inputs` list cho cả `kd` và `uncertainty`
  - [x] 4.1.5 Decision: khai báo explicit `inputs: [teacher.logits, student.logits, student.var]`
  - [x] 4.1.6 Training: `kd.schedule: warmup`, `kd.warmup_steps: 100`, `kd.lambda_max: 0.5`
  - [x] 4.1.7 Decision fallback: `no_teacher: student`, `no_student: teacher`, `no_uncertainty: equal_blend`

- [x] 4.2 Tạo `configs/profiles/flow1_local.yaml`
  - [x] 4.2.1 `extends: [../flow1_teacher_student.yaml]`
  - [x] 4.2.2 Override: `runtime.device: cpu`, `runtime.seed: 42`, `training.loop.epochs: 5`

## Phase 5: Unit Tests (`tests/quality/test_flow1_nodes.py`)

- [ ] 5.1 Tests cho `SeqBiLSTMEncoderRuntime`
  - [x] 5.1.1 Output shape `(B, proj_dim)` với drug_seq key
  - [x] 5.1.2 Output shape `(B, proj_dim)` với protein_seq key (khác vocab_size)
  - [x] 5.1.3 Output values finite
  - [x] 5.1.4 Input rank != 2 raises `InvalidConfigError`
  - [x] 5.1.5 `state_dict()` / `load_state_dict()` round-trip
  - [x] 5.1.6 Gradient flow: `.backward()` không raise, source.grad không None

- [x] 5.2 Tests cho `GNNDrugEncoderRuntime`
  - [x] 5.2.1 Output shape `(B, proj_dim)` với adj + node_feat hợp lệ
  - [x] 5.2.2 Output values finite
  - [x] 5.2.3 Adj normalization: row sums ≈ 1.0 sau normalize
  - [x] 5.2.4 **Missing modality**: khi `allow_missing_input=True` và drug_graph=None → output là zeros(B, proj_dim)
  - [x] 5.2.5 **Missing modality**: khi `allow_missing_input=False` và drug_graph=None → raises `MissingDependencyError`
  - [x] 5.2.6 `state_dict()` / `load_state_dict()` round-trip
  - [x] 5.2.7 Gradient flow

- [ ] 5.3 Tests cho `DenseHeadRuntime`
  - [x] 5.3.1 Output shape `(B, output_dim)`, no sigmoid (logits có thể âm)
  - [x] 5.3.2 Input rank != 2 raises `InvalidConfigError`
  - [x] 5.3.3 `state_dict()` / `load_state_dict()` round-trip
  - [x] 5.3.4 Gradient flow

- [x] 5.4 Tests cho `fusion.concat` N-input
  - [x] 5.4.1 3 inputs: output shape `(B, project_dim)` đúng
  - [x] 5.4.2 2 inputs: backward compatible
  - [x] 5.4.3 Batch mismatch raises `InvalidConfigError`
  - [x] 5.4.4 Rank != 2 raises `InvalidConfigError`
  - [x] 5.4.5 Inputs được đọc theo thứ tự khai báo (không phụ thuộc dict order)

- [x] 5.5 Tests cho `_scheduled_kd_weight()`
  - [x] 5.5.1 step=0, warmup_steps=100, lambda_max=0.5 → lambda ≈ 0.005
  - [x] 5.5.2 step=99, warmup_steps=100, lambda_max=0.5 → lambda == 0.5
  - [x] 5.5.3 step=200, warmup_steps=100, lambda_max=0.5 → lambda == 0.5 (capped)
  - [x] 5.5.4 schedule=constant → đọc từ loss.map.kd.weight

- [ ] 5.6 Tests registry
  - [x] 5.6.1 3 type keys mới có trong `registry.registered_type_keys()`
  - [x] 5.6.2 Duplicate registration raises `InvalidConfigError`

## Phase 6: Integration Tests (`tests/integration/test_flow1_pipeline.py`)

- [x] 6.1 Full pipeline S1 (warm-start)
  - [x] 6.1.1 Batch có `drug_seq`, `drug_graph`, `protein_seq`, `labels`
  - [x] 6.1.2 Assert state keys: `teacher.logits`, `student.logits`, `logits`, `gate.alpha`, `interaction.kd.loss_component`, `teacher.var`, `student.var`, `loss.total`
  - [x] 6.1.3 Assert tất cả output tensors finite
  - [x] 6.1.4 Assert `gate.alpha ∈ [0, 1]`

- [x] 6.2 Pipeline S2/S4 — missing drug_graph
  - [x] 6.2.1 Batch KHÔNG có `drug_graph`
  - [x] 6.2.2 Assert `t_graph_enc.embedding` là zeros
  - [x] 6.2.3 Assert pipeline không crash
  - [x] 6.2.4 Assert student pipeline chạy thành công (không cần drug_graph)

- [x] 6.3 Decision fallback
  - [x] 6.3.1 Khi không có `teacher.logits` → dùng student fallback
  - [x] 6.3.2 Khi không có `student.var` → dùng equal_blend (alpha=0.5)

- [x] 6.4 Frozen teacher
  - [x] 6.4.1 Config `training.teacher.freeze: true`
  - [x] 6.4.2 Assert teacher params không thay đổi sau training step
  - [x] 6.4.3 Assert student params thay đổi sau training step

- [x] 6.5 KD warmup
  - [x] 6.5.1 step=0 → effective lambda ≈ 0
  - [x] 6.5.2 step=warmup_steps → effective lambda == lambda_max

- [x] 6.6 Config validation
  - [x] 6.6.1 `flow1_teacher_student.yaml` load và pass validation
  - [x] 6.6.2 `flow1_local.yaml` resolve extends và pass validation

## Phase 7: Traceability Update

- [x] 7.1 Cập nhật `.docs/traceability.md` với 3 node type keys mới và mapping tới REQ-FLOW1-001 đến REQ-FLOW1-020
