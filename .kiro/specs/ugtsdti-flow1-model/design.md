# Tài Liệu Thiết Kế: UGTSDTI Flow 1 — Contract-Compliant Teacher-Student

## 1. Mục đích

Thiết kế **Hướng xây dựng 1 (Flow 1)**: Teacher multimodal (sequence + structure) + Student sequence-only.
Tài liệu này 100% tuân thủ `contracts.md`. Mọi dependency đều explicit. Không có implicit behavior.

---

## 2. So sánh Teacher vs Student

| Thành phần | Teacher | Student |
|---|---|---|
| Input | drug_seq + protein_seq + drug_graph | drug_seq + protein_seq |
| Drug encoder | `encoder.seq_bilstm` (vocab=64) | `encoder.seq_bilstm` (vocab=64) |
| Protein encoder | `encoder.seq_bilstm` (vocab=32) | `encoder.seq_bilstm` (vocab=32) |
| Structure encoder | `encoder.gnn_drug` (allow_missing) | — |
| Fusion | 3-way concat | 2-way concat |
| Head | `head.dense` | `head.dense` |
| Phù hợp | S1 (warm-start) | S2, S3, S4 (cold-start) |

---

## 3. Execution Phase — State Flow

Pipeline cố định theo contracts.md §17.1:

```
initialize_state(batch)
  → graph stage
  → role_binding
  → interaction stage
  → decision stage
  → loss + metrics
```

### 3.1 State keys theo phase

| Phase | Keys được commit vào State |
|---|---|
| Batch | `drug_seq`, `drug_graph`*, `protein_seq`, `labels`, `scenario` |
| Graph | `t_drug_enc.embedding`, `t_prot_enc.embedding`, `t_graph_enc.embedding`*, `teacher_fusion.embedding`, `teacher_head.logits`, `s_drug_enc.embedding`, `s_prot_enc.embedding`, `student_fusion.embedding`, `student_head.logits` |
| Role Binding | `teacher.logits`, `student.logits` |
| Interaction | `interaction.kd.loss_component`, `kd.teacher_target`, `kd.student_target`, `teacher.var`, `student.var` |
| Decision | `logits`, `gate.alpha` |
| Loss | `loss.hard`, `loss.kd`, `loss.total` |
| Metrics | `metrics.auroc`, `metrics.auprc`, `metrics.<scenario>.auroc`, ... |

*`drug_graph` và `t_graph_enc.embedding` optional khi `allow_missing_input: true`.

### 3.2 Access discipline (contracts.md §6.3)

- Graph nodes: chỉ đọc batch keys và graph outputs đã khai báo trong `inputs`.
- Interaction modules: chỉ đọc role outputs (`teacher.logits`, `student.logits`) và prior interaction outputs.
- Decision module: chỉ đọc role outputs và interaction outputs — explicit trong config.
- Loss: chỉ đọc `logits`, `labels`, và interaction outputs được map tường minh.

---

## 4. Graph Topology

### 4.1 Teacher graph

```
drug_seq    → t_drug_enc (encoder.seq_bilstm, vocab=64)  → t_drug_enc.embedding  ──┐
protein_seq → t_prot_enc (encoder.seq_bilstm, vocab=32)  → t_prot_enc.embedding  ──┼→ teacher_fusion → teacher_head → teacher_head.logits
drug_graph  → t_graph_enc (encoder.gnn_drug, allow_miss) → t_graph_enc.embedding ──┘
```

Role binding: `teacher_head.logits` → `teacher.logits`

### 4.2 Student graph

```
drug_seq    → s_drug_enc (encoder.seq_bilstm, vocab=64) → s_drug_enc.embedding ──┐
protein_seq → s_prot_enc (encoder.seq_bilstm, vocab=32) → s_prot_enc.embedding ──┴→ student_fusion → student_head → student_head.logits
```

Role binding: `student_head.logits` → `student.logits`

### 4.3 Interaction graph (sau role binding)

```
teacher.logits ──┬→ kd.standard (temperature=4.0) → interaction.kd.loss_component, kd.teacher_target, kd.student_target
student.logits ──┘

teacher.logits ──┬→ uncertainty.confidence_proxy → teacher.var, student.var
student.logits ──┘
```

Interaction chỉ đọc role outputs — KHÔNG đọc graph outputs trực tiếp.

### 4.4 Decision (sau interaction)

```
teacher.logits + student.logits + student.var → SoftBlendingDecisionModule → logits, gate.alpha
```

Explicit inputs: `[teacher.logits, student.logits, student.var]`

---

## 5. Low-Level Design

### 5.1 Node: `encoder.seq_bilstm`

**Input key**: khai báo trong `inputs` list (drug_seq hoặc protein_seq). Đọc bằng `inputs[self.declared_input_key]`.

**Architecture**:
```
tokens = to_token_ids(inputs[declared_key], vocab_size)  // (B, L) long, clamp [0, vocab_size)
x = embedding(tokens)                                     // (B, L, E)
x = x.transpose(1, 2)                                    // (B, E, L)
x = ReLU(conv1(x))                                       // (B, H, L) — block 1
x = ReLU(conv2(x))                                       // (B, H, L) — block 2
x = x.transpose(1, 2)                                    // (B, L, H)
x, _ = bilstm(x)                                         // (B, L, 2*lstm_dim)
x = x.mean(dim=1)                                        // (B, 2*lstm_dim)
x = projection(x)                                        // (B, proj_dim)
return {"embedding": x}
```

**Preconditions**: input rank == 2, B ≥ 1, L ≥ 1.
**Postconditions**: output shape `(B, proj_dim)`, finite.

**Params**: `vocab_size=64`, `embedding_dim=32`, `hidden_dim=64`, `kernel_size_1=3`, `kernel_size_2=5`, `lstm_dim=64`, `proj_dim=64`.

---

### 5.2 Node: `encoder.gnn_drug`

**Input key**: `drug_graph` (dict với `adj` và `node_feat`). Đọc bằng `inputs["drug_graph"]`.

**Missing modality handling**:
```python
if "drug_graph" not in inputs or inputs["drug_graph"] is None:
    if self.allow_missing_input:
        B = context.batch_size  # hoặc suy ra từ context
        return {"embedding": torch.zeros(B, self.proj_dim, device=...)}
    raise MissingDependencyError(...)
```

**Architecture**:
```
adj = inputs["drug_graph"]["adj"]          // (B, N, N)
x   = inputs["drug_graph"]["node_feat"]    // (B, N, F)

if normalize_adj:
    deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
    adj_norm = adj / deg                   // D^{-1} row-norm

h = linear1(x)                            // (B, N, H)
h = bmm(adj_norm, h)                      // (B, N, H) — GCN block 1
h = ReLU(h)

h = linear2(h)                            // (B, N, H)
h = bmm(adj_norm, h)                      // (B, N, H) — GCN block 2
h = ReLU(h)

h = h.mean(dim=1)                         // (B, H)
h = projection(h)                         // (B, proj_dim)
return {"embedding": h}
```

**Params**: `node_feat_dim=9`, `hidden_dim=64`, `proj_dim=64`, `normalize_adj=True`, `allow_missing_input=False`.

---

### 5.3 Node: `head.dense`

**Input key**: khai báo trong `inputs` list. Đọc bằng `inputs[self.declared_input_key]`.

**Architecture**:
```
x = inputs[declared_key]          // (B, D)
assert x.ndim == 2
x = ReLU(hidden(x))               // (B, hidden_dim)
x = output_layer(x)               // (B, output_dim)
return {"logits": x}              // NO sigmoid
```

**Params**: `hidden_dim=64`, `output_dim=1`, `input_dim=None`.

---

### 5.4 Fusion: `fusion.concat` — N ≥ 2

```python
# Đọc inputs theo thứ tự khai báo, không phụ thuộc dict order
tensors = [inputs[k] for k in self.declared_inputs]
assert len(tensors) >= 2
assert all(t.ndim == 2 for t in tensors), raise InvalidConfigError(...)
B = tensors[0].shape[0]
assert all(t.shape[0] == B for t in tensors), raise InvalidConfigError(...)
fused = torch.cat([t.to(dtype=torch.float32) for t in tensors], dim=1)  // (B, sum(D_i))
return {"embedding": self.projection(fused)}
```

---

### 5.5 KD Interaction — numerical stability

```python
# Softened probabilities với clamp để tránh NaN/Inf
def binary_logits_to_dist(logits, temperature):
    scaled = logits / temperature
    baseline = torch.zeros_like(scaled)
    stacked = torch.stack([baseline, scaled], dim=-1)
    probs = torch.softmax(stacked, dim=-1)
    probs = probs.clamp(1e-6, 1 - 1e-6)  // numerical stability
    return probs

teacher_target = binary_logits_to_dist(teacher_logits, T)
student_target = binary_logits_to_dist(student_logits, T)
kd_loss = F.kl_div(student_target.log(), teacher_target, reduction="batchmean") * T**2
```

Inputs đọc từ state: `teacher.logits`, `student.logits` — role outputs, không phải graph outputs.

---

### 5.6 Fix `_scheduled_kd_weight()` — lambda_max từ training.kd

```python
def _scheduled_kd_weight(cfg, step_idx):
    kd_cfg = cfg.get("training", {}).get("kd", {})
    schedule = str(kd_cfg.get("schedule", "constant")).lower()

    if schedule == "warmup":
        lambda_max = float(kd_cfg.get("lambda_max", 0.5))   # đọc từ training.kd
        warmup_steps = max(1, int(kd_cfg.get("warmup_steps", 100)))
        progress = min(1.0, float(step_idx + 1) / float(warmup_steps))
        return lambda_max * progress

    if schedule == "constant":
        mapping = cfg.get("loss", {}).get("map", {}).get("kd")
        if isinstance(mapping, dict):
            return float(mapping.get("weight", 1.0))
        return None

    raise InvalidConfigError(f"Unsupported KD schedule {schedule!r}.", ...)
```

---

## 6. Scenario Handling (S1–S4)

| Scenario | drug_graph | teacher GNN | student | Pipeline |
|---|---|---|---|---|
| S1 (warm) | ✓ có | normal forward | normal forward | full teacher + student |
| S2 (cold drug) | ✗ không có | zero embedding | normal forward | teacher dùng zero graph emb |
| S3 (cold target) | ✓ có | normal forward | normal forward | full teacher + student |
| S4 (fully cold) | ✗ không có | zero embedding | normal forward | teacher dùng zero graph emb |

**S2/S4 handling**: `t_graph_enc` có `allow_missing_input: true`. Khi `drug_graph` không có trong batch, node trả về `torch.zeros(B, proj_dim)`. Teacher fusion vẫn nhận 3 embeddings (drug_seq_emb + prot_emb + zero_graph_emb).

---

## 7. Config: `configs/flow1_teacher_student.yaml`

```yaml
contract_version: v1
version: v1

experiment:
  name: ugtsdti-flow1

data:
  dataset: davis
  preprocessing_version: v1
  split_version: v1

scenario:
  train: s1
  eval: [s1, s2, s3, s4]

modalities:
  available: [sequence, structure]
  teacher:
    uses: [sequence, structure]
  student:
    uses: [sequence]

graph:
  nodes:
    - name: t_drug_enc
      type: encoder.seq_bilstm
      inputs: [drug_seq]
      output_attrs: [embedding]
      params:
        vocab_size: 64
        embedding_dim: 32
        hidden_dim: 64
        lstm_dim: 64
        proj_dim: 64

    - name: t_prot_enc
      type: encoder.seq_bilstm
      inputs: [protein_seq]
      output_attrs: [embedding]
      params:
        vocab_size: 32
        embedding_dim: 32
        hidden_dim: 64
        lstm_dim: 64
        proj_dim: 64

    - name: t_graph_enc
      type: encoder.gnn_drug
      inputs: [drug_graph]
      output_attrs: [embedding]
      params:
        node_feat_dim: 9
        hidden_dim: 64
        proj_dim: 64
        normalize_adj: true
        allow_missing_input: true   # S2/S4: zero embedding khi không có drug_graph

    - name: teacher_fusion
      type: fusion.concat
      inputs:
        - t_drug_enc.embedding
        - t_prot_enc.embedding
        - t_graph_enc.embedding
      output_attrs: [embedding]
      params:
        project_dim: 128

    - name: teacher_head
      type: head.dense
      inputs: [teacher_fusion.embedding]
      output_attrs: [logits]
      params:
        hidden_dim: 64
        output_dim: 1

    - name: s_drug_enc
      type: encoder.seq_bilstm
      inputs: [drug_seq]
      output_attrs: [embedding]
      params:
        vocab_size: 64
        embedding_dim: 32
        hidden_dim: 64
        lstm_dim: 64
        proj_dim: 64

    - name: s_prot_enc
      type: encoder.seq_bilstm
      inputs: [protein_seq]
      output_attrs: [embedding]
      params:
        vocab_size: 32
        embedding_dim: 32
        hidden_dim: 64
        lstm_dim: 64
        proj_dim: 64

    - name: student_fusion
      type: fusion.concat
      inputs:
        - s_drug_enc.embedding
        - s_prot_enc.embedding
      output_attrs: [embedding]
      params:
        project_dim: 128

    - name: student_head
      type: head.dense
      inputs: [student_fusion.embedding]
      output_attrs: [logits]
      params:
        hidden_dim: 64
        output_dim: 1

roles:
  teacher:
    outputs: [teacher_head.logits]
    aggregation: first
  student:
    outputs: [student_head.logits]
    aggregation: first

interaction:
  order: [kd, uncertainty]
  dependencies: {}
  kd:
    type: kd.standard
    inputs: [teacher.logits, student.logits]   # role outputs, NOT graph outputs
    params:
      enabled: true
      mode: logits
      temperature: 4.0
  uncertainty:
    type: uncertainty.confidence_proxy
    inputs: [teacher.logits, student.logits]   # role outputs
    params:
      enabled: true
      targets:
        teacher: true
        student: true

decision:
  type: gate.uncertainty
  mode: heuristic
  strategy: soft
  use_uncertainty: true
  inputs: [teacher.logits, student.logits, student.var]   # explicit declared inputs
  fallback:
    no_teacher: student
    no_student: teacher
    no_uncertainty: equal_blend   # alpha=0.5 khi không có student.var

training:
  teacher:
    freeze: false
  student:
    freeze: false
  kd:
    schedule: warmup
    warmup_steps: 100
    lambda_max: 0.5
  optimizer:
    type: adam
    lr: 0.001
    weight_decay: 1.0e-4
  scheduler:
    type: plateau
    mode: max
    factor: 0.5
    patience: 3
  loop:
    epochs: 50
    checkpoint_every_epochs: 5
    eval_every_epochs: 1
    eval_partition: test
    max_grad_norm: 5.0
    fail_on_nonfinite_loss: true
    best_metric: metrics.auroc
    best_mode: max
    early_stopping:
      enabled: true
      patience: 10
      min_delta: 0.001

loss:
  type: composite
  hard_weight: 1.0
  map:
    kd:
      from: interaction.kd.loss_component
      weight: 0.5   # overridden by warmup schedule at runtime

metrics:
  primary: auroc
  enabled: [auroc, auprc]
  by_scenario: true

runtime:
  seed: 42
  deterministic: true
  device: cpu
```

---

## 8. Config: `configs/profiles/flow1_local.yaml`

```yaml
extends:
  - ../flow1_teacher_student.yaml

experiment:
  name: ugtsdti-flow1-local

runtime:
  device: cpu
  seed: 42
  deterministic: true

logging:
  backend: file
  trace_execution: true

training:
  loop:
    epochs: 5
    checkpoint_every_epochs: 1
```

---

## 9. Validation Rules

| Case | Error | Stage |
|---|---|---|
| Missing declared input key | `MissingDependencyError` | graph / interaction |
| Duplicate state key | `KeyCollisionError` | state |
| Fusion shape mismatch | `InvalidConfigError` | graph |
| Non-finite decision input | `InvalidDecisionOutputError` | decision |
| Interaction reads undeclared key | `MissingDependencyError` | interaction |
| Node input rank sai | `InvalidConfigError` | graph |
| Unregistered type key | `MissingDependencyError` | graph |
| Duplicate type key registration | `InvalidConfigError` | registry |

---

## 10. Correctness Properties

```
∀ batch S1: teacher_head.logits.shape == (B, 1) ∧ finite
∀ batch S1: student_head.logits.shape == (B, 1) ∧ finite
∀ batch S2/S4: t_graph_enc.embedding == zeros(B, proj_dim)
∀ batch: gate.alpha ∈ [0, 1]
∀ batch: kd_teacher_target.sum(dim=-1) ≈ 1.0 (probability distribution)
∀ step: kd_lambda(step) ∈ [0, lambda_max]
∀ step ≥ warmup_steps: kd_lambda(step) == lambda_max
∀ step (freeze=true): teacher_params_before == teacher_params_after
∀ batch: student pipeline runs without drug_graph
∀ config+seed: output is deterministic
```

---

## 11. Files Tác Động

| File | Thay đổi |
|---|---|
| `ugtsdti/nodes/flow1.py` | Tạo mới: `SeqBiLSTMEncoderRuntime`, `GNNDrugEncoderRuntime`, `DenseHeadRuntime` |
| `ugtsdti/nodes/baseline.py` | Fix `ConcatFusionRuntime`: N≥2, ordered inputs |
| `ugtsdti/runtime/defaults.py` | Đăng ký 3 type keys mới |
| `ugtsdti/trainer/trainer.py` | Fix `_scheduled_kd_weight()` đọc `lambda_max` từ `training.kd` |
| `configs/flow1_teacher_student.yaml` | Tạo mới |
| `configs/profiles/flow1_local.yaml` | Tạo mới |
| `tests/quality/test_flow1_nodes.py` | Tạo mới |
| `tests/integration/test_flow1_pipeline.py` | Tạo mới |

---

## 12. Out of Scope

- Pretrained weights cho teacher
- Drug graph preprocessing pipeline
- Flow 2 (structure-only student)
- Feature-level / relation-level KD
- Learned gate
