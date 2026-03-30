# Tài Liệu Thiết Kế: UGTSDTI Baseline Model

## 1. Mục đích

Tài liệu này chuyển `requirements.md` của spec baseline thành thiết kế triển khai cụ thể cho framework UGTSDTI hiện tại.

Mục tiêu là dựng một baseline model:

- đủ thật để train được;
- đủ tối giản để dễ audit;
- đủ tương thích với pipeline hiện tại để trở thành reference implementation cho các model về sau.

---

## 2. Nguyên Tắc Thiết Kế

### 2.1 Không bypass framework

Mọi tính toán của baseline phải sống trong graph nodes. Trainer chỉ orchestration, không được chứa logic model riêng.

### 2.2 Ưu tiên compatibility với contracts hiện có

Decision canonical của baseline dùng `identity`, interaction dùng `noop`, loss dùng hard loss hiện tại, metrics dùng reporter hiện tại.

### 2.3 Baseline mới phải coexist với placeholder runtimes cũ

Các type keys cũ như `encoder.baseline`, `head.linear` vẫn cần được giữ cho compatibility tests/config hiện hữu. Baseline spec này thêm type keys mới thay vì thay thô toàn bộ surface cũ ngay lập tức.

### 2.4 Chỉ harden core contracts khi baseline thực sự cần

Điểm duy nhất cần harden core surface ngay trong spec này là duplicate registration behavior của `NodeRegistry`, vì baseline type keys mới cần deterministic fail-fast semantics.

---

## 3. Public Surface Mới

### 3.1 Graph node type keys mới

Baseline sẽ thêm 4 type keys mới vào graph registry mặc định:

- `encoder.simple_drug`
- `encoder.cnn_protein`
- `fusion.concat`
- `head.mlp`

### 3.2 Canonical graph config

Canonical baseline graph:

```yaml
graph:
  nodes:
    drug_encoder:
      type: encoder.simple_drug
      inputs: [drug_seq]
      params:
        vocab_size: 64
        embedding_dim: 32
        hidden_dim: 64

    protein_encoder:
      type: encoder.cnn_protein
      inputs: [protein_seq]
      params:
        vocab_size: 32
        embedding_dim: 32
        hidden_dim: 64
        kernel_size: 5
        pooling: mean

    fusion:
      type: fusion.concat
      inputs:
        - drug_encoder.embedding
        - protein_encoder.embedding
      params:
        project_dim: 64

    head:
      type: head.mlp
      inputs: [fusion.embedding]
      params:
        hidden_dim: 64
        output_dim: 1
```

### 3.3 Canonical baseline control path

Do validator/runtime hiện tại, baseline control path canonical sẽ là:

```yaml
roles:
  student:
    outputs: [head.logits]
    aggregation: first

interaction:
  order: [noop]
  dependencies: {}
  noop:
    type: noop
    inputs: [student.logits]
    params:
      enabled: true

decision:
  type: identity
  strategy: identity
  source_key: student.logits

loss:
  type: hard
  hard_weight: 1.0
  map: {}

metrics:
  enabled: [auroc, auprc]
  by_scenario: true
```

Lưu ý:

- `decision.type: none` không phù hợp runtime hiện tại; spec baseline dùng `identity`.
- `interaction` không được bỏ trống; baseline dùng `noop`.

---

## 4. Node Runtime Design

## 4.1 Simple Drug Encoder

### Input contract

- required input: `drug_seq`
- input shape expected: `(B, L)` integer token ids

### Computation

1. token embedding
2. mask-aware hoặc simple mean pooling trên sequence axis
3. linear projection sang `hidden_dim`

### Output

- `embedding`: `(B, D)`

### Notes

- Nếu batch hiện tại chưa có explicit mask, baseline có thể dùng simple mean pooling trên toàn chiều `L`.
- Runtime phải cast input sang integer token ids trước khi vào embedding nếu input đến dưới dạng tensor số.

## 4.2 CNN Protein Encoder

### Input contract

- required input: `protein_seq`
- input shape expected: `(B, L)` integer token ids

### Computation

1. token embedding
2. transpose sang `(B, C, L)` cho `Conv1d`
3. `Conv1d`
4. `ReLU`
5. global pooling (`mean` hoặc `max`)
6. optional projection nếu cần normalize output dim

### Output

- `embedding`: `(B, D)`

## 4.3 Concat Fusion

### Input contract

- two embeddings `(B, D1)` và `(B, D2)`

### Computation

1. validate same batch dimension
2. concatenate theo feature axis
3. nếu `project_dim` có set, apply linear projection sang `project_dim`

### Output

- `embedding`

## 4.4 MLP Head

### Input contract

- required input: fusion embedding `(B, D)`

### Computation

1. `Linear(D -> H)`
2. `ReLU`
3. `Linear(H -> output_dim)`

### Output

- `logits`

### Notes

- Không sigmoid.
- `output_dim` canonical là `1`.

---

## 5. Runtime Integration

## 5.1 Node runtime statefulness

Các baseline runtimes đều là trainable graph runtimes, nên phải expose:

- `parameters()`
- `state_dict()`
- `load_state_dict()`

Điều này làm baseline tương thích với:

- `PipelineExecutor.parameter_groups()`
- `PipelineExecutor.model_state()`
- `PipelineExecutor.load_model_state()`
- `Trainer.step()` optimizer/checkpoint/artifact flow

## 5.2 Registry integration

### Current gap

`NodeRegistry.register()` hiện overwrite silently nếu `type_key` trùng.

### Required design delta

Sửa `NodeRegistry.register()` thành fail-fast:

- nếu `type_key` chưa có -> register
- nếu `type_key` đã có -> raise explicit error

Điều này không đổi stage contracts, nhưng làm plugin surface an toàn hơn và đáp ứng baseline requirement về deterministic registration.

## 5.3 Default graph registry

`build_default_graph_registry()` sẽ:

- giữ nguyên các type keys cũ đang có;
- thêm 4 baseline type keys mới.

Không nên xóa runtime placeholders cũ trong spec này để tránh làm vỡ tests/config cũ không liên quan.

---

## 6. Config Design

## 6.1 Canonical baseline config file

Repo nên ship ít nhất một config mới, ví dụ:

- `configs/baseline_reference.yaml`

Config này phải có đủ sections validator yêu cầu:

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
- `metrics`

## 6.2 Modalities section

Vì baseline student-only dùng sequence inputs:

```yaml
modalities:
  available: [sequence]
  student:
    uses: [sequence]
```

## 6.3 Data assumptions

Framework hiện có thể load batch fields từ processed artifacts. Baseline spec không thêm preprocessing mới, nên canonical baseline cần giả định:

- `drug_seq`
- `protein_seq`
- `labels`
- `scenario`

đã được materialize bởi data layer hiện tại hoặc bởi test fixtures.

---

## 7. Validation Strategy

## 7.1 Unit tests

Mỗi runtime node cần unit tests cho:

- output key names đúng
- output shapes đúng
- wrong input rank hoặc batch mismatch fail rõ
- state_dict/load_state_dict round-trip

## 7.2 Integration tests

### Baseline pipeline test

Một integration test tối thiểu phải:

1. build graph plan từ config baseline
2. run `PipelineExecutor.run_batch()`
3. assert state có:
   - `head.logits`
   - `student.logits`
   - `logits`
   - `loss.total`
   - `metrics.auroc`

### Baseline training test

Một training integration test phải:

1. dựng fixture dataset nhỏ có quan hệ learnable
2. chạy nhiều `Trainer.step()` với optimizer thật
3. assert loss cuối thấp hơn loss đầu hoặc moving average giảm rõ

### Baseline checkpoint restore test

Một checkpoint/model-state test phải:

1. train vài steps
2. collect `model_state`
3. tạo executor mới
4. restore state
5. confirm forward outputs khớp hoặc rất gần trên cùng batch

## 7.3 Optional CLI smoke

Nếu có thể reuse data fixtures trong repo, nên thêm một integration smoke cho CLI train/eval baseline config. Nhưng đây không phải blocker của spec baseline nếu tests framework-level đã chứng minh path train/eval tương đương.

---

## 8. Metrics and Learning Signal

## 8.1 Loss

Baseline dùng hard loss hiện tại của framework:

- BCE-with-logits cho torch tensors

Không cần custom loss mới.

## 8.2 Metrics

Canonical metrics cho baseline:

- `auroc`
- `auprc`

`f1` có thể bật thêm trong tests cục bộ nếu cần, nhưng baseline reference nên bám hai metric ranking chính của DTI binary classification.

## 8.3 Learnability fixture

Để test loss giảm ổn định, fixture nên có pattern đơn giản và learnable, ví dụ:

- label phụ thuộc rõ vào tương tác giữa token statistics của drug/protein sequences
- cùng batch size nhỏ nhưng signal sạch

Mục tiêu là kiểm framework + optimizer + node gradients, không phải benchmark khoa học.

---

## 9. Files Tác Động Dự Kiến

Core implementation:

- `ugtsdti/runtime/defaults.py`
- `ugtsdti/graph/registry.py`

Tests:

- `tests/quality/test_graph.py` hoặc file node-focused mới
- `tests/integration/test_pipeline.py`
- `tests/integration/test_orchestration.py`

Config:

- `configs/baseline_reference.yaml`

Collateral:

- `.docs/traceability.md`

---

## 10. Out of Scope Cho Spec Này

- teacher graph
- dual-branch baseline
- drug graph encoder thật
- pretrained protein LM
- uncertainty-aware decision
- benchmark quality claims trên dataset thật

Spec này chỉ nhằm đưa framework từ trạng thái “có placeholder runtimes” sang trạng thái “có baseline model reference chạy thật”.
