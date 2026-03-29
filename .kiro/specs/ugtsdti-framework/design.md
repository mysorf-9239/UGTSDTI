# Tài Liệu Thiết Kế: UGTSDTI Framework

## 1. Mục đích

Tài liệu này chuyển hóa [requirements.md](requirements.md) thành thiết kế triển khai cho `UGTSDTI v1`.

Mục tiêu của tài liệu:

- khóa kiến trúc triển khai bám sát requirements đã được làm sạch;
- chỉ định decomposition hợp lý để code được mà không phá stage boundaries;
- cung cấp data models, validation boundaries, và runtime flow đủ rõ cho implementation;
- loại bỏ các mâu thuẫn cũ, đặc biệt ở decision layer, batch contract, và minimal baseline.

Nếu có mâu thuẫn giữa tài liệu này và [requirements.md](requirements.md), **requirements.md thắng**.

---

## 2. Phạm vi

Thiết kế này áp dụng cho `v1` của framework với các capability chính:

- pipeline cố định `Batch -> Graph -> Role Binding -> Interaction -> Decision -> Loss + Metrics`;
- graph stage dùng node plugins và DAG planning;
- teacher-student asymmetry là khái niệm ngữ nghĩa, không phải graph primitive;
- interaction layer chứa KD, uncertainty, disagreement;
- decision layer sinh final `logits`;
- runtime tách khỏi data acquisition;
- hỗ trợ reproducibility, validation, artifacts, CLI, và offline logging.

Thiết kế này **không** cố gắng giải quyết triệt để:

- nhiều teachers hoặc nhiều students trong một run;
- hard guarantee reproducibility giữa khác platform/PyTorch/CUDA release;
- online data acquisition trong train/eval runtime;
- full N-branch trust allocation beyond V1.

---

## 3. Design Drivers

Thiết kế này bị chi phối bởi các requirement sau:

- fixed pipeline invariant;
- explicit stage contracts;
- single State medium với write-once và single-producer;
- config-driven assembly;
- modality-aware teacher-student asymmetry;
- KD là interaction, uncertainty là interaction, decision là stage sinh final logits;
- deterministic planning, dry-run validation, và reproducibility bundle;
- baseline tối thiểu vẫn giữ đủ stage boundaries.

---

## 4. Kiến Trúc Tổng Thể

### 4.1 Pipeline

Pipeline runtime chuẩn:

```text
Batch
-> Graph
-> Role Binding
-> Interaction
-> Decision
-> Loss + Metrics
```

### 4.2 Ownership theo stage

| Stage | Input chính | Output chính | Không được làm |
| --- | --- | --- | --- |
| Batch | materialized batch payload | initial State keys | gọi external acquisition service |
| Graph | batch keys, graph outputs đã khai báo | `<node>.<attr>` | đọc role/interaction/decision outputs |
| Role Binding | graph outputs | `<role>.logits` | sửa graph outputs |
| Interaction | role outputs, prior interaction outputs | `teacher.var`, `student.var`, `interaction.*`, `kd.*` | sinh final `logits` |
| Decision | role outputs, interaction outputs | `logits`, `gate.alpha`, `gate.*` | sửa stage trước |
| Loss + Metrics | final outputs, labels, scenario | `loss.*`, `metrics.*`, `diagnostics.*` | giả định hidden interaction behavior |

### 4.3 Runtime modes

Ba runtime modes được hỗ trợ:

- `train`
- `eval`
- `infer`

Graph semantics không đổi giữa ba modes; chỉ:

- postprocess behavior;
- metrics computation;
- checkpoint/logging hooks

mới khác.

---

## 5. Package Layout

Package structure mục tiêu:

```text
ugtsdti/
  core/
    state.py
    context.py
    errors.py
    schema.py

  config/
    loader.py
    validate.py
    normalize.py
    models.py

  graph/
    specs.py
    registry.py
    builder.py
    planner.py
    engine.py

  nodes/
    base.py
    encoder/
    fusion/
    head/

  roles/
    binder.py

  interaction/
    base.py
    registry.py
    kd.py
    uncertainty.py
    diagnostics.py

  decision/
    base.py
    trust.py
    policy.py
    module.py

  postprocess/
    loss.py
    metrics.py

  data/
    contracts.py
    acquisition.py
    preprocessing.py
    splitting.py
    loader.py
    validate.py

  runtime/
    seed.py
    adapter.py
    identity.py
    checkpoint.py

  logging/
    base.py
    file_logger.py
    wandb_logger.py
    composite.py

  trainer/
    trainer.py
    evaluator.py

  cli/
    main.py
```

Nguyên tắc:

- `requirements.md` không khóa file layout.
- `design.md` chọn layout này vì nó giữ stage separation rõ và đủ cho traceability.

---

## 6. Core Data Models

### 6.1 State

`State` là medium giao tiếp duy nhất giữa các stage.

Internal rules:

- key-value store có write-once semantics;
- commit chỉ được thực hiện bởi stage executor;
- read access phải đi qua declared inputs hoặc stage-allowed surfaces;
- state không bị mutate trực tiếp bởi node/plugin/module.

Suggested public surface:

```python
class State:
    def get(self, key: str) -> Any: ...
    def has(self, key: str) -> bool: ...
    def keys(self) -> list[str]: ...
    def snapshot(self) -> dict[str, Any]: ...
```

Suggested internal executor-only surface:

```python
class StateWriter:
    def commit(self, producer: str, outputs: dict[str, Any]) -> None: ...
```

Lý do tách `State` và `StateWriter`:

- enforce write-once;
- tránh plugin mutate State trực tiếp;
- thuận tiện cho debug, event hooks, và schema validation.

### 6.2 ExecutionContext

```python
@dataclass
class ExecutionContext:
    mode: Literal["train", "eval", "infer"]
    seed: int
    device: str
    deterministic: bool
    precision: Literal["fp32", "mixed"] = "fp32"
```

`ExecutionContext` không chứa config đầy đủ; nó chỉ chứa runtime controls cần ở hot path.

### 6.3 Error Taxonomy

Base errors:

- `MissingDependencyError`
- `KeyCollisionError`
- `InvalidConfigError`
- `InvalidRoleBindingError`
- `InvalidInteractionGraphError`
- `InvalidDecisionOutputError`
- `InvalidAlphaRangeError`
- `InvalidLossMappingError`
- `BatchSchemaError`
- `InvalidStateSchemaError`
- `NumericalInstabilityError`
- `DeviceInconsistencyError`
- `CheckpointCorruptedError`
- `MissingRawSnapshotError`
- `ProcessedSplitMismatchError`
- `InconsistentSplitError`

Mỗi error nên chứa:

- stage
- component
- offending key/config field
- short human-readable message
- optional debug payload

### 6.4 Stage-Aware Schema

Thiết kế mới **không** dùng một canonical state schema cứng cho mọi run. Thay vào đó:

- có một **core schema fragment** cho keys luôn tồn tại;
- phần còn lại được **build từ config và plugin specs**.

Điều này giải quyết hai vấn đề của design cũ:

- batch contract không còn ép buộc modality thừa;
- KD outputs không còn bị chốt shape sai cho mọi mode.

Suggested model:

```python
@dataclass
class StateSpec:
    key: str
    stage: str
    required: bool
    shape: tuple[Any, ...] | None
    dtype: str | None
    semantic: str
```

```python
class StateSchemaBuilder:
    def build_for_experiment(self, cfg: "NormalizedConfig") -> list[StateSpec]: ...
```

Core required keys:

- `labels`
- `scenario`
- graph outputs declared by selected nodes
- canonical role outputs
- decision `logits`

Conditionally required keys:

- modality keys required bởi selected graph nodes
- teacher branch outputs
- KD outputs
- uncertainty outputs
- `gate.alpha`
- diagnostics keys

### 6.5 BatchContract

`BatchContract` mới là **conditional contract**, không phải static dataclass bắt buộc mọi key.

Model:

```python
@dataclass
class BatchSpec:
    required_common: list[str]
    conditional_keys: dict[str, str]
```

For V1:

- always required:
  - `labels`
  - `scenario`
- conditionally required:
  - `drug_seq`
  - `protein_seq`
  - `drug_graph`
  - `drug_id`
  - `protein_id`

Điều kiện để một key trở thành required:

- có node/validator/materializer trong experiment cần key đó.

Điều này cho phép hỗ trợ:

- reduced-modality student;
- modality-specific student;
- teacher richer modality than student.

### 6.6 Dataset Manifests

Core manifests:

```python
@dataclass
class DatasetVersion:
    dataset_version: str
    preprocessing_version: str
    split_version: str
```

```python
@dataclass
class SplitManifest:
    dataset: str
    dataset_version: str
    preprocessing_version: str
    split_version: str
    seed: int
    scenarios: list[str]
    paths: dict[str, str]
```

### 6.7 Experiment Identity

```python
@dataclass
class ExperimentIdentity:
    run_id: str
    config_hash: str
    git_commit: str | None
    timestamp: str
```

Full reproducibility tuple:

```text
(config_hash, dataset_version, preprocessing_version, split_version, seed)
```

---

## 7. Config System Design

### 7.1 Processing pipeline

Config processing flow:

```text
load YAML
-> resolve extends
-> validate required sections
-> validate schema
-> validate cross-sections
-> normalize
-> produce NormalizedConfig
```

### 7.2 Normalized config model

Normalized config là internal representation mà runtime dùng.

Suggested shape:

```python
@dataclass
class NormalizedConfig:
    version: str
    experiment: dict
    data: dict
    scenario: dict
    modalities: dict
    graph: dict
    roles: dict
    interaction: dict
    decision: dict
    training: dict
    loss: dict
    metrics: dict
    diagnostics: dict
    logging: dict
    runtime: dict
```

Normalized config phải:

- deterministic;
- idempotent;
- serializable;
- stable enough để hash.

### 7.3 Cross-section validation

Validator chạy các checks sau:

- role bindings chỉ vào valid graph outputs;
- interaction dependencies acyclic;
- decision prerequisites tương thích với selected interaction outputs;
- loss map chỉ vào valid produced keys;
- modality compatibility đúng theo selected nodes;
- teacher/student availability tương thích với KD/decision config;
- baseline no-op path hợp lệ khi teacher hoặc KD bị disable.

### 7.4 Plugin-aware validation

Để dry-run build khả thi, config validation dựa vào **plugin specs**, không load full runtime modules.

Mỗi plugin đăng ký:

- type key;
- supported input kinds;
- output attrs hoặc output schema;
- capability metadata;
- optional config validator.

---

## 8. Graph Subsystem

### 8.1 Design goals

Graph subsystem phải thỏa:

- explicit dependencies;
- deterministic planning;
- dry-run validation không load weights;
- runtime execution không có hidden reads;
- output naming ổn định.

### 8.2 Static vs runtime model

Thiết kế mới tách rõ hai lớp:

1. **Static graph model** cho validation/planning
2. **Runtime node instances** cho execution

Static layer:

```python
@dataclass
class NodePluginSpec:
    type_key: str
    output_attrs: list[str]
    capabilities: dict[str, Any]
    input_kinds: list[str]
```

```python
@dataclass
class NodeDefinition:
    name: str
    type_key: str
    inputs: list[str]
    params: dict[str, Any]
```

Runtime layer:

```python
class NodeRuntime(ABC):
    def forward(self, inputs: dict[str, Any], context: ExecutionContext) -> dict[str, Any]: ...
```

Output key resolution:

```text
<node_name>.<output_attr>
```

Ví dụ:

- `student_encoder.embedding`
- `teacher_head.logits`

### 8.3 Registry and factory

```python
class NodeRegistry:
    def register(self, spec: NodePluginSpec, runtime_cls: type["NodeRuntime"]) -> None: ...
    def get_spec(self, type_key: str) -> NodePluginSpec: ...
    def build_runtime(self, definition: NodeDefinition) -> "NodeRuntime": ...
```

### 8.4 GraphBuilder

`GraphBuilder` làm việc trên `NodeDefinition + NodePluginSpec`.

Responsibilities:

- parse `graph.nodes`;
- resolve output keys từ plugin specs;
- build producer map;
- validate dependencies;
- build adjacency and in-degree tables.

Outputs:

```python
@dataclass
class GraphPlan:
    node_definitions: list[NodeDefinition]
    order: list[str]
    produced_keys: dict[str, list[str]]
    producers: dict[str, str]
    edges: dict[str, list[str]]
```

### 8.5 Dry-run validation

Dry-run build không instantiate weights hoặc external resources.

Nó chỉ cần:

- config;
- plugin specs;
- batch key availability assumptions.

Điều này giải quyết mâu thuẫn cũ nơi `declared_outputs` bị dùng trước khi có runtime instance.

### 8.6 GraphEngine

Runtime graph loop:

```python
for node_name in plan.order:
    definition = plan.definition_map[node_name]
    runtime = node_factory.get_or_build(definition)
    inputs = state_reader.materialize_inputs(definition.inputs)
    outputs = runtime.forward(inputs, context)
    writer.commit(node_name, qualify_outputs(node_name, outputs))
```

Validation hooks:

- input presence;
- undeclared output attrs;
- output key collision;
- state schema fragment;
- device/dtype checks khi strict mode bật.

### 8.7 Missing modality handling

Node-level missing modality policy **không** được quyết định ngầm trong runtime.

Chỉ ba con đường hợp lệ:

1. config không chọn node cần modality đó
2. node spec khai báo fallback rõ ràng
3. batch contract và validation fail sớm

---

## 9. Role Binding

### 9.1 Purpose

Role binding biến graph outputs thành semantic branch surfaces.

Input:

- graph outputs

Output:

- `student.logits`
- `teacher.logits` nếu configured

### 9.2 Binding model

```python
@dataclass
class RoleBinding:
    role: str
    outputs: list[str]
    aggregation: Literal["first", "mean"]
```

V1 rules:

- `student` là role bắt buộc cho canonical pipeline;
- `teacher` là optional;
- mỗi role phải expose đúng một canonical logits tensor sau aggregation.

### 9.3 Validation

RoleBinder phải validate:

- selected graph outputs tồn tại;
- output count phù hợp aggregation mode;
- tensors shape-compatible để merge;
- resulting role logits schema hợp lệ.

### 9.4 Optional future support

Thiết kế cho phép sau này thêm:

- nhiều producer heads cho một role;
- nhiều teachers/students trước aggregation.

Nhưng V1 runtime chỉ cần guarantee `teacher` và `student`.

---

## 10. Interaction Subsystem

### 10.1 Interaction model

Interaction stage là ordered acyclic graph của modules.

Static model:

```python
@dataclass
class InteractionPluginSpec:
    type_key: str
    output_keys_fn: Callable[[dict[str, Any]], list[str]]
```

```python
@dataclass
class InteractionDefinition:
    name: str
    type_key: str
    inputs: list[str]
    params: dict[str, Any]
    dependencies: list[str]
```

Runtime model:

```python
class InteractionRuntime(ABC):
    def forward(self, inputs: dict[str, Any], context: ExecutionContext) -> dict[str, Any]: ...
```

### 10.2 Interaction planning

`InteractionPlanner` validates:

- explicit order
- dependency completeness
- no cycles
- single producer per output key
- all declared inputs resolvable from:
  - role outputs
  - prior interaction outputs

### 10.3 KD interaction

#### Purpose

KD là directed branch interaction `teacher -> student`.

#### V1 supported modes

- `logits`
- `feature`
- `relation`

#### Logits KD design

V1 chooses một design nhất quán cho binary DTI:

- role logits vẫn là scalar `(B, 1)`;
- logits KD chuyển scalar logits thành **2-class distillation distributions**;
- `kd.teacher_target` và `kd.student_target` có shape `(B, 2)` trong logits mode;
- `interaction.kd.loss_component` là scalar tensor.

Why:

- tương thích với KL-based distillation;
- tránh mơ hồ giữa `sigmoid` scalar và `softmax` categorical view;
- tách rõ logits surface và KD target surface.

Suggested helper:

```python
def binary_logits_to_dist(logits: Tensor, temperature: float) -> Tensor:
    # logits: (B, 1)
    # return: (B, 2)
```

#### Feature and relation KD

Feature/relation mode có thể emit thêm:

- `interaction.kd.feature_loss_component`
- `interaction.kd.relation_loss_component`
- alignment diagnostics

Nhưng final loss mapping vẫn phải explicit qua config.

#### Scenario-aware KD

Nếu `scenario.policy.affect_kd = true`, KD module được phép điều chỉnh:

- activation
- weight
- schedule multiplier

Nhưng không được thay graph topology hoặc role semantics.

### 10.4 Uncertainty interaction

V1 uncertainty design:

- branch-specific;
- typically MC Dropout based;
- teacher-only, student-only, hoặc both.

Outputs:

- `teacher.var`
- `student.var`

Optional diagnostics:

- `interaction.disagreement`
- calibration helper stats

### 10.5 Diagnostics interactions

Disagreement và research diagnostics khác có thể là:

- interaction module riêng;
- hoặc optional outputs từ KD/uncertainty modules.

Quy tắc:

- diagnostic ownership phải explicit;
- output keys phải stable;
- producer map phải không mơ hồ.

---

## 11. Decision Subsystem

### 11.1 External contract

Decision stage có một external contract duy nhất:

```python
class DecisionModule(ABC):
    def forward(self, state: State, context: ExecutionContext) -> dict[str, Any]:
        # returns:
        # {
        #   "logits": Tensor,
        #   "gate.alpha": Tensor | optional,
        #   "gate.*": ...
        # }
```

Chỉ `DecisionModule` được runtime gọi ở decision stage.

Điểm này giải quyết mâu thuẫn cũ giữa:

- `Gate` tự sinh final logits
- `Gate` chỉ sinh trust signals còn policy sinh logits

### 11.2 Internal decomposition

V1 implementation **chọn** decomposition nội bộ sau:

```text
DecisionModule
  = TrustEstimator (optional internal component)
  + DecisionPolicy (optional internal component)
```

Nhưng đây là internal design, không phải external contract của stage.

#### TrustEstimator

Tính trust signals như:

- `alpha`
- raw gate features
- uncertainty-aware scores

#### DecisionPolicy

Biến trust signals + branch logits thành final `logits`.

### 11.3 Concrete V1 decision modules

V1 cần tối thiểu:

1. `IdentityDecisionModule`
2. `SoftBlendingDecisionModule`
3. `HardSelectionDecisionModule`

#### IdentityDecisionModule

Use cases:

- student-only baseline
- explicit fallback path

Behavior:

- copy một branch canonical logits sang final `logits`;
- không emit `gate.alpha` trừ khi config yêu cầu explicit constant alpha.

#### SoftBlendingDecisionModule

Canonical two-branch formula:

```text
y = alpha * y_teacher + (1 - alpha) * y_student
```

Constraints:

- if `gate.alpha` emitted, it must be finite and within `[0, 1]`.

#### HardSelectionDecisionModule

Behavior:

- chọn teacher hoặc student theo threshold/rule;
- có thể emit alpha dạng `0/1` để giữ diagnostics consistent.

### 11.4 Fallback resolution

Fallback order:

1. if both branches available and preferred signals available -> normal path
2. if one branch missing -> single-branch decision
3. if uncertainty missing but strategy requires it -> configured fallback
4. if policy cannot decide -> explicit failure hoặc configured static fallback

Fallback policy phải cấu hình được và trace được.

### 11.5 Decision validation

DecisionModule phải validate:

- required input keys tồn tại;
- branch logits shape-compatible nếu cần blend;
- alpha finite và range-safe;
- final logits finite.

---

## 12. Loss and Metrics

### 12.1 LossComposer

LossComposer nhận:

- normalized loss config
- State sau decision
- labels

Outputs:

- `loss.total`
- `loss.hard`
- `loss.kd` nếu applicable
- optional named aux losses

Hard rule:

- chỉ compose những interaction outputs được map explicit trong config.

Example:

```yaml
loss:
  type: composite
  hard_weight: 0.7
  map:
    kd:
      from: interaction.kd.loss_component
      weight: 0.3
```

### 12.2 MetricsReporter

MetricsReporter làm hai việc:

1. compute per-batch/per-split metrics
2. aggregate per-scenario reports

Core metrics:

- AUROC
- AUPRC
- F1

Important rule:

- nếu metric cần probability threshold như F1, reporter phải convert từ logits sang probabilities;
- không threshold trực tiếp trên raw logits như implementation cũ đã từng làm.

### 12.3 Diagnostics outputs

Diagnostics được tách khỏi training loss.

Suggested outputs:

- `diagnostics.disagreement`
- `diagnostics.gate_alpha`
- `diagnostics.uncertainty_error`

Diagnostics có thể được materialize:

- ở eval time
- hoặc ở train time theo config

---

## 13. Data Lifecycle

### 13.1 Principle

Runtime không được phụ thuộc vào online acquisition.

Pipeline dữ liệu:

```text
acquisition -> raw snapshot -> preprocessing -> processed tensors -> splitting -> runtime loading
```

### 13.2 Acquisition

`data/acquisition.py` có thể dùng pyTDC để download hoặc export raw dataset.

Outputs:

- `data/raw/<dataset>.csv`

### 13.3 Preprocessing

`data/preprocessing.py` chuyển raw data thành materialized features phù hợp experiment families:

- tokenized sequences
- graphs
- ids
- labels

Outputs:

- `data/processed/<dataset>/<preprocessing_version>/...`

### 13.4 Splitting

`data/splitting.py` sinh deterministic splits cho:

- `S1`
- `S2`
- `S3`
- `S4`

Outputs:

- split files
- `SplitManifest`

### 13.5 Runtime loading

`data/loader.py` chỉ đọc materialized data.

Suggested abstraction:

```python
class DataLoaderFactory:
    def build(self, cfg: NormalizedConfig, manifest: SplitManifest) -> dict[str, Any]:
        # train_loader, eval_loaders, batch_spec
```

Loader factory đồng thời build:

- dataloaders
- `BatchSpec`
- dataset version metadata

### 13.6 Data validation

`DataValidator` chạy:

- artifact presence check
- split/version consistency check
- batch contract validation
- optional sanity checks

---

## 14. Trainer and Evaluator

### 14.1 Trainer orchestration

`Trainer.step(batch)` thực hiện:

```text
init State from batch
-> validate batch contract
-> graph
-> role binding
-> interaction
-> decision
-> loss
-> backward/update (train mode)
```

Trainer còn chịu trách nhiệm:

- freeze policy;
- KD schedule;
- precision/autocast policy;
- checkpoint hooks;
- logger hooks.

### 14.2 Evaluator orchestration

Evaluator dùng cùng pipeline core nhưng:

- không backward;
- compute metrics/diagnostics;
- aggregate theo scenario.

Scenario reporting contract:

- per-scenario metrics cho configured scenarios;
- aggregated summary;
- structured diagnostics artifact.

### 14.3 Shared pipeline executor

Để tránh divergence giữa train và eval, nên có shared executor:

```python
class PipelineExecutor:
    def run_until_decision(self, batch, cfg, context) -> State: ...
```

Trainer và Evaluator chỉ khác ở postprocess và optimizer logic.

---

## 15. Runtime, Logging, CLI, and Artifacts

### 15.1 RuntimeAdapter

`RuntimeAdapter` chỉ được phép điều chỉnh operational knobs:

- dataloader kwargs
- artifact/checkpoint dirs
- debug defaults
- offline logging defaults

Nó không được thay đổi:

- graph semantics
- role semantics
- interaction semantics
- loss semantics

### 15.2 Logging abstraction

Logger interface:

```python
class Logger(ABC):
    def log_metrics(self, metrics: dict[str, Any], step: int) -> None: ...
    def log_config(self, config: dict[str, Any]) -> None: ...
    def log_artifacts(self, artifact_dir: str) -> None: ...
    def finish(self) -> None: ...
```

Concrete backends:

- `FileLogger`
- `WandbLogger`
- `CompositeLogger`

Policy:

- core pipeline không import `wandb` trực tiếp;
- fallback sang offline/file logging theo config policy.

### 15.3 CLI

CLI commands:

- `train`
- `eval`
- `validate`
- `sweep`

Dispatch path:

```text
load -> validate -> normalize -> build runtime -> execute command
```

`validate` phải:

- không init heavy model weights;
- không call external acquisition;
- report dry-run failures rõ ràng.

### 15.4 Artifacts

Artifact directory tối thiểu:

```text
artifacts/<run_id>/
  config.yaml
  identity.json
  metrics.json
  diagnostics.json
  split_manifest.json
  model.pt
  logs/
```

Nếu tracing bật:

- lưu execution trace;
- lưu state boundary summaries.

---

## 16. Validation Boundary Matrix

| Boundary | Validator | Checks |
| --- | --- | --- |
| Config load/normalize | ConfigValidator | required sections, schema, cross-section rules |
| Before data loading | DataValidator | manifests, version consistency, artifact presence |
| Batch init | BatchValidator | required conditional keys, NaN/Inf, shapes |
| After graph | StateSchemaValidator | produced graph keys, device/dtype consistency |
| After role binding | RoleBindingValidator | canonical logits presence and compatibility |
| After interaction | InteractionValidator | expected outputs, producer map integrity |
| After decision | DecisionValidator | final logits, alpha bounds, finite outputs |
| Before loss | LossMapValidator | explicit loss map resolvability |
| Before checkpoint resume | CheckpointValidator | integrity, config hash compatibility |

---

## 17. Determinism, Precision, and Concurrency

### 17.1 Determinism

Framework dùng deterministic controls ở mức tốt nhất có thể:

- Python seed
- NumPy seed
- torch seed
- deterministic algorithm toggle
- worker-specific seeding

Design note:

- theo PyTorch docs, reproducibility tuyệt đối không được bảo đảm giữa khác release/platform;
- framework phải log rõ assumption này.

### 17.2 Precision

Supported precision modes:

- `fp32`
- `mixed`

Rules:

- precision mode phải explicit trong runtime config;
- mixed precision không được bypass numerical validation của final logits/loss.

### 17.3 Concurrency

State mutability rule:

- node/plugin không mutate State;
- stage executor là nơi duy nhất commit outputs.

Multi-worker rule:

- dataloader workers không chạm shared mutable runtime state.

Multi-process rule:

- mỗi process có State riêng;
- metric aggregation xảy ra sau compute, không xảy ra trong forward pass.

---

## 18. Checkpoint and Resume

Checkpoint bundle cần chứa:

- model state
- optimizer state
- scheduler state nếu có
- RNG state
- epoch/step
- normalized config snapshot
- experiment identity
- dataset/split metadata

Suggested model:

```python
@dataclass
class CheckpointBundle:
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any] | None
    rng_state: dict[str, Any]
    epoch: int
    step: int
    identity: ExperimentIdentity
    config: dict[str, Any]
    dataset_version: DatasetVersion
```

Write policy:

- atomic write hoặc temp-then-rename.

Resume policy:

- validate bundle completeness;
- validate config compatibility;
- restore RNG trước khi resume loop.

---

## 19. Minimal Baseline and Growth Path

### 19.1 Minimal baseline

V1 minimal runnable baseline:

- student encoder
- student head
- role binding -> `student.logits`
- no-op interaction
- identity decision module
- hard loss
- S1 evaluation

Expected State growth:

```text
labels, scenario, modality keys
-> student_encoder.embedding
-> student_head.logits
-> student.logits
-> logits
-> loss.total
```

Điểm quan trọng:

- decision stage vẫn tồn tại;
- final logits vẫn sinh ở decision stage;
- minimal baseline không phá fixed pipeline invariant.

### 19.2 Growth path

Từ baseline tối thiểu có thể mở rộng tuần tự:

1. add teacher branch
2. add KD interaction
3. add uncertainty interaction
4. add soft/hard decision modules
5. add scenario-aware policies
6. add reproducibility bundle and sweep workflows

---

## 20. Key Design Resolutions So Với Bản Cũ

Thiết kế mới chốt rõ các điểm từng mâu thuẫn:

### 20.1 Decision contract

- external contract duy nhất là `DecisionModule.forward(...) -> logits + optional gate outputs`
- internal trust/policy decomposition là optional implementation choice

### 20.2 Batch contract

- batch keys là conditional theo experiment, không ép cứng mọi modality keys

### 20.3 Dry-run graph validation

- dry-run dùng `NodePluginSpec`, không cần runtime node instances

### 20.4 KD target semantics

- binary logits KD trong V1 dùng 2-class distillation distributions `(B, 2)` ở logits mode
- không còn schema mơ hồ `(B,1)` vs `(B,C)` cho cùng một concept

### 20.5 Minimal baseline

- giữ đủ stage boundaries bằng interaction no-op + identity decision

---

## 21. Tác Động Tới Implementation

Implementation nên đi theo thứ tự:

1. `core/` + error taxonomy + state writer model
2. `config/` + normalized config + plugin specs
3. `graph/` static planning + dry-run
4. `roles/`
5. `interaction/`
6. `decision/`
7. `postprocess/`
8. `data/`
9. `trainer/`, `runtime/`, `logging/`, `cli/`

Điểm cần tránh khi implement:

- node runtime tự đọc State trực tiếp thay vì declared inputs;
- runtime import `wandb` trong hot path core;
- data loader gọi pyTDC;
- final logits sinh trước decision stage;
- batch contract validate cứng mọi modality keys cho mọi experiment.

---

## 22. Tài Liệu Cần Sync Tiếp

Sau khi design này được chốt, các tài liệu sau cần sync lại:

- `traceability.md`
- implementation plan/checkpoints nếu còn dùng assumptions cũ
- configs mẫu

Đặc biệt:

- traceability phải map theo REQ IDs mới;
- test plan phải reflect identity decision module ở minimal baseline;
- state schema tests phải chuyển sang schema-derived validation thay vì canonical hard-coded schema toàn cục.
