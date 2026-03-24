# Requirements Document

## Introduction

Phase 11 của dự án UGTSDTI thay thế `baseline_teacher` (dummy `nn.Embedding`) bằng một Teacher GNN thực sự dựa trên đồ thị tương đồng Drug-Drug (DD) và Protein-Protein (PP). Teacher GNN sử dụng Morgan fingerprint (RDKit) cho drug nodes và k-mer frequency vector cho protein nodes, sau đó chạy GCN/GAT để tạo embedding có ý nghĩa. Kết quả là PairGate và KD loss mới có thể hoạt động đúng: Teacher confident ở S1 (warm-start), uncertain ở S4 (cold-start), cho phép gate α học được hành vi thích ứng.

## Glossary

- **Graph_Builder**: Module `ugtsdti/data/graph_builder.py` chịu trách nhiệm xây dựng và cache DD/PP graphs.
- **DD_Graph**: Drug-Drug similarity graph, nodes là drug molecules, edges dựa trên Tanimoto similarity của Morgan fingerprints.
- **PP_Graph**: Protein-Protein similarity graph, nodes là protein sequences, edges dựa trên cosine similarity của k-mer frequency vectors.
- **GCNTeacher**: Model GNN Teacher được đăng ký với `@MODELS.register("gcn_teacher")`, nhận `batch["drug_index"]` và `batch["target_index"]` để lookup trong global graph.
- **Morgan_Fingerprint**: Bit-vector biểu diễn cấu trúc phân tử, radius=2 (ECFP4), nBits=2048, tính bằng RDKit.
- **KMer_Vector**: Frequency vector của k-mer (k=3, 20³=8000 dims) trên chuỗi amino acid, normalized theo unit sum.
- **Tanimoto_Similarity**: Độ đo tương đồng giữa hai fingerprint bit-vectors: `|A ∩ B| / |A ∪ B|`.
- **KNN_Sparsification**: Giữ lại top-k neighbors có similarity cao nhất cho mỗi node để tạo sparse graph.
- **MC_Dropout**: Monte Carlo Dropout — chạy forward pass nhiều lần với dropout bật để ước lượng epistemic uncertainty.
- **EMA**: Exponential Moving Average — cập nhật node features theo công thức `feat_ema = decay * feat_ema + (1 - decay) * new_feat`.
- **Edge_Churn**: Tỷ lệ edges thay đổi sau mỗi lần rebuild graph: `|symmetric_difference(old, new)| / max(|old|, 1)`.
- **PairGate**: Fusion module dùng `(student_var, teacher_var)` làm gate input để tính scalar α ∈ (0,1).
- **HybridDTIModel**: Model orchestration kết hợp Student, Teacher và Fusion.
- **drug_index**: LongTensor trong batch, là sequential node index (0..n_unique_drugs-1) dùng để lookup node trong DD_Graph.
- **target_index**: LongTensor trong batch, là sequential node index (0..n_unique_proteins-1) dùng để lookup node trong PP_Graph.

---

## Requirements

### Requirement 1: Morgan Fingerprint Extraction cho Drug Nodes

**User Story:** As a researcher, I want drug nodes in the DD graph to be initialized with RDKit Morgan fingerprints, so that the GNN has chemically meaningful input features.

#### Acceptance Criteria

1. WHEN `Graph_Builder` receives a list of SMILES strings, THE `Graph_Builder` SHALL compute a Morgan fingerprint bit-vector (radius=2, nBits=2048) for each valid SMILES using RDKit.
2. IF a SMILES string cannot be parsed by RDKit, THEN THE `Graph_Builder` SHALL assign a zero vector of shape `(2048,)` to that drug node.
3. THE `Graph_Builder` SHALL return node feature matrix `x` of shape `(n_drugs, 2048)` as a `torch.FloatTensor`.
4. THE `Graph_Builder` SHALL preserve node ordering such that node `i` corresponds to the drug at index `i` in the input SMILES list.

---

### Requirement 2: K-mer Feature Extraction cho Protein Nodes

**User Story:** As a researcher, I want protein nodes in the PP graph to be initialized with k-mer frequency vectors, so that the GNN has sequence-based meaningful input features.

#### Acceptance Criteria

1. WHEN `Graph_Builder` receives a list of amino acid sequences, THE `Graph_Builder` SHALL compute a 3-mer frequency vector of dimension 8000 (20³) for each sequence.
2. THE `Graph_Builder` SHALL normalize each k-mer vector by its L1 sum; IF the sum is zero, THEN THE `Graph_Builder` SHALL return a zero vector.
3. THE `Graph_Builder` SHALL return node feature matrix `x` of shape `(n_proteins, 8000)` as a `torch.FloatTensor`.
4. THE `Graph_Builder` SHALL preserve node ordering such that node `i` corresponds to the protein at index `i` in the input sequence list.

---

### Requirement 3: Drug-Drug Similarity Graph Construction

**User Story:** As a researcher, I want a Drug-Drug similarity graph built from Tanimoto similarity on Morgan fingerprints, so that the Teacher GNN can propagate information between structurally similar drugs.

#### Acceptance Criteria

1. WHEN `Graph_Builder` builds the DD_Graph, THE `Graph_Builder` SHALL compute pairwise Tanimoto similarity between all drug fingerprint pairs using `DataStructs.BulkTanimotoSimilarity`.
2. THE `Graph_Builder` SHALL apply kNN sparsification, keeping the top-k (default k=10) most similar neighbors per node, excluding self-loops.
3. THE `Graph_Builder` SHALL produce an undirected graph by adding both `(i, j)` and `(j, i)` for each selected edge.
4. THE `Graph_Builder` SHALL return a `torch_geometric.data.Data` object with `x` (node features), `edge_index` (shape `(2, E)`), and `num_nodes`.
5. WHEN no valid edges exist, THE `Graph_Builder` SHALL return `edge_index` of shape `(2, 0)` rather than raising an error.

---

### Requirement 4: Protein-Protein Similarity Graph Construction

**User Story:** As a researcher, I want a Protein-Protein similarity graph built from cosine similarity on k-mer vectors, so that the Teacher GNN can propagate information between sequence-similar proteins.

#### Acceptance Criteria

1. WHEN `Graph_Builder` builds the PP_Graph, THE `Graph_Builder` SHALL compute pairwise cosine similarity between all protein k-mer vectors using `sklearn.metrics.pairwise.cosine_similarity`.
2. THE `Graph_Builder` SHALL apply kNN sparsification, keeping the top-k (default k=10) most similar neighbors per node, excluding self-loops.
3. THE `Graph_Builder` SHALL produce an undirected graph by adding both `(i, j)` and `(j, i)` for each selected edge.
4. THE `Graph_Builder` SHALL return a `torch_geometric.data.Data` object with `x`, `edge_index`, and `num_nodes`.
5. WHEN no valid edges exist, THE `Graph_Builder` SHALL return `edge_index` of shape `(2, 0)` rather than raising an error.

---

### Requirement 5: Graph Disk Cache

**User Story:** As a developer, I want DD and PP graphs to be cached to disk as `.pt` files, so that graph construction is not repeated on every training run.

#### Acceptance Criteria

1. WHEN `Graph_Builder` is called and a cache file exists at the configured path, THE `Graph_Builder` SHALL load and return the cached graphs without recomputing.
2. WHEN `Graph_Builder` is called and no cache file exists, THE `Graph_Builder` SHALL build both DD_Graph and PP_Graph, then save them to disk as a single `.pt` file containing `{"dd": dd_graph, "pp": pp_graph}`.
3. THE `Graph_Builder` SHALL create parent directories automatically if they do not exist before saving.
4. THE `Graph_Builder` SHALL log a message indicating whether graphs were loaded from cache or freshly built.
5. WHEN graphs are built, THE `Graph_Builder` SHALL verify `dd_graph.num_nodes == len(unique_drugs)` and `pp_graph.num_nodes == len(unique_proteins)`; IF the assertion fails, THEN THE `Graph_Builder` SHALL raise a `ValueError` with a descriptive message.

---

### Requirement 6: GCNTeacher Model

**User Story:** As a researcher, I want a GCN-based Teacher model registered in the model registry, so that it can be selected via Hydra config and produce meaningful DTI predictions from graph structure.

#### Acceptance Criteria

1. THE `GCNTeacher` SHALL be registered with `@MODELS.register("gcn_teacher")` so that it is instantiable via the registry.
2. WHEN `GCNTeacher.forward(batch)` is called, THE `GCNTeacher` SHALL read `batch["drug_index"]` and `batch["target_index"]` to lookup node embeddings from the global DD_Graph and PP_Graph respectively.
3. THE `GCNTeacher` SHALL return a dict `{"logits": tensor}` where `tensor` has shape `(B,)` and dtype `torch.float32`.
4. THE `GCNTeacher` SHALL include `nn.Dropout` layers with `dropout >= 0.2` in both the GCN encoder and the predictor MLP, so that MC-Dropout uncertainty estimation produces non-zero variance.
5. THE `GCNTeacher` SHALL expose a `set_graphs(dd_graph, pp_graph)` method that stores the global graphs as instance attributes.
6. WHEN `GCNTeacher.set_graphs()` has not been called before `forward()`, THE `GCNTeacher` SHALL raise an `AttributeError` or `RuntimeError` with a descriptive message.
7. THE `GCNTeacher` SHALL accept hyperparameters `drug_feat_dim`, `protein_feat_dim`, `hidden_dim`, `num_layers`, and `dropout` — all configurable via YAML, with no hardcoded values in Python.
8. WHEN `GCNTeacher` is run in `model.train()` mode with MC-Dropout (20 forward passes), THE `GCNTeacher` SHALL produce logit variance `> 0` for all samples in the batch.

---

### Requirement 7: YAML Config cho GCNTeacher

**User Story:** As a developer, I want a Hydra-compatible YAML config for GCNTeacher, so that hyperparameters can be changed from the CLI without editing Python code.

#### Acceptance Criteria

1. THE system SHALL provide a config file at `configs/model/teacher/gcn.yaml` with keys `name: gcn_teacher` and `params` containing `drug_feat_dim`, `protein_feat_dim`, `hidden_dim`, `num_layers`, and `dropout`.
2. THE `GCNTeacher` SHALL be instantiable from the config using `MODELS.build(cfg.model.teacher)` without additional arguments.
3. WHEN a user overrides `model.teacher.params.hidden_dim=256` via Hydra CLI, THE system SHALL instantiate `GCNTeacher` with `hidden_dim=256`.

---

### Requirement 8: Module Registration và Import

**User Story:** As a developer, I want GCNTeacher to be automatically registered when the package is imported, so that the registry can find it without manual registration calls.

#### Acceptance Criteria

1. THE `GCNTeacher` class SHALL be imported in `ugtsdti/models/__init__.py` so that the `@MODELS.register` decorator executes at package load time.
2. WHEN `from ugtsdti.models import GCNTeacher` is executed, THE system SHALL not raise an `ImportError`.
3. WHEN `MODELS.build({"name": "gcn_teacher", "params": {...}})` is called after package import, THE system SHALL return a `GCNTeacher` instance.

---

### Requirement 9: Unit Tests và Smoke Test

**User Story:** As a developer, I want unit tests for GCNTeacher and a smoke test verifying AUROC > 0.5 on S1, so that correctness can be verified automatically.

#### Acceptance Criteria

1. THE test suite SHALL include a test that instantiates `GCNTeacher` with mock DD/PP graphs, calls `forward()` with a batch of size 4, and asserts `out["logits"].shape == (4,)`.
2. THE test suite SHALL include a test that runs 20 MC-Dropout forward passes in `model.train()` mode and asserts `logits.var(dim=0) > 0` for all samples.
3. THE test suite SHALL include a test that calls `Graph_Builder` with a list of valid SMILES and asserts `dd_graph.num_nodes == len(smiles_list)` and `dd_graph.x.shape[1] == 2048`.
4. THE test suite SHALL include a test that calls `Graph_Builder` with a list of amino acid sequences and asserts `pp_graph.num_nodes == len(sequences)` and `pp_graph.x.shape[1] == 8000`.
5. WHEN `GCNTeacher` is trained on DAVIS S1 split for at least 10 epochs using `model=only_teacher`, THE `GCNTeacher` SHALL achieve AUROC > 0.5 on the S1 validation set.

---

### Requirement 10: Graph Stability Controls — Warmup và Rebuild

**User Story:** As a researcher, I want graph topology to be frozen during early training and periodically rebuilt, so that the GNN learns stable embeddings before the graph structure changes.

#### Acceptance Criteria

1. THE system SHALL support a `warmup_epochs` config parameter; WHILE `current_epoch < warmup_epochs`, THE system SHALL freeze the graph topology (no edge updates).
2. THE system SHALL support a `rebuild_every` config parameter; WHEN `current_epoch % rebuild_every == 0` and `current_epoch >= warmup_epochs`, THE system SHALL rebuild the kNN graph from current node features.
3. THE `warmup_epochs` and `rebuild_every` parameters SHALL be configurable via `configs/trainer/default_trainer.yaml` under a `graph:` key, with no hardcoded values in Python.

---

### Requirement 11: EMA Embedding Update

**User Story:** As a researcher, I want node features to be updated with EMA smoothing after each epoch, so that sudden feature shifts do not destabilize training.

#### Acceptance Criteria

1. THE system SHALL support an `ema_decay` config parameter (default 0.99); WHEN node features are updated after an epoch, THE system SHALL apply `feat_ema = ema_decay * feat_ema + (1 - ema_decay) * new_feat`.
2. THE `ema_decay` parameter SHALL be configurable via `configs/trainer/default_trainer.yaml` under the `graph:` key.
3. WHEN `ema_decay = 1.0`, THE system SHALL keep node features unchanged (no update), effectively disabling EMA.

---

### Requirement 12: Edge Churn Logging

**User Story:** As a researcher, I want the percentage of edges that changed after each graph rebuild to be logged to WandB, so that I can monitor graph stability during training.

#### Acceptance Criteria

1. WHEN the kNN graph is rebuilt, THE system SHALL compute Edge_Churn as `len(symmetric_difference(old_edges, new_edges)) / max(len(old_edges), 1)`.
2. THE system SHALL log Edge_Churn to WandB under the key `"graph/edge_churn"` after each rebuild.
3. WHEN no rebuild occurs in an epoch, THE system SHALL not log `"graph/edge_churn"` for that epoch.

---

### Requirement 13: Round-Trip Graph Serialization

**User Story:** As a developer, I want graph cache files to be correctly serializable and deserializable, so that loaded graphs are identical to the originally built graphs.

#### Acceptance Criteria

1. FOR ALL valid DD_Graph objects built by `Graph_Builder`, saving then loading via `torch.save` / `torch.load` SHALL produce a graph with identical `x`, `edge_index`, and `num_nodes`.
2. FOR ALL valid PP_Graph objects built by `Graph_Builder`, saving then loading via `torch.save` / `torch.load` SHALL produce a graph with identical `x`, `edge_index`, and `num_nodes`.
