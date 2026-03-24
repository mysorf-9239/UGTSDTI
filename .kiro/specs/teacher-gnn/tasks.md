# Tasks: Teacher GNN (Phase 11 UGTSDTI)

## Task List

- [x] 0. Refactor dataset indexing — sequential index thay MD5 hash (Option B)
  - [x] 0.1 Sửa `tdc_dataset.py`: build `smiles_to_idx` và `fasta_to_idx` mapping từ unique drugs/proteins, lưu `drug_index` và `target_index` là sequential index (0..N-1)
  - [x] 0.2 Sửa `baseline_teacher`: đổi `num_drugs=100003` → nhận `num_drugs` và `num_targets` từ config (= số unique drugs/proteins thực tế)
  - [x] 0.3 Xóa cache cũ (`data/cache/`) vì `drug_index` values thay đổi
  - [x] 0.4 Cập nhật `DECISIONS.md` ghi lại quyết định Option B

- [x] 1. Implement Graph Builder (`ugtsdti/data/graph_builder.py`)
  - [x] 1.1 Implement `build_drug_drug_graph()` — Morgan fingerprints + Tanimoto kNN
  - [x] 1.2 Implement `build_protein_protein_graph()` — k-mer vectors + cosine kNN
  - [x] 1.3 Implement `build_and_cache_graphs()` — build both graphs and save to `.pt`
  - [x] 1.4 Implement `load_graphs()` — load from cache
  - [x] 1.5 Add index mapping (`drug_id_to_node`, `protein_id_to_node`) to cache

- [x] 2. Implement GCNTeacher (`ugtsdti/models/teacher/gcn_teacher.py`)
  - [x] 2.1 Implement `GCNTeacher` class with `@MODELS.register("gcn_teacher")`
  - [x] 2.2 Implement `set_graphs(dd_graph, pp_graph)` method
  - [x] 2.3 Implement `forward(batch)` — lookup by index, GCN encode, predict
  - [x] 2.4 Add `RuntimeError` guard when `set_graphs()` not called before `forward()`
  - [x] 2.5 Ensure `nn.Dropout(dropout >= 0.2)` in both GCN encoder and predictor MLP

- [x] 3. Config and Registration
  - [x] 3.1 Create `configs/model/teacher/gcn.yaml`
  - [x] 3.2 Add `graph:` section to `configs/trainer/default_trainer.yaml`
  - [x] 3.3 Import `GCNTeacher` in `ugtsdti/models/__init__.py`

- [x] 4. Graph Stability Controls
  - [x] 4.1 Implement `ema_update(feat_old, feat_new, ema_decay)` utility function
  - [x] 4.2 Implement `compute_edge_churn(old_edges, new_edges)` utility function
  - [x] 4.3 Implement `maybe_rebuild_graph()` — warmup + rebuild_every logic + WandB logging

- [x] 5. Tests (`tests/test_teacher_gnn.py`)
  - [x] 5.1 Unit test: `GCNTeacher` registered, forward shape `(B,)`, no-graphs raises
  - [x] 5.2 Unit test: Graph Builder — invalid SMILES zero vector, single node no error, directory creation
  - [x] 5.3 Unit test: EMA decay=1.0 no change, cache hit returns same graphs
  - [x] 5.4 Property test (P1): Drug fingerprint shape and ordering invariant
  - [x] 5.5 Property test (P2): Protein k-mer shape and normalization invariant
  - [x] 5.6 Property test (P3): Graph symmetry (no self-loops, undirected)
  - [x] 5.7 Property test (P4): Graph degree bound <= 2*top_k
  - [x] 5.8 Property test (P5): Graph cache round-trip
  - [x] 5.9 Property test (P6): GCNTeacher forward shape for any batch size
  - [x] 5.10 Property test (P7): MC-Dropout variance > 0 for dropout >= 0.2
  - [x] 5.11 Property test (P8): EMA update formula correctness
  - [x] 5.12 Property test (P9): Edge churn computation formula
