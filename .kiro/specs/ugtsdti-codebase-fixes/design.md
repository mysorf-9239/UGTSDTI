# Design Document — UGTSDTI Codebase Fixes

## Technical Context

- Python 3.10+, PyTorch 2.0+
- Pipeline contract: `Batch → Graph → Role Binding → Interaction → Decision → Loss + Metrics`
- State: write-once, single-producer, monotonically-growing key-value store
- Tests: `tests/quality/`, `tests/pbt/`, `tests/integration/`

---

## Bug Condition Methodology

### Bug 1 — `State.get()` detach phá gradient graph

**Bug Condition:**
```pascal
FUNCTION isBugCondition_1(X)
  INPUT: X of type (key: str, value: torch.Tensor)
  OUTPUT: boolean
  RETURN X.value.requires_grad = True
END FUNCTION
```

**Fix Checking:**
```pascal
FOR ALL X WHERE isBugCondition_1(X) DO
  writer.commit("p", {X.key: X.value})
  restored = state.get(X.key)
  ASSERT restored.grad_fn IS NOT None OR restored.requires_grad = True
  restored.backward()
  ASSERT X.value.grad IS NOT None
END FOR
```

**Preservation:**
```pascal
FOR ALL X WHERE NOT isBugCondition_1(X) DO
  ASSERT F(X) = F'(X)  // isolation behavior unchanged
END FOR
```

**Root Cause:** `_isolate_value()` gọi `value.detach().clone()` cho mọi tensor. `detach()` cắt gradient graph. Fix: chỉ `.clone()` (không `.detach()`) khi commit. Khi read (`state.get()`), vẫn clone để tránh mutation nhưng không detach.

**Implementation:**
```python
# _isolate_value() — dùng khi commit (write isolation)
if isinstance(value, torch.Tensor):
    return value.clone()  # giữ grad_fn, chỉ tránh shared memory

# state.get() — dùng khi read (read isolation)
if isinstance(value, torch.Tensor):
    return value.clone()  # clone để tránh caller mutation, giữ grad_fn
```

**Files:** `ugtsdti/core/state.py`

---

### Bug 2 — Dual code path interaction/decision, risk double execution

**Bug Condition:**
```pascal
FUNCTION isBugCondition_2(cfg)
  INPUT: cfg of type dict
  OUTPUT: boolean
  RETURN cfg.has("interaction_plan") AND run_batch() is called
END FUNCTION
```

**Fix Checking:**
```pascal
FOR ALL cfg WHERE isBugCondition_2(cfg) DO
  state, trace = executor.run_batch(batch, cfg, context)
  ASSERT interaction_committed_count(state) = 1
  ASSERT decision_committed_count(state) = 1
  ASSERT no KeyCollisionError raised
END FOR
```

**Root Cause:** `_run_core_pipeline()` có hai conditional blocks với logic rẽ nhánh không nhất quán:
1. `if cfg.get("interaction_plan")` → chạy interaction
2. `if not cfg.get("postprocess")` → chạy decision

Khi `run_batch()` gọi `run_postprocess()`, postprocess cũng chạy decision. Fix: `_run_core_pipeline()` luôn chạy interaction (nếu có plan), không bao giờ chạy decision. Decision luôn thuộc về postprocess.

**Implementation:**
```python
def _run_core_pipeline(self, batch, cfg, context):
    # ... graph, role binding ...

    # Interaction: luôn chạy nếu có plan
    if cfg.get("interaction_plan"):
        self._run_interactions(state, writer, cfg, context)
        trace.state_boundary_summaries["interaction"] = state.keys()

    # Decision: KHÔNG chạy ở đây — thuộc về postprocess
    # (xóa block `if not cfg.get("postprocess")`)

    return state, writer, trace
```

`run_until_decision()` cần chạy decision sau core pipeline:
```python
def run_until_decision(self, batch, cfg, context):
    state, writer, trace = self._run_core_pipeline(batch, cfg, context)
    decision = self._build_decision_module(cfg)
    writer.commit("decision", decision.forward(state, context))
    trace.state_boundary_summaries["decision"] = state.keys()
    return state, trace
```

**Files:** `ugtsdti/trainer/trainer.py`

---

### Bug 3 — Metrics không tính trong train mode

**Bug Condition:**
```pascal
FUNCTION isBugCondition_3(context, metrics_cfg)
  INPUT: context.mode: str, metrics_cfg: dict
  OUTPUT: boolean
  RETURN context.mode = "train" AND metrics_cfg.get("enabled") IS truthy
END FUNCTION
```

**Fix Checking:**
```pascal
FOR ALL (context, cfg) WHERE isBugCondition_3(context, cfg.metrics) DO
  state = run_postprocess(state, cfg, context, labels)
  ASSERT state.has("metrics.f1") OR state.has("metrics.auroc")
END FOR
```

**Root Cause:** Condition `if context.mode in ("eval", "inference")` loại train mode. Fix: bỏ mode check, chỉ check `metrics_cfg.get("enabled")`.

**Implementation:**
```python
# run_postprocess() trong pipeline.py
# Trước:
if context.mode in ("eval", "inference") and metrics_cfg and metrics_cfg.get("enabled"):
    state = run_metrics(...)

# Sau:
if metrics_cfg and metrics_cfg.get("enabled"):
    state = run_metrics(...)
```

**Files:** `ugtsdti/postprocess/pipeline.py`

---

### Bug 4 — Interaction không chạy trong postprocess, split ownership

**Bug Condition:**
```pascal
FUNCTION isBugCondition_4(cfg)
  INPUT: cfg of type dict
  OUTPUT: boolean
  RETURN cfg.loss.map references interaction output keys
         AND interaction runs in _run_core_pipeline (not postprocess)
END FUNCTION
```

**Fix Checking:**
```pascal
FOR ALL cfg WHERE isBugCondition_4(cfg) DO
  state, trace = executor.run_batch(batch, cfg, context)
  ASSERT state.has("interaction.kd.loss_component")
  ASSERT state.has("loss.kd")
  ASSERT no InvalidLossMappingError raised
END FOR
```

**Root Cause:** Pipeline contract bị vi phạm — interaction chạy trong core pipeline, loss/metrics chạy trong postprocess, nhưng postprocess không nhận interaction outputs. Fix: đảm bảo interaction luôn chạy trước loss trong `run_batch()` flow. Với fix Bug 2, interaction đã chạy trong `_run_core_pipeline()` trước khi `run_postprocess()` được gọi — đây là đúng thứ tự. Không cần thay đổi thêm nếu Bug 2 được fix đúng.

**Verification:** Sau fix Bug 2, `run_batch()` flow là:
1. `_run_core_pipeline()`: Graph → Role Binding → Interaction
2. `run_postprocess()`: Decision → Loss → Metrics

State từ step 1 được pass vào step 2, nên interaction outputs có sẵn cho loss mapping.

**Files:** `ugtsdti/trainer/trainer.py` (covered by Bug 2 fix)

---

### Bug 5 — Cross-section validation chỉ work khi `output_attrs` explicit

**Bug Condition:**
```pascal
FUNCTION isBugCondition_5(node_cfg, registry)
  INPUT: node_cfg: dict, registry: NodeRegistry
  OUTPUT: boolean
  RETURN "output_attrs" NOT IN node_cfg
         AND node_cfg.type_key IS registered in registry
END FUNCTION
```

**Fix Checking:**
```pascal
FOR ALL node_cfg WHERE isBugCondition_5(node_cfg, registry) DO
  produced = validator._collect_graph_output_keys(cfg)
  ASSERT produced IS NOT empty
  ASSERT no InvalidConfigError raised for valid role references
END FOR
```

**Root Cause:** `_collect_graph_output_keys()` chỉ dùng `node_cfg.get("output_attrs", [])`. Fix: nếu `output_attrs` không có trong config nhưng registry bound, lookup từ spec.

**Implementation:**
```python
def _collect_graph_output_keys(self, cfg):
    graph = cfg.get("graph", {})
    nodes = graph.get("nodes", {})
    produced = set()

    if isinstance(nodes, dict):
        items = nodes.items()
    else:
        items = ((n.get("name", ""), n) for n in nodes if isinstance(n, dict))

    for node_name, node_cfg in items:
        if not isinstance(node_cfg, dict):
            continue
        declared = node_cfg.get("output_attrs", [])
        if declared:
            for attr in declared:
                produced.add(f"{node_name}.{attr}")
        elif self._graph_registry is not None:
            type_key = str(node_cfg.get("type_key", node_cfg.get("type", "")))
            if type_key:
                try:
                    spec = self._graph_registry.get_spec(type_key)
                    for attr in getattr(spec, "output_attrs", []):
                        produced.add(f"{node_name}.{attr}")
                except Exception:
                    pass  # unregistered type handled elsewhere
    return produced
```

**Files:** `ugtsdti/config/validate.py`

---

### Bug 6 — `LazyLinear` unsafe với checkpoint trước forward pass

**Bug Condition:**
```pascal
FUNCTION isBugCondition_6(runtime)
  INPUT: runtime of type _TorchRuntime
  OUTPUT: boolean
  RETURN runtime has LazyLinear module
         AND runtime.forward() has NOT been called yet
END FUNCTION
```

**Fix Checking:**
```pascal
FOR ALL runtime WHERE isBugCondition_6(runtime) DO
  result = runtime.state_dict()
  ASSERT result = {} OR RuntimeError raised
  // NOT: uninitialized weights serialized
END FOR
```

**Root Cause:** `_TorchRuntime.state_dict()` gọi `module.state_dict()` mà không check materialization. Fix: detect `LazyLinear` chưa materialized và trả về empty dict hoặc raise.

**Implementation:**
```python
def state_dict(self) -> dict[str, Any]:
    _, nn = _require_torch()
    state: dict[str, Any] = {}
    for name, value in self.__dict__.items():
        if isinstance(value, nn.Module):
            # Check for uninitialized LazyLinear
            if _has_uninitialized_lazy(value):
                # Skip — không serialize uninitialized weights
                continue
            state[name] = value.state_dict()
    return state

def _has_uninitialized_lazy(module) -> bool:
    try:
        import torch.nn as nn
        for m in module.modules():
            if isinstance(m, nn.modules.lazy.LazyModuleMixin):
                if m.has_uninitialized_params():
                    return True
    except Exception:
        pass
    return False
```

**Files:** `ugtsdti/nodes/baseline.py`

---

### Bug 7 — `deterministic` bị loại khỏi config hash

**Bug Condition:**
```pascal
FUNCTION isBugCondition_7(cfg1, cfg2)
  INPUT: cfg1, cfg2 of type dict
  OUTPUT: boolean
  RETURN cfg1 và cfg2 chỉ khác runtime.deterministic
END FUNCTION
```

**Fix Checking:**
```pascal
FOR ALL (cfg1, cfg2) WHERE isBugCondition_7(cfg1, cfg2) DO
  hash1 = hash_config(cfg1)
  hash2 = hash_config(cfg2)
  ASSERT hash1 ≠ hash2
END FOR
```

**Root Cause:** `_RUNTIME_OPERATIONAL_HASH_FIELDS` include `"deterministic"`. Fix: xóa `"deterministic"` khỏi set này.

**Implementation:**
```python
_RUNTIME_OPERATIONAL_HASH_FIELDS = {
    "artifacts_dir",
    "batch_size",
    "checkpoint_every_epochs",
    "checkpoint_dir",
    "checkpoint_path",
    "data_dir",
    "debug",
    # "deterministic" — REMOVED: affects algorithm selection, must be in hash
    "device",
    "num_workers",
    "pin_memory",
    "precision",
}
```

**Files:** `ugtsdti/runtime/identity.py`

---

### Bug 8 — GraphEngine fingerprint check O(N×tensor_size) per node

**Bug Condition:**
```pascal
FUNCTION isBugCondition_8(state, node_count, batch_size)
  INPUT: state: State, node_count: int, batch_size: int
  OUTPUT: boolean
  RETURN node_count > 1 AND batch_size > 64
         AND get_fingerprint() called multiple times per node
END FUNCTION
```

**Fix Checking:**
```pascal
FOR ALL (state, cfg) WHERE isBugCondition_8(state, ...) DO
  t_before = time()
  engine.run(plan, state, writer, context)
  t_after = time()
  ASSERT (t_after - t_before) does not scale with tensor byte size
  ASSERT mutation detection still works
END FOR
```

**Root Cause:** `state.get_fingerprint()` luôn recompute từ đầu. `state._fingerprint` là cached value nhưng không được dùng trong mutation check. Fix: dùng `state._fingerprint` (cached) cho comparison, chỉ recompute khi cần.

**Implementation:**

Trong `GraphEngine.run()`:
```python
# (A) BEFORE node execution — dùng cached fingerprint
fp_before_node = state._fingerprint  # không trigger recompute

# ... forward, validate ...

# (B) AFTER node execution (BEFORE commit) — compare cached vs current
# StateWriter.commit() đã update _fingerprint, nên nếu node mutate state
# ngoài commit(), _fingerprint sẽ stale
# Thay vì gọi get_fingerprint() 2 lần, chỉ cần check 1 lần:
if state._fingerprint != fp_before_node:
    raise RuntimeError("Illegal mutation INSIDE node")

writer.commit(node_name, qualified)

# (C) AFTER commit — fingerprint đã được update bởi commit()
# Không cần assert thêm
```

Trong `State.get_fingerprint()`: giữ nguyên để backward compat, nhưng `GraphEngine` không gọi nó trong hot path.

**Files:** `ugtsdti/graph/engine.py`

---

## Correctness Properties

### Property 1: Gradient Preservation
```pascal
FOR ALL tensor X WHERE X.requires_grad = True DO
  writer.commit("p", {"loss.total": X * 3.0})
  restored = state.get("loss.total")
  restored.backward()
  ASSERT X.grad IS NOT None
  ASSERT X.grad ≈ 3.0
END FOR
```

### Property 2: No Double Execution
```pascal
FOR ALL cfg WHERE cfg.has("interaction_plan") DO
  state, _ = executor.run_batch(batch, cfg, context)
  FOR ALL key IN interaction_output_keys DO
    ASSERT count_commits(key) = 1
  END FOR
END FOR
```

### Property 3: Metrics Mode Independence
```pascal
FOR ALL mode IN {"train", "eval", "inference"} DO
  FOR ALL cfg WHERE cfg.metrics.enabled IS truthy DO
    state = run_postprocess(state_with_logits, cfg, ExecutionContext(mode=mode), labels)
    ASSERT state.has("metrics.f1")
  END FOR
END FOR
```

### Property 4: Config Hash Determinism Sensitivity
```pascal
FOR ALL cfg DO
  cfg_det_true = cfg WITH runtime.deterministic = True
  cfg_det_false = cfg WITH runtime.deterministic = False
  ASSERT hash_config(cfg_det_true) ≠ hash_config(cfg_det_false)
END FOR
```

### Property 5: Fingerprint Mutation Detection
```pascal
FOR ALL state, node_mutation_fn DO
  fp_before = state._fingerprint
  node_mutation_fn(state)  // illegal mutation
  ASSERT engine detects mutation AND raises RuntimeError
END FOR
```
