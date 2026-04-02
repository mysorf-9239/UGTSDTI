# Tasks — UGTSDTI Codebase Fixes

## Task List

- [x] 1 Fix Bug 1: State.get() detach phá gradient graph
  - [x] 1.1 Sửa `_isolate_value()` trong `ugtsdti/core/state.py`: thay `value.detach().clone()` bằng `value.clone()` để giữ gradient graph
  - [x] 1.2 Verify `state.get()` cũng chỉ clone (không detach) khi trả về tensor
  - [x] 1.3 Thêm/update test `test_committed_tensor_preserves_autograd_graph` trong `tests/quality/test_state.py` để assert `source.grad is not None` sau `backward()`
  - [x] 1.4 Verify các test isolation hiện tại vẫn pass (mutation isolation không bị phá)

- [x] 2 Fix Bug 2: Dual code path interaction/decision, risk double execution
  - [x] 2.1 Xóa block `if not cfg.get("postprocess"):` trong `_run_core_pipeline()` của `ugtsdti/trainer/trainer.py` (không commit decision trong core pipeline)
  - [x] 2.2 Thêm decision execution vào `run_until_decision()` sau khi gọi `_run_core_pipeline()`
  - [x] 2.3 Verify `run_batch()` không còn double-execute interaction hay decision
  - [x] 2.4 Update integration tests nếu cần để reflect pipeline contract mới

- [x] 3 Fix Bug 3: Metrics không tính trong train mode
  - [x] 3.1 Sửa condition trong `run_postprocess()` của `ugtsdti/postprocess/pipeline.py`: bỏ `context.mode in ("eval", "inference")` check, chỉ giữ `metrics_cfg and metrics_cfg.get("enabled")`
  - [x] 3.2 Verify test `test_disabled_kd_pipeline_uses_explicit_noop_path_without_hidden_outputs` pass với `state.has("metrics.f1")` trong train mode

- [x] 4 Fix Bug 4: Interaction split ownership (covered by Bug 2 fix)
  - [x] 4.1 Verify sau fix Bug 2, `run_batch()` flow đúng thứ tự: Graph → Role Binding → Interaction (trong core pipeline) → Decision → Loss → Metrics (trong postprocess)
  - [x] 4.2 Verify test `test_teacher_student_kd_pipeline_maps_interaction_loss_explicitly` pass: `state.has("interaction.kd.loss_component")` và `state.has("loss.kd")`

- [x] 5 Fix Bug 5: Cross-section validation chỉ work khi output_attrs explicit
  - [x] 5.1 Sửa `_collect_graph_output_keys()` trong `ugtsdti/config/validate.py`: nếu node config không có `output_attrs` và registry bound, lookup attrs từ `registry.get_spec(type_key).output_attrs`
  - [x] 5.2 Thêm test trong `tests/quality/test_config.py` cho config hợp lệ không có `output_attrs` explicit

- [x] 6 Fix Bug 6: LazyLinear unsafe với checkpoint trước forward pass
  - [x] 6.1 Thêm helper `_has_uninitialized_lazy(module)` trong `ugtsdti/nodes/baseline.py`
  - [x] 6.2 Sửa `_TorchRuntime.state_dict()`: skip modules có uninitialized LazyLinear thay vì serialize chúng
  - [x] 6.3 Thêm test trong `tests/quality/test_baseline_nodes.py` cho `state_dict()` trước forward pass

- [x] 7 Fix Bug 7: deterministic bị loại khỏi config hash
  - [x] 7.1 Xóa `"deterministic"` khỏi `_RUNTIME_OPERATIONAL_HASH_FIELDS` trong `ugtsdti/runtime/identity.py`
  - [x] 7.2 Thêm test trong `tests/quality/test_runtime.py` assert hai configs chỉ khác `deterministic` có hash khác nhau

- [x] 8 Fix Bug 8: GraphEngine fingerprint check O(N×tensor_size) per node
  - [x] 8.1 Sửa `GraphEngine.run()` trong `ugtsdti/graph/engine.py`: dùng `state._fingerprint` (cached) thay vì gọi `state.get_fingerprint()` trong mutation check hot path
  - [x] 8.2 Verify mutation detection vẫn hoạt động đúng sau optimization
  - [x] 8.3 Thêm test trong `tests/quality/test_graph_engine.py` verify mutation detection vẫn raise `RuntimeError`
