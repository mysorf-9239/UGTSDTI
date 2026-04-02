# Bugfix Requirements Document

## Introduction

Tám bugs tồn đọng trong codebase UGTSDTI ảnh hưởng đến tính đúng đắn của training (gradient bị cắt, metrics thiếu trong train mode), tính ổn định của pipeline (double execution, split ownership), tính hợp lệ của config validation (false reject), tính an toàn của checkpoint (LazyLinear uninitialized), tính chính xác của reproducibility (config hash thiếu `deterministic`), và hiệu năng (fingerprint O(N×tensor_size) per node). Các bugs này cần được fix đồng thời để đảm bảo pipeline contract và research reproducibility.

---

## Bug Analysis

### Bug 1 — `State.get()` detach phá gradient graph

#### Current Behavior (Defect)

1.1 WHEN `state.get(key)` được gọi với một key chứa tensor có `requires_grad=True` THEN hệ thống trả về tensor đã bị `.detach().clone()`, cắt đứt gradient graph.

1.2 WHEN `loss = state.get("loss.total")` và `loss.backward()` được gọi THEN hệ thống không flow gradient về source tensor vì tensor đã bị detach tại thời điểm `commit()`.

#### Expected Behavior (Correct)

2.1 WHEN `state.get(key)` được gọi với key chứa tensor có `requires_grad=True` THEN hệ thống SHALL trả về tensor giữ nguyên gradient graph (không detach), chỉ clone để tránh shared memory mutation.

2.2 WHEN `loss = state.get("loss.total")` và `loss.backward()` được gọi THEN hệ thống SHALL flow gradient về source tensor, `source.grad` SHALL không phải `None`.

#### Unchanged Behavior (Regression Prevention)

3.1 WHEN tensor được commit vào state và source tensor bị mutate sau đó THEN hệ thống SHALL CONTINUE TO trả về giá trị gốc tại thời điểm commit (isolation vẫn đảm bảo).

3.2 WHEN `state.get(key)` trả về tensor và caller mutate tensor đó THEN hệ thống SHALL CONTINUE TO giữ nguyên giá trị trong internal store (read isolation vẫn đảm bảo).

3.3 WHEN tensor không có `requires_grad` được commit THEN hệ thống SHALL CONTINUE TO trả về tensor isolated như trước.

---

### Bug 2 — Dual code path interaction/decision, risk double execution

#### Current Behavior (Defect)

1.3 WHEN config có cả `interaction_plan` và `postprocess` THEN hệ thống chạy interaction trong `_run_core_pipeline()` VÀ `run_postprocess()` cũng có thể chạy lại interaction-dependent logic, dẫn đến double execution.

1.4 WHEN `_run_core_pipeline()` chạy với `cfg.get("postprocess")` là falsy THEN hệ thống commit decision vào state, nhưng `run_batch()` sau đó gọi `run_postprocess()` cũng chạy decision, gây `KeyCollisionError` hoặc silent double execution.

#### Expected Behavior (Correct)

2.3 WHEN `run_batch()` được gọi THEN hệ thống SHALL chạy interaction đúng một lần duy nhất, không phụ thuộc vào sự hiện diện của `postprocess` key trong config.

2.4 WHEN `run_batch()` được gọi THEN hệ thống SHALL chạy decision đúng một lần duy nhất trong postprocess stage, không commit decision trong `_run_core_pipeline()`.

#### Unchanged Behavior (Regression Prevention)

3.4 WHEN `run_until_decision()` được gọi THEN hệ thống SHALL CONTINUE TO chạy interaction và decision trong core pipeline và trả về state sau decision.

3.5 WHEN config không có `interaction_plan` THEN hệ thống SHALL CONTINUE TO bỏ qua interaction stage mà không raise error.

---

### Bug 3 — Metrics không tính trong train mode

#### Current Behavior (Defect)

1.5 WHEN `run_postprocess()` được gọi với `context.mode="train"` THEN hệ thống không chạy metrics computation, `state.has("metrics.f1")` trả về `False`.

1.6 WHEN `Trainer.step()` chạy training step THEN `logged_metrics` không chứa `metrics.*` keys, monitoring và early stopping không có data.

#### Expected Behavior (Correct)

2.5 WHEN `run_postprocess()` được gọi với `metrics_cfg.get("enabled")` là truthy THEN hệ thống SHALL chạy metrics computation bất kể `context.mode` là `"train"`, `"eval"`, hay `"inference"`.

2.6 WHEN `Trainer.step()` chạy training step với metrics enabled THEN `logged_metrics` SHALL chứa `metrics.*` keys.

#### Unchanged Behavior (Regression Prevention)

3.6 WHEN `metrics_cfg` là `None` hoặc `metrics_cfg.get("enabled")` là falsy THEN hệ thống SHALL CONTINUE TO bỏ qua metrics computation.

3.7 WHEN `context.mode="eval"` và metrics enabled THEN hệ thống SHALL CONTINUE TO chạy metrics như trước.

---

### Bug 4 — Interaction không chạy trong postprocess, split ownership vi phạm pipeline contract

#### Current Behavior (Defect)

1.7 WHEN `run_batch()` được gọi với config có `interaction_plan` THEN interaction chạy trong `_run_core_pipeline()` nhưng `run_postprocess()` không biết về interaction outputs, tạo split ownership.

1.8 WHEN loss mapping reference key từ interaction (e.g., `interaction.kd.loss_component`) THEN hệ thống có thể raise `InvalidLossMappingError` nếu interaction chưa chạy trước postprocess.

#### Expected Behavior (Correct)

2.7 WHEN `run_batch()` được gọi THEN hệ thống SHALL đảm bảo interaction outputs đã có trong state trước khi postprocess (loss/metrics) chạy.

2.8 WHEN pipeline contract là `Graph → Role Binding → Interaction → Decision → Loss + Metrics` THEN hệ thống SHALL thực thi đúng thứ tự này, không split ownership giữa core pipeline và postprocess.

#### Unchanged Behavior (Regression Prevention)

3.8 WHEN interaction disabled (noop hoặc `enabled=False`) THEN hệ thống SHALL CONTINUE TO chạy loss và metrics mà không cần interaction outputs.

3.9 WHEN `run_until_decision()` được gọi THEN hệ thống SHALL CONTINUE TO trả về state sau decision mà không chạy loss/metrics.

---

### Bug 5 — Cross-section validation chỉ work khi `output_attrs` explicit

#### Current Behavior (Defect)

1.9 WHEN node config không khai báo `output_attrs` explicit (dùng spec từ registry) THEN `_collect_graph_output_keys()` trả về empty set, và `_check_roles_reference_graph_outputs()` raise `InvalidConfigError` với "not produced by any graph node".

1.10 WHEN config hợp lệ (node type registered, output attrs đúng theo spec) nhưng không có `output_attrs` trong YAML THEN hệ thống reject config hợp lệ.

#### Expected Behavior (Correct)

2.9 WHEN `_collect_graph_output_keys()` được gọi với registry bound THEN hệ thống SHALL lookup output attrs từ registry spec nếu node config không khai báo `output_attrs` explicit.

2.10 WHEN config hợp lệ với node type registered và output attrs đúng theo spec THEN hệ thống SHALL accept config mà không raise error, bất kể `output_attrs` có được khai báo explicit hay không.

#### Unchanged Behavior (Regression Prevention)

3.10 WHEN node config khai báo `output_attrs` explicit THEN hệ thống SHALL CONTINUE TO dùng declared attrs cho cross-section validation.

3.11 WHEN node type không được registered và registry bound THEN hệ thống SHALL CONTINUE TO raise `InvalidConfigError`.

---

### Bug 6 — `LazyLinear` unsafe với checkpoint trước forward pass

#### Current Behavior (Defect)

1.11 WHEN `state_dict()` được gọi trên `ConcatFusionRuntime` hoặc `MLPHeadRuntime` trước lần forward đầu tiên THEN hệ thống serialize uninitialized `LazyLinear` weights.

1.12 WHEN checkpoint được save trước epoch 1 và sau đó `load_state_dict()` được gọi THEN hệ thống có thể fail hoặc restore model state corrupt.

#### Expected Behavior (Correct)

2.11 WHEN `state_dict()` được gọi trước forward pass trên runtime có `LazyLinear` THEN hệ thống SHALL raise `RuntimeError` hoặc trả về empty dict với warning, không serialize uninitialized weights.

2.12 WHEN checkpoint save/restore cycle xảy ra THEN hệ thống SHALL đảm bảo chỉ serialize weights đã được materialized.

#### Unchanged Behavior (Regression Prevention)

3.12 WHEN `state_dict()` được gọi sau ít nhất một forward pass THEN hệ thống SHALL CONTINUE TO serialize weights đúng.

3.13 WHEN `input_dim` được cung cấp explicit (dùng `nn.Linear` thay `nn.LazyLinear`) THEN hệ thống SHALL CONTINUE TO hoạt động như trước.

---

### Bug 7 — `deterministic` bị loại khỏi config hash, reproducibility gap

#### Current Behavior (Defect)

1.13 WHEN `hash_config()` được gọi với config có `runtime.deterministic=True` THEN hệ thống loại `deterministic` khỏi hash, trả về cùng hash với config có `runtime.deterministic=False`.

1.14 WHEN researcher chạy experiment với `deterministic=False` rồi chạy lại với `deterministic=True` THEN cùng `config_hash` nhưng kết quả có thể khác nhau, reproducibility claim sai.

#### Expected Behavior (Correct)

2.13 WHEN `hash_config()` được gọi THEN hệ thống SHALL include `deterministic` trong config hash computation.

2.14 WHEN hai configs chỉ khác `runtime.deterministic` THEN hệ thống SHALL trả về hai `config_hash` khác nhau.

#### Unchanged Behavior (Regression Prevention)

3.14 WHEN các operational fields khác (`artifacts_dir`, `batch_size`, `device`, v.v.) bị loại khỏi hash THEN hệ thống SHALL CONTINUE TO loại chúng như trước.

3.15 WHEN config không có `runtime.deterministic` THEN hệ thống SHALL CONTINUE TO compute hash bình thường.

---

### Bug 8 — GraphEngine fingerprint check O(N×tensor_size) per node

#### Current Behavior (Defect)

1.15 WHEN `GraphEngine.run()` thực thi mỗi node THEN hệ thống gọi `state.get_fingerprint()` 2-3 lần per node, mỗi lần recompute toàn bộ MD5 hash của tất cả tensor values trong state.

1.16 WHEN batch size lớn (e.g., 512 samples × 512-dim embeddings) và nhiều nodes THEN training throughput giảm đáng kể do fingerprint recomputation overhead O(N×tensor_size).

#### Expected Behavior (Correct)

2.15 WHEN `GraphEngine.run()` thực thi nodes THEN hệ thống SHALL dùng cached fingerprint từ `state._fingerprint` thay vì recompute, chỉ recompute khi state thực sự thay đổi.

2.16 WHEN mutation check được thực hiện THEN hệ thống SHALL so sánh cached fingerprint values mà không trigger full recomputation.

#### Unchanged Behavior (Regression Prevention)

3.16 WHEN node mutate state ngoài `StateWriter.commit()` THEN hệ thống SHALL CONTINUE TO detect mutation và raise `RuntimeError`.

3.17 WHEN `StateWriter.commit()` được gọi THEN hệ thống SHALL CONTINUE TO update fingerprint sau mỗi commit hợp lệ.
