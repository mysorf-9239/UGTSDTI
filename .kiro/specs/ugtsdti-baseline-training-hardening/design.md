# Thiết Kế: UGTSDTI Baseline Training Hardening

## 1. Mục Tiêu Thiết Kế

Giữ baseline graph nguyên trạng, chỉ harden orchestration quanh training để baseline trở thành nền đáng tin cho các model khó hơn.

## 2. Thiết Kế Cấu Hình

### 2.1 `training.optimizer`

```yaml
training:
  optimizer:
    type: adam
    lr: 0.001
    weight_decay: 0.0
```

### 2.2 `training.scheduler`

```yaml
training:
  scheduler:
    type: none
```

hoặc:

```yaml
training:
  scheduler:
    type: step
    step_size: 10
    gamma: 0.5
```

hoặc:

```yaml
training:
  scheduler:
    type: plateau
    mode: max
    factor: 0.5
    patience: 3
```

### 2.3 `training.loop`

```yaml
training:
  loop:
    epochs: 50
    checkpoint_every_epochs: 1
    summary_every_steps: 10
    eval_every_epochs: 1
    eval_partition: val
    max_grad_norm: 5.0
    fail_on_nonfinite_loss: true
    fail_on_nonfinite_grad: true
    select_checkpoint: best
    best_metric: metrics.auroc
    best_mode: max
    early_stopping:
      enabled: true
      patience: 5
      min_delta: 0.0
```

## 3. Runtime Design

### 3.1 Optimizer Builder

CLI/runtime builder tạo optimizer từ `training.optimizer.type`.

### 3.2 Scheduler Builder

Scheduler được tạo sau optimizer. `plateau` scheduler cần metric đầu vào nên sẽ được step sau validation metric; các scheduler khác step theo epoch.

### 3.3 Resume Path

`runtime.checkpoint_path` sẽ được dùng cho cả:
- `eval`: load checkpoint rồi evaluate
- `train`: resume nếu checkpoint tồn tại và tương thích

Resume path restore:
- graph runtime state
- optimizer state
- scheduler state
- counters `epoch`, `step`

### 3.4 Best Checkpoint

Train loop materialize:
- `last_checkpoint`
- `best_checkpoint`

Best selection dựa trên:
- `training.loop.best_metric`
- `training.loop.best_mode`

Nếu không có eval cadence hợp lệ, best checkpoint bị disable hoặc config validation phải reject nếu user vẫn yêu cầu.

### 3.5 Final Evaluation Policy

Cuối train, policy `select_checkpoint` quyết định final evaluation:
- `last`
- `best`

Nếu `best` được chọn, evaluator phải load best checkpoint trước khi eval/final summary.

### 3.6 Safety Guards

`Trainer.step()` thêm:
- kiểm finite loss trước backward
- backward
- optional grad clipping
- kiểm finite grad trước optimizer step

Nếu guard bật và vi phạm, raise `NumericalInstabilityError`.

## 4. Reporting Design

Train summary JSON cuối cần chứa:
- `epochs_requested`
- `epochs_ran`
- `steps`
- `checkpoint`
- `best_checkpoint`
- `best_metric_name`
- `best_metric_value`
- `stopped_early`
- `final_eval_metrics`

Artifact root bundle sẽ ưu tiên final eval metrics theo checkpoint policy đã chọn.

## 5. Baseline Workflow

`examples/baseline.py` phải:
1. prepare synthetic fixture
2. validate config
3. train baseline
4. run eval với checkpoint vừa train hoặc checkpoint tốt nhất

Script này vẫn là smoke workflow, không được giả làm research benchmark.
