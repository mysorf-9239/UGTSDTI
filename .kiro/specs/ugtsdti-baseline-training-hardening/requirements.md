# Đặc Tả Yêu Cầu: UGTSDTI Baseline Training Hardening

## Mục Tiêu

Hoàn thiện baseline reference để có thể chạy train nhiều epoch một cách ổn định, resume được, chọn được best checkpoint, và đánh giá đúng model đã train. Pha này **không** thay đổi kiến trúc baseline graph; chỉ harden training/runtime/reporting surface.

## Ngôn Ngữ Chuẩn Tắc

- `MUST`: bắt buộc
- `SHOULD`: nên có, chỉ được bỏ khi có lý do rõ ràng
- `MAY`: tùy chọn

## Phạm Vi

Pha này bao gồm:
- training loop baseline nhiều epoch
- optimizer / scheduler surface tối thiểu
- gradient clipping và non-finite guards
- best checkpoint / last checkpoint / resume
- final eval dùng đúng checkpoint đã train
- reporting và artifact summary cho long-run training

Pha này không bao gồm:
- thêm teacher/student mới
- thêm KD, uncertainty, learned gate
- thay đổi graph contract cốt lõi

## Yêu Cầu

### REQ-BTH-001 Fixed Pipeline Compatibility
Baseline training hardening MUST giữ nguyên pipeline hiện có:
`batch -> graph -> role binding -> interaction.noop -> decision.identity -> postprocess`.

### REQ-BTH-002 Long-Run Trainability
Baseline MUST train qua nhiều epoch liên tiếp mà không cần bypass framework, và train state MUST được quản lý qua `Trainer`, `PipelineExecutor`, `CheckpointIO`, `ArtifactWriter`.

### REQ-BTH-003 Optimizer Config Surface
Framework MUST hỗ trợ cấu hình optimizer tối thiểu cho baseline gồm:
- `sgd`
- `adam`
- `adamw`

### REQ-BTH-004 Scheduler Config Surface
Framework MUST hỗ trợ scheduler tối thiểu gồm:
- `none`
- `step`
- `plateau`

### REQ-BTH-005 Gradient Safety
Training loop MUST hỗ trợ gradient clipping theo `max_grad_norm`, và MUST fail-closed khi loss hoặc gradients là non-finite nếu config bật strict guard.

### REQ-BTH-006 Resume Integrity
Training loop MUST support resume từ checkpoint bằng cách restore:
- model state
- optimizer state
- scheduler state nếu có
- epoch/step counters
- lineage compatibility checks hiện có

### REQ-BTH-007 Best Checkpoint Tracking
Training loop MUST support theo dõi `best checkpoint` dựa trên validation metric được cấu hình.

### REQ-BTH-008 Early Stopping
Training loop SHOULD support early stopping theo validation metric với `patience` và `min_delta`.

### REQ-BTH-009 Final Evaluation Correctness
Nếu train loop có validation/eval cadence và best checkpoint tracking, final evaluation/reporting MUST có khả năng dùng đúng checkpoint tốt nhất hoặc checkpoint cuối theo policy cấu hình.

### REQ-BTH-010 Baseline Workflow Correctness
User-facing baseline workflow (`examples/baseline.py`, `scripts/baseline.sh`) MUST đánh giá model đã train, không được đánh giá model mới khởi tạo.

### REQ-BTH-011 Reporting Surface
Train summary MUST báo rõ:
- số epoch thực chạy
- số step
- checkpoint cuối
- best checkpoint
- best metric name / value nếu có
- final eval metrics

### REQ-BTH-012 Deterministic Surface Preservation
Các thay đổi mới MUST không làm vỡ deterministic/config-driven behavior hiện có. Resume và best-checkpoint selection MUST explicit, không được suy đoán ngầm.

### REQ-BTH-013 Contract Preservation
Không được bypass state write-once, graph execution, hay direct model call trong trainer.

### REQ-BTH-014 Validation
Config validation MUST reject:
- optimizer type không hỗ trợ
- scheduler type không hỗ trợ
- giá trị epoch/step/patience/clip không hợp lệ
- yêu cầu best/early-stop khi không có eval cadence hợp lệ

### REQ-BTH-015 Verification
Phải có test chứng minh:
- baseline train nhiều epoch vẫn giảm loss
- gradient flow đi qua toàn branch
- resume giữ tiếp tục training được
- best checkpoint được materialize
- baseline workflow eval đúng checkpoint vừa train
