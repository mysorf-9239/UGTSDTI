# Kế Hoạch Triển Khai: UGTSDTI Baseline Training Hardening

## Tổng Quan

Triển khai theo thứ tự:

1. khóa config surface cho long-run baseline training
2. harden optimizer / scheduler / safety guards
3. thêm resume + best checkpoint + early stopping
4. sửa baseline workflow để eval đúng model đã train
5. chứng minh baseline đủ bền để làm nền cho model phức tạp hơn

Có 4 checkpoint:

1. Config/training surface mới đã hợp lệ và builder path tạo được optimizer/scheduler
2. Long-run train step có safety guards và scheduler path ổn
3. Resume + best checkpoint + early stopping chạy đúng
4. Baseline workflow train/eval đúng checkpoint và reporting hoàn chỉnh

## Tasks

- [x] 1. Audit baseline training hardening fit với runtime hiện tại
  - [x] 1.1 Rà lại `Trainer`, `CLI`, `CheckpointIO`, `ArtifactWriter`, `baseline.py`
  - [x] 1.2 Chốt minimal config surface cho optimizer/scheduler/loop
  - [x] 1.3 Chốt final checkpoint selection policy (`last` vs `best`)

- [x] 2. Mở rộng config defaults và validation cho training hardening
  - [x] 2.1 Thêm defaults cho `training.optimizer`, `training.scheduler`, `training.loop`
  - [x] 2.2 Validate supported optimizer/scheduler types
  - [x] 2.3 Validate `epochs`, `checkpoint_every_epochs`, `eval_every_epochs`, `max_grad_norm`, early-stopping settings
  - [x] 2.4 Reject config đòi `best`/early stopping khi không có eval cadence hợp lệ

- [x] 3. Implement optimizer / scheduler builders
  - [x] 3.1 Hỗ trợ `sgd`, `adam`, `adamw`
  - [x] 3.2 Hỗ trợ `none`, `step`, `plateau`
  - [x] 3.3 Thêm unit tests cho builder path

- [x] 4. Harden `Trainer.step()` cho long-run stability
  - [x] 4.1 Fail-closed với non-finite loss nếu guard bật
  - [x] 4.2 Add optional gradient clipping
  - [x] 4.3 Fail-closed với non-finite gradients nếu guard bật
  - [x] 4.4 Ghi thêm grad norm vào logged metrics nếu available

- [x] 5. Checkpoint 1 — Long-run train step foundation complete
  - Verify:
    - config surface pass validator/normalizer
    - optimizer/scheduler build được
    - step có safety guards

- [x] 6. Implement resume path cho train command
  - [x] 6.1 Nếu `runtime.checkpoint_path` có mặt, restore model/optimizer/scheduler/counters
  - [x] 6.2 Resume path phải fail-closed khi checkpoint không tương thích
  - [x] 6.3 Thêm integration tests cho resume training

- [x] 7. Implement best checkpoint tracking
  - [x] 7.1 Theo dõi best metric theo `training.loop.best_metric`
  - [x] 7.2 Materialize `best_checkpoint`
  - [x] 7.3 Thêm tests cho best checkpoint selection

- [x] 8. Implement early stopping
  - [x] 8.1 Support `enabled`, `patience`, `min_delta`
  - [x] 8.2 Early stop chỉ dựa trên validation metric
  - [x] 8.3 Ghi `stopped_early`, `epochs_ran` vào summary
  - [x] 8.4 Thêm tests

- [x] 9. Checkpoint 2 — Resume / best / early-stop complete
  - Verify:
    - resume chạy được
    - best checkpoint được chọn đúng
    - early stopping hoạt động đúng

- [x] 10. Sửa final eval policy và reporting
  - [x] 10.1 Cuối train, final eval dùng checkpoint theo policy `select_checkpoint`
  - [x] 10.2 Summary JSON chứa `best_checkpoint`, `best_metric_name`, `best_metric_value`, `epochs_ran`
  - [x] 10.3 Artifact final bundle phản ánh final eval đúng checkpoint đã chọn

- [x] 11. Sửa baseline workflow public entrypoints
  - [x] 11.1 `examples/baseline.py` eval đúng checkpoint vừa train
  - [x] 11.2 `scripts/baseline.sh` giữ flow validate -> train -> eval
  - [x] 11.3 Nếu cần, set config smoke thân thiện hơn cho nhiều epoch

- [x] 12. Prove baseline long-run stability
  - [x] 12.1 Viết test baseline nhiều epoch với scheduler/clip path
  - [x] 12.2 Viết test overfit fixture vẫn hội tụ
  - [x] 12.3 Viết test final eval summary không lệch checkpoint

- [x] 13. Checkpoint 3 — Baseline long-run workflow complete
  - Verify:
    - baseline train nhiều epoch ổn định
    - eval dùng đúng trained checkpoint
    - reporting/summary hoàn chỉnh

- [x] 14. Sync docs / traceability
  - [x] 14.1 Sync `.docs/traceability.md`
  - [x] 14.2 Sync baseline collateral nếu surface config đổi

- [x] 15. Checkpoint 4 — Baseline ready for more complex models
  - Verify:
    - baseline long-run train loop ổn
    - checkpoint / resume / best selection ổn
    - smoke workflow user-facing không gây hiểu sai
