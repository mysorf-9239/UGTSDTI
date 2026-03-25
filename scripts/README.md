# scripts/

Shell scripts cho các ablation canonical, dùng naming mô tả theo slot và scenario.

## Nguyên tắc

- Không encode kiến trúc vào chuỗi compound khó đọc
- Tên script nên cho biết rõ:
  - branch nào đang bật
  - loss nào dùng
  - dataset/scenario nào chạy
- Command bên trong script phải dùng slot-based CLI:
  - `model=hybrid`
  - `teacher=...`
  - `student=...`
  - `fusion=...`
  - `loss=...`
  - `data=...`

## Danh sách hiện tại

| Script | Scenario | Mô tả |
|--------|----------|-------|
| `student_baseline_bce_davis_s1.sh` | `S1` | Student-only baseline |
| `teacher_baseline_bce_davis_s1.sh` | `S1` | Teacher-only baseline embedding |
| `teacher_gcn_bce_davis_s2.sh` | `S2` | Teacher-only GCN |
| `hybrid_baseline_baseline_ug_bce_davis_s4.sh` | `S4` | Hybrid baseline teacher + baseline student + UG |
| `hybrid_gcn_baseline_ug_bce_davis_s4.sh` | `S4` | Hybrid GCN teacher + baseline student + UG |
| `hybrid_gcn_baseline_ug_kd_davis_s4.sh` | `S4` | Hybrid GCN teacher + baseline student + UG + KD |
| `smoke.sh` | mixed | Smoke test tập canonical configs |

## Cách chạy

```bash
bash scripts/hybrid_gcn_baseline_ug_kd_davis_s4.sh
EPOCHS=20 bash scripts/hybrid_gcn_baseline_ug_kd_davis_s4.sh
WANDB_MODE=disabled bash scripts/smoke.sh
```

## Ghi chú

1. Tất cả script giả định Conda env là `ugtsdti`.
2. Muốn tạo script mới, sửa các slot Hydra thay vì tạo alias config riêng cho từng tổ hợp.
