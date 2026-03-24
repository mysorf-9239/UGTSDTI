# scripts/

Shell scripts để chạy training. Mỗi script tương ứng 1-1 với một thư mục trong `examples/`.

## Naming Convention

```
<teacher>.<student>.<fusion>.<loss>.<data>.sh
```

Cùng convention với `examples/` — đọc tên script là biết ngay đang chạy combo nào.

> `_` = slot không dùng | `smoke.sh` = special script chạy tất cả combo nhanh

## Danh sách hiện tại

| Script | Model config | Loss | Mô tả |
|--------|-------------|------|-------|
| `smoke.sh` | tất cả | — | Smoke test 2 epochs, `WANDB_MODE=disabled` |
| `_.baseline.bce._.davis.sh` | `_.baseline._` | `bce` | Only student baseline |
| `baseline._.bce._.davis.sh` | `baseline._._` | `bce` | Only teacher dummy |
| `gcn._.bce._.davis.sh` | `gcn._._` | `bce` | Only teacher GCN |
| `baseline.baseline.bce.ug.davis.sh` | `baseline.baseline.ug` | `bce` | Hybrid baseline + UG + BCE |
| `baseline.baseline.kd.ug.davis.sh` | `baseline.baseline.ug` | `kd` | Hybrid baseline + UG + KD |
| `gcn.baseline.bce.ug.davis.sh` | `gcn.baseline.ug` | `bce` | Hybrid GCN + UG + BCE |
| `gcn.baseline.kd.ug.davis.sh` | `gcn.baseline.ug` | `kd` | Hybrid GCN + UG + KD |

## Cách chạy

```bash
# Chạy một script cụ thể
bash scripts/gcn.baseline.kd.ug.davis.sh

# Override epochs
EPOCHS=50 bash scripts/gcn.baseline.kd.ug.davis.sh

# Smoke test tất cả (2 epochs mỗi combo)
WANDB_MODE=disabled bash scripts/smoke.sh
```

## Thêm script mới

1. Copy script gần nhất
2. Đổi tên theo convention: `<teacher>.<student>.<fusion>.<loss>.<data>.sh`
3. Sửa `model=`, `trainer.loss.name=`, `run_name=`
4. Tạo `examples/<same_name>/train.py` tương ứng

## Ghi chú

- Hiện tại chạy local. Để chạy trên HPC/Kaggle, thêm SLURM header hoặc Kaggle notebook wrapper sau.
- `EPOCHS` env var override số epochs (default 100).
- `WANDB_MODE=disabled` để tắt WandB logging khi test nhanh.
