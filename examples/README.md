# examples/

Mỗi thư mục là một experiment configuration hoàn chỉnh, chứa `train.py` để chạy trực tiếp.

## Naming Convention

```
<teacher>.<student>.<fusion>.<loss>.<data>/train.py
```

| Slot | Mô tả | Giá trị hiện tại |
|------|-------|-----------------|
| `teacher` | Teacher model | `baseline`, `gcn` |
| `student` | Student model | `baseline`, `esm` |
| `fusion` | Fusion module | `ug` (Uncertainty-Gated) |
| `loss` | Loss function | `bce`, `kd` |
| `data` | Dataset | `davis` |

> `_` = slot không dùng (only-teacher hoặc only-student mode)

## Ví dụ đọc tên

| Thư mục | Ý nghĩa |
|---------|---------|
| `_.baseline.bce._.davis` | Only student=baseline, BCE loss, DAVIS |
| `gcn._.bce._.davis` | Only teacher=GCN, BCE loss, DAVIS |
| `gcn.baseline.ug.kd.davis` | Hybrid GCN+baseline, UG fusion, KD loss, DAVIS |

## Danh sách hiện tại

```
_.baseline.bce._.davis/              # only student (baseline)
baseline._.bce._.davis/              # only teacher (dummy embedding)
gcn._.bce._.davis/                   # only teacher (GCN)
_.esm.bce._.davis/                   # only student (ESM)
baseline.baseline.bce.ug.davis/      # hybrid baseline + UG + BCE
baseline.baseline.kd.ug.davis/       # hybrid baseline + UG + KD
gcn.baseline.bce.ug.davis/           # hybrid GCN + UG + BCE
gcn.baseline.kd.ug.davis/            # hybrid GCN + UG + KD  ← main experiment
```

## Cách chạy

```bash
conda run -n ugtsdti python -m examples.gcn.baseline.kd.ug.davis.train

# Override epochs
EPOCHS=50 conda run -n ugtsdti python -m examples.gcn.baseline.kd.ug.davis.train
```

## Thêm experiment mới

1. Tạo thư mục theo convention: `<teacher>.<student>.<fusion>.<loss>.<data>/`
2. Copy `train.py` từ combo gần nhất, sửa CMD
3. Tạo script tương ứng trong `scripts/`
