# Examples

Các ví dụ ở đây phục vụ 2 mục tiêu:

- chỉ cách chạy baseline reference nhanh trên artifact fixture nhỏ;
- chỉ cách gọi framework qua CLI hoặc API mà không phải đoán config/runtime layout.

Flow tối thiểu:

1. Tạo fixture artifacts:

```bash
python /Users/mysorf/PythonProject/Bioinformatics/UGTSDTI/scripts/prepare_baseline_fixture.py --data-dir /Users/mysorf/PythonProject/Bioinformatics/UGTSDTI/data
```

2. Validate config:

```bash
bash /Users/mysorf/PythonProject/Bioinformatics/UGTSDTI/scripts/validate_baseline.sh
```

3. Train baseline:

```bash
bash /Users/mysorf/PythonProject/Bioinformatics/UGTSDTI/scripts/train_baseline.sh
```

4. Eval baseline:

```bash
bash /Users/mysorf/PythonProject/Bioinformatics/UGTSDTI/scripts/eval_baseline.sh
```

Lưu ý:

- fixture do `prepare_baseline_fixture.py` tạo chỉ để smoke/local verification;
- không dùng fixture này để báo cáo kết quả nghiên cứu.
