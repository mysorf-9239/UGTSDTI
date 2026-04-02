"""Helper: generate temp training config for train_flow1_full.sh"""
import sys
import yaml
from pathlib import Path

(
    config_base, out_path, data_dir, artifacts_dir, checkpoint_dir,
    batch_size, seed, device, wandb_mode, wandb_project,
    epochs, patience,
) = sys.argv[1:]

config = yaml.safe_load(Path(config_base).read_text(encoding="utf-8"))

config["runtime"] = {
    "data_dir":       data_dir,
    "artifacts_dir":  artifacts_dir,
    "checkpoint_dir": checkpoint_dir,
    "batch_size":     int(batch_size),
    "seed":           int(seed),
    "device":         device,
    "deterministic":  True,
    "precision":      "fp32",
    "num_workers":    0,
}
config["logging"] = {
    "backend":       "wandb",
    "wandb_mode":    wandb_mode,
    "wandb_project": wandb_project,
}
loop = dict(config.get("training", {}).get("loop", {}))
loop.update({
    "epochs":                  int(epochs),
    "checkpoint_every_epochs": max(1, int(epochs) // 10),
    "eval_every_epochs":       1,
    "summary_every_steps":     1,
    "eval_partition":          "test",
    "early_stopping": {
        "enabled":   True,
        "patience":  int(patience),
        "min_delta": 0.001,
    },
})
config["training"]["loop"] = loop

Path(out_path).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
print(f"[config] Written to {out_path}")
