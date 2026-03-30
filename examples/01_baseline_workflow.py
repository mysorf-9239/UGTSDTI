"""Programmatic example for the baseline reference workflow."""

from __future__ import annotations

from pathlib import Path

from ugtsdti.cli.main import run_cli


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "baseline_reference.yaml"

    print("1. Validate baseline config")
    if run_cli(["validate", str(config_path)]) != 0:
        return 1

    print("2. Train baseline")
    if run_cli(["train", str(config_path)]) != 0:
        return 1

    print("3. Eval baseline")
    return run_cli(["eval", str(config_path)])


if __name__ == "__main__":
    raise SystemExit(main())
