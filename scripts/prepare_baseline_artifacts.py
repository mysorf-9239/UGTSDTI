"""Manual utility for materializing baseline artifacts via the shared bootstrap flow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ugtsdti.config.loader import ConfigLoader  # noqa: E402
from ugtsdti.config.normalize import ConfigNormalizer  # noqa: E402
from ugtsdti.data.bootstrap import DataBootstrapOrchestrator  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare baseline artifacts through the shared bootstrap path.")
    parser.add_argument(
        "--config",
        default=str(ROOT_DIR / "configs" / "profiles" / "baseline_local.yaml"),
        help="Config file containing data.source bootstrap settings.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Optional override for runtime.data_dir before preparing artifacts.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    raw_cfg = ConfigLoader().load(Path(args.config))
    cfg = ConfigNormalizer().normalize(raw_cfg).to_dict()
    if args.data_dir:
        cfg.setdefault("runtime", {})
        cfg["runtime"]["data_dir"] = str(Path(args.data_dir).resolve())
    report = DataBootstrapOrchestrator().ensure_artifacts(
        cfg,
        data_root=cfg.get("runtime", {}).get("data_dir", "data"),
    )
    print(json.dumps(report.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
