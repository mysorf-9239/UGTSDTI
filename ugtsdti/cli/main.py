"""Command-line entrypoints for train/eval/validate/sweep."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from ugtsdti.config.loader import ConfigLoader
from ugtsdti.config.normalize import ConfigNormalizer
from ugtsdti.config.validate import ConfigValidator
from ugtsdti.core.errors import UGTSDTIError
from ugtsdti.runtime.identity import build_experiment_identity

CommandHandler = Callable[[dict[str, Any], argparse.Namespace], Any]


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(prog="ugtsdti")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command_name in ("train", "eval", "sweep"):
        command = subparsers.add_parser(command_name)
        command.add_argument("config")
        command.add_argument("--config", dest="config_flag")

    validate = subparsers.add_parser("validate")
    validate.add_argument("config", nargs="*")
    validate.add_argument("--config", dest="config_flag", nargs="+")

    return parser


def run_cli(
    argv: list[str] | None = None,
    *,
    stdout: Any = None,
    config_loader: ConfigLoader | None = None,
    validator: ConfigValidator | None = None,
    normalizer: ConfigNormalizer | None = None,
    handlers: dict[str, CommandHandler] | None = None,
) -> int:
    """Run the CLI command and return a process-like exit code."""
    stream = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    loader = config_loader or ConfigLoader()
    validator = validator or ConfigValidator()
    normalizer = normalizer or ConfigNormalizer()
    command_handlers = handlers or {}

    try:
        config_paths = _resolve_config_paths(args)
        if not config_paths:
            raise UGTSDTIError("Missing config path.", stage="cli", component="main", key="config")
        if args.command == "validate":
            for config_path in config_paths:
                raw_cfg = loader.load(Path(config_path))
                validator.validate(raw_cfg)
            stream.write("VALID\n")
            return 0

        raw_cfg = loader.load(Path(config_paths[0]))
        validator.validate(raw_cfg)
        normalized_cfg = normalizer.normalize(raw_cfg).to_dict()
        identity = build_experiment_identity(normalized_cfg)
        stream.write(
            f"run_id={identity.run_id} config_hash={identity.config_hash} "
            f"git_commit={identity.git_commit or 'unknown'}\n"
        )

        if args.command == "sweep":
            targets = resolve_sweep_targets(normalized_cfg)
            stream.write(json.dumps({"targets": targets}, sort_keys=True) + "\n")

        handler = command_handlers.get(args.command)
        if handler is not None:
            handler(normalized_cfg, args)
        return 0
    except UGTSDTIError as exc:
        stream.write(f"ERROR: {exc}\n")
        return 1


def resolve_sweep_targets(cfg: dict[str, Any]) -> list[str]:
    """Resolve normalized sweep parameter targets."""
    sweep_cfg = cfg.get("sweep", {})
    parameters = sweep_cfg.get("parameters", {})

    if isinstance(parameters, dict):
        return sorted(str(target) for target in parameters.keys())
    if isinstance(parameters, list):
        targets = []
        for entry in parameters:
            if isinstance(entry, dict) and "target" in entry:
                targets.append(str(entry["target"]))
        return sorted(targets)
    return []


def _resolve_config_paths(args: argparse.Namespace) -> list[str]:
    explicit = getattr(args, "config_flag", None)
    if explicit:
        if isinstance(explicit, list):
            return [str(path) for path in explicit]
        return [str(explicit)]
    positional = getattr(args, "config", None)
    if positional is None:
        return []
    if isinstance(positional, list):
        return [str(path) for path in positional]
    return [str(positional)]


def main(argv: list[str] | None = None) -> int:
    """Console-script entrypoint."""
    return run_cli(argv)
