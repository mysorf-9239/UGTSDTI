"""Thin benchmark CLI for matrix execution and artifact generation.

Benchmark orchestration, protocol probing, aggregation, and report shaping live
under :mod:`ugtsdti.experiment.benchmarking`. This module stays intentionally
small so the CLI surface remains stable.
"""

import hydra
from omegaconf import DictConfig

from ugtsdti.data.protocols.probe import probe_tdc_split
from ugtsdti.experiment.benchmarking import build_report_rows as _build_report_rows
from ugtsdti.experiment.benchmarking import load_experiment_cfg as _load_experiment_cfg
from ugtsdti.experiment.benchmarking import run_benchmark_matrix
from ugtsdti.experiment.runtime import run_experiment

__all__ = [
    "_build_report_rows",
    "_load_experiment_cfg",
    "probe_tdc_split",
    "run_benchmark_matrix",
    "run_experiment",
]


@hydra.main(config_path="../configs/benchmark", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the configured benchmark matrix and write benchmark artifacts."""
    run_benchmark_matrix(cfg)


if __name__ == "__main__":
    main()
