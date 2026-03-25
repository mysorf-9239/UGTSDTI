"""Public data-layer exports and registry bootstrap imports.

Importing :mod:`ugtsdti.data` is enough to register dataset plugins and expose
split-protocol helpers used by runtime audit and benchmark probing.
"""

from .datasets.tdc_dataset import TDCCachingDataset, normalize_tdc_column_name, normalize_tdc_split_method
from .protocols import probe_tdc_split, summarize_split_frames, summarize_split_overlap, write_probe_summary

__all__ = [
    "TDCCachingDataset",
    "normalize_tdc_column_name",
    "normalize_tdc_split_method",
    "summarize_split_overlap",
    "probe_tdc_split",
    "summarize_split_frames",
    "write_probe_summary",
]
