"""
Only Student — ESM-2 protein encoder, BCE loss, DAVIS dataset.

Slot convention: <teacher>.<student>.<fusion>.<loss>.<data>
  teacher : _ (none)
  student : esm
  fusion  : _ (none)
  loss    : bce
  data    : davis

Usage:
    conda run -n ugtsdti python -m examples._.esm.bce._.davis.train
"""

import subprocess
import sys

CMD = [
    "python",
    "-m",
    "ugtsdti.main",
    "model=hybrid",
    "teacher=none",
    "student=esm",
    "fusion=none",
    "loss=bce",
    "data=tdc_davis_s4",
]

if __name__ == "__main__":
    sys.exit(subprocess.call(CMD))
