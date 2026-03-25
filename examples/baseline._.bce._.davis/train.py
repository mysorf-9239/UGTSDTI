"""
Only Teacher — baseline embedding, BCE loss, DAVIS dataset.

Slot convention: <teacher>.<student>.<fusion>.<loss>.<data>
  teacher : baseline
  student : _ (none)
  fusion  : _ (none)
  loss    : bce
  data    : davis

Usage:
    conda run -n ugtsdti python -m examples.baseline._.bce._.davis.train
"""

import subprocess
import sys

CMD = [
    "python",
    "-m",
    "ugtsdti.main",
    "model=hybrid",
    "teacher=baseline",
    "student=none",
    "fusion=none",
    "loss=bce",
    "data=tdc_davis_s1",
]

if __name__ == "__main__":
    sys.exit(subprocess.call(CMD))
