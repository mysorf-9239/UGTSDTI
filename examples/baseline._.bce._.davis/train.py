"""
Only Teacher — baseline embedding (dummy), BCE loss, no gate, DAVIS dataset.

Slot convention: <teacher>.<student>.<loss>.<gate>.<data>
  teacher : baseline
  student : _ (none)
  loss    : bce
  gate    : _ (none)
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
    "model=baseline._._",
    "data=tdc_davis",
    "trainer=default_trainer",
    "trainer.loss.name=bce",
]

if __name__ == "__main__":
    sys.exit(subprocess.call(CMD))
