"""
Only Teacher — GCN encoder, BCE loss, no gate, DAVIS dataset.

Slot convention: <teacher>.<student>.<loss>.<gate>.<data>
  teacher : gcn
  student : _ (none)
  loss    : bce
  gate    : _ (none)
  data    : davis

Usage:
    conda run -n ugtsdti python -m examples.gcn._.bce._.davis.train
"""

import subprocess
import sys

CMD = [
    "python",
    "-m",
    "ugtsdti.main",
    "model=gcn._._",
    "data=tdc_davis",
    "trainer=default_trainer",
    "trainer.loss.name=bce",
]

if __name__ == "__main__":
    sys.exit(subprocess.call(CMD))
