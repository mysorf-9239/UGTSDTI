Getting Started
===============

Installation
------------

Clone the repository and create the conda environment:

.. code-block:: bash

    git clone https://github.com/mysorf-9239/UGTSDTI.git
    cd UGTSDTI
    conda env create -f conda-recipes/full.yaml
    conda activate ugtsdti
    pip install -e .

**Stack:** Python 3.10 · PyTorch 2.1+ · PyTorch Geometric · Hydra-core · WandB · PyTDC ·
RDKit · HuggingFace Transformers · Loguru

Quick runs
----------

All commands use `Hydra <https://hydra.cc/>`_ for config composition. Parameters can be
overridden at the command line.

.. code-block:: bash

    # Student-only baseline
    conda run -n ugtsdti python -m ugtsdti.main model=_.baseline._ data=tdc_davis

    # Teacher-only baseline (GCN)
    conda run -n ugtsdti python -m ugtsdti.main model=gcn._._ data=tdc_davis

    # Hybrid: GCN teacher + baseline student + UG fusion (BCE loss)
    conda run -n ugtsdti python -m ugtsdti.main model=gcn.baseline.ug data=tdc_davis

    # Hybrid with Knowledge Distillation loss
    conda run -n ugtsdti python -m ugtsdti.main model=gcn.baseline.ug data=tdc_davis \
        trainer.loss.name=kd trainer.loss.alpha=0.5

    # Smoke test all combos (WandB disabled)
    WANDB_MODE=disabled bash scripts/smoke.sh

Running tests
-------------

.. code-block:: bash

    conda run -n ugtsdti python -m pytest tests/ -v

All 155 tests should pass. The suite covers transforms, models, losses, metrics, registry,
MC-Dropout consistency, and UG fusion.

Project layout
--------------

.. code-block:: text

    UGTSDTI/
    ├── configs/                  # Hydra YAML configs
    │   ├── default.yaml
    │   ├── model/                # <teacher>.<student>.<fusion>.yaml
    │   ├── data/                 # tdc_davis
    │   └── trainer/              # default_trainer
    ├── ugtsdti/
    │   ├── main.py               # Entry point (@hydra.main)
    │   ├── core/                 # FROZEN: Registry, Trainer, Metrics
    │   ├── data/                 # TDCCachingDataset, transforms
    │   ├── models/               # HybridDTIModel, student/, teacher/, fusion/
    │   ├── losses/               # BCELoss (bce), KDLoss (kd)
    │   └── utils/                # logger, seed
    ├── examples/                 # Per-combo train.py (<teacher>.<student>.<fusion>.<loss>.<data>/)
    ├── scripts/                  # Shell scripts (<teacher>.<student>.<fusion>.<loss>.<data>.sh)
    ├── tests/                    # pytest suite (155 tests)
    └── .agent/                   # AI context, task tracking, research notes
