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
    python -m ugtsdti.main model=only_student data=tdc_davis

    # Teacher-only baseline
    python -m ugtsdti.main model=only_teacher data=tdc_davis

    # Hybrid: student + teacher + PairGate (BCE loss)
    python -m ugtsdti.main model=hybrid_baseline data=tdc_davis

    # Hybrid with Knowledge Distillation loss
    python -m ugtsdti.main model=hybrid_baseline data=tdc_davis \
        trainer.params.loss.name=kd_dual_loss trainer.params.loss.alpha=0.5

    # Full 4-mode ablation suite (WandB disabled)
    bash scripts/run_baselines.sh

Running tests
-------------

.. code-block:: bash

    conda run -n ugtsdti python -m pytest tests/ -v

All 138 tests should pass. The suite covers transforms, models, losses, metrics, registry,
MC-Dropout consistency, and PairGate fusion.

Project layout
--------------

.. code-block:: text

    UGTSDTI/
    ├── configs/                  # Hydra YAML configs
    │   ├── default.yaml
    │   ├── model/                # only_student, only_teacher, hybrid_baseline
    │   ├── data/                 # tdc_davis, davis_s4
    │   └── trainer/              # default_trainer
    ├── ugtsdti/
    │   ├── main.py               # Entry point (@hydra.main)
    │   ├── core/                 # FROZEN: Registry, Trainer, Metrics
    │   ├── data/                 # TDCCachingDataset, transforms
    │   ├── models/               # HybridDTIModel, student/, teacher/, fusion/
    │   ├── losses/               # BCEWithLogitsLossWrapper, KDDualLoss
    │   └── utils/                # logger, seed
    ├── tests/                    # pytest suite (138 tests)
    ├── docs/                     # This documentation
    └── .agent/                   # AI context, task tracking, research notes
