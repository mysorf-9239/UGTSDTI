Getting Started
===============

Installation
------------

.. code-block:: bash

    git clone https://github.com/mysorf-9239/UGTSDTI.git
    cd UGTSDTI
    conda env create -f conda-recipes/full.yaml
    conda activate ugtsdti
    pip install -e .

Quick Runs
----------

.. code-block:: bash

    # Student-only baseline on S1
    conda run -n ugtsdti python -m ugtsdti.main \
        model=hybrid teacher=none student=baseline fusion=none \
        data=tdc_davis_s1 loss=bce

    # Teacher-only GCN on S2
    conda run -n ugtsdti python -m ugtsdti.main \
        model=hybrid teacher=gcn student=none fusion=none \
        data=tdc_davis_s2 loss=bce

    # Hybrid GCN + baseline + UG + KD on S4
    conda run -n ugtsdti python -m ugtsdti.main \
        model=hybrid teacher=gcn student=baseline fusion=ug \
        data=tdc_davis_s4 loss=kd loss.params.alpha=0.5

    # Benchmark matrix
    conda run -n ugtsdti python -m ugtsdti.benchmark

    # Smoke scripts
    WANDB_MODE=disabled bash scripts/smoke.sh

Running Tests
-------------

.. code-block:: bash

    conda run -n ugtsdti python -m pytest tests/ -v

The test suite is split by intent:

- ``tests/research/`` checks research invariants
- ``tests/quality/`` checks code correctness and regressions

Codebase Layout
---------------

.. code-block:: text

    ugtsdti/
    ├── experiment/        # experiment composition and runtime orchestration
    ├── data/
    │   ├── datasets/
    │   ├── transforms/
    │   ├── protocols/
    │   └── graph_builder.py
    ├── models/
    ├── losses/
    ├── core/
    ├── main.py
    └── benchmark.py
