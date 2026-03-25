Configuration Reference
=======================

UGTSDTI uses Hydra with a slot-based composition model.

Default Composition
-------------------

The default entry config is [``configs/default.yaml``] and currently resolves to:

.. code-block:: yaml

    defaults:
      - _self_
      - model: hybrid
      - teacher: baseline
      - student: baseline
      - fusion: ug
      - loss: bce
      - data: tdc_davis_s4
      - trainer: default_trainer

Config Groups
-------------

+---------------------+--------------------------------------------+
| Group               | Purpose                                    |
+=====================+============================================+
| ``model``           | top-level model shell                      |
+---------------------+--------------------------------------------+
| ``teacher``         | teacher plugin                             |
+---------------------+--------------------------------------------+
| ``student``         | student plugin                             |
+---------------------+--------------------------------------------+
| ``fusion``          | fusion plugin                              |
+---------------------+--------------------------------------------+
| ``loss``            | training loss                              |
+---------------------+--------------------------------------------+
| ``data``            | dataset and split protocol                 |
+---------------------+--------------------------------------------+
| ``trainer``         | optimizer/training loop hyperparameters    |
+---------------------+--------------------------------------------+
| ``benchmark``       | experiment matrix for benchmarking         |
+---------------------+--------------------------------------------+

Model Slots
-----------

Preferred usage:

.. code-block:: bash

    conda run -n ugtsdti python -m ugtsdti.main \
        model=hybrid teacher=gcn student=baseline fusion=ug \
        data=tdc_davis_s4 loss=kd

Available slot groups in the current repo:

- ``model=hybrid``
- ``teacher=none|baseline|gcn``
- ``student=none|baseline|esm``
- ``fusion=none|ug``
- ``loss=bce|kd``

Trainer Config
--------------

``configs/trainer/default_trainer.yaml`` defines:

- epochs
- patience
- validation interval
- gradient clipping
- learning rate
- batch size
- checkpoint output path

The trainer now references the top-level loss via:

.. code-block:: yaml

    loss: ${loss}

This keeps ``loss=kd`` and trainer wiring aligned.

Data Config
-----------

Scenario configs are explicit:

- ``tdc_davis_s1``
- ``tdc_davis_s2``
- ``tdc_davis_s3``
- ``tdc_davis_s4``

Typical fields:

- ``split_type``
- ``column_name``
- ``frac``
- ``seed``
- ``scenario_name``
- optional ``cache_dir``

Benchmark Config
----------------

Benchmarking uses explicit experiment specs rather than encoded model aliases.

.. code-block:: yaml

    experiments:
      - name: hybrid_gcn_baseline_ug
        model: hybrid
        teacher: gcn
        student: baseline
        fusion: ug

    losses:
      - bce
      - kd

    data_configs:
      - tdc_davis_s1
      - tdc_davis_s2
      - tdc_davis_s3
      - tdc_davis_s4

CLI Override Examples
---------------------

.. code-block:: bash

    # Change only the loss
    conda run -n ugtsdti python -m ugtsdti.main loss=kd loss.params.alpha=0.3

    # Increase MC-Dropout passes at eval time
    conda run -n ugtsdti python -m ugtsdti.main fusion.params.eval_mc_samples=10

    # Student-only ablation
    conda run -n ugtsdti python -m ugtsdti.main \
        model=hybrid teacher=none student=baseline fusion=none

    # Different scenario
    conda run -n ugtsdti python -m ugtsdti.main data=tdc_davis_s2
