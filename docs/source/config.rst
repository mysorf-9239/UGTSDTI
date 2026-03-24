Configuration Reference
=======================

UGTSDTI uses `Hydra <https://hydra.cc>`_ for hierarchical configuration.
All YAML files live under ``configs/`` and are composed at runtime via the
``defaults`` list in ``configs/default.yaml``.

----

Default Config (``configs/default.yaml``)
------------------------------------------

.. code-block:: yaml

    defaults:
      - _self_
      - model: hybrid_baseline   # configs/model/hybrid_baseline.yaml
      - data: tdc_davis          # configs/data/tdc_davis.yaml
      - trainer: default_trainer # configs/trainer/default_trainer.yaml

    run_name: "ugtsdti_baseline"
    seed: 42
    strict_cudnn: false
    log_level: "INFO"
    output_dir: "./outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}"

    logging:
      wandb_enabled: true

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Key
     - Default
     - Description
   * - ``run_name``
     - ``"ugtsdti_baseline"``
     - Experiment name used in WandB and log output.
   * - ``seed``
     - ``42``
     - Global random seed (Python, NumPy, PyTorch).
   * - ``strict_cudnn``
     - ``false``
     - Set ``torch.backends.cudnn.deterministic = True`` when enabled.
   * - ``log_level``
     - ``"INFO"``
     - Loguru log level (``DEBUG`` / ``INFO`` / ``WARNING`` / ``ERROR``).
   * - ``output_dir``
     - timestamped path
     - Root output directory; Hydra interpolation supported.
   * - ``logging.wandb_enabled``
     - ``true``
     - Enable / disable WandB run initialisation.

----

Model Configs (``configs/model/``)
------------------------------------

Three pre-built model configs are provided:

hybrid_baseline.yaml
~~~~~~~~~~~~~~~~~~~~

Full Teacher–Student–PairGate pipeline.

.. code-block:: yaml

    name: "hybrid_dti"
    params:
      student_cfg:
        name: "baseline_student"
        params:
          hidden_dim: 128
      teacher_cfg:
        name: "baseline_teacher"
        params:
          hidden_dim: 128
          num_drugs: 100003
          num_targets: 100003
      fusion_cfg:
        name: "pairgate_fusion"
        params:
          input_dim: 128
          gate_hidden: 32
          mc_samples: 5    # number of MC-Dropout passes at eval time

only_student.yaml
~~~~~~~~~~~~~~~~~

Inductive-only ablation (no teacher, no gate).

.. code-block:: yaml

    name: "hybrid_dti"
    params:
      student_cfg:
        name: "baseline_student"
        params:
          hidden_dim: 128
      teacher_cfg: null
      fusion_cfg: null

only_teacher.yaml
~~~~~~~~~~~~~~~~~

Transductive-only ablation.

.. code-block:: yaml

    name: "hybrid_dti"
    params:
      student_cfg: null
      teacher_cfg:
        name: "baseline_teacher"
        params:
          hidden_dim: 128
          num_drugs: 100003
          num_targets: 100003
      fusion_cfg: null

**Model config keys**:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Description
   * - ``name``
     - Registry key passed to ``MODELS.build()``.
   * - ``params.student_cfg``
     - Nested config for ``BaselineStudent``; set to ``null`` to disable.
   * - ``params.teacher_cfg``
     - Nested config for ``BaselineTeacher``; set to ``null`` to disable.
   * - ``params.fusion_cfg``
     - Nested config for ``PairGateFusion``; set to ``null`` to disable.
   * - ``params.fusion_cfg.params.mc_samples``
     - Number of MC-Dropout passes. ``0`` disables uncertainty gating.
   * - ``params.fusion_cfg.params.gate_hidden``
     - Hidden dimension of the gate MLP.

----

Data Config (``configs/data/tdc_davis.yaml``)
----------------------------------------------

.. code-block:: yaml

    train:
      name: "tdc_caching_dataset"
      params:
        name: "DAVIS"
        split: "train"
        split_type: "cold_split"
        frac: [0.8, 0.1, 0.1]
        seed: 42

    val:
      name: "tdc_caching_dataset"
      params:
        name: "DAVIS"
        split: "valid"
        split_type: "cold_split"
        frac: [0.8, 0.1, 0.1]
        seed: 42

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Description
   * - ``name``
     - PyTDC dataset name (``"DAVIS"``, ``"KIBA"``, ``"BindingDB_Kd"``).
   * - ``split``
     - Which split to load: ``"train"`` / ``"valid"`` / ``"test"``.
   * - ``split_type``
     - PyTDC split method: ``"cold_split"`` (cold-start) or ``"random_split"``.
   * - ``frac``
     - Train / val / test fractions. Must sum to 1.0.
   * - ``seed``
     - Split seed for reproducibility.

----

Trainer Config (``configs/trainer/default_trainer.yaml``)
----------------------------------------------------------

.. code-block:: yaml

    name: default_trainer
    params:
      epochs: 100
      patience: 15
      val_check_interval: 1
      grad_clip: 1.0
      lr: 0.001
      batch_size: 32
      num_workers: 0
      output_dir: "./outputs/checkpoints"
    loss:
      name: "bce_with_logits"

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Key
     - Default
     - Description
   * - ``params.epochs``
     - ``100``
     - Maximum training epochs.
   * - ``params.patience``
     - ``15``
     - Early-stopping patience (epochs without AUROC improvement).
   * - ``params.val_check_interval``
     - ``1``
     - Run validation every N epochs.
   * - ``params.grad_clip``
     - ``1.0``
     - Max gradient norm for clipping. Set ``0`` to disable.
   * - ``params.lr``
     - ``0.001``
     - Adam learning rate.
   * - ``params.batch_size``
     - ``32``
     - DataLoader batch size.
   * - ``params.num_workers``
     - ``0``
     - DataLoader worker processes.
   * - ``params.output_dir``
     - ``"./outputs/checkpoints"``
     - Directory for ``best_model.pt`` checkpoint.
   * - ``loss.name``
     - ``"bce_with_logits"``
     - Registry key for the loss function.

----

CLI Override Examples
---------------------

Hydra supports inline overrides for any config key:

.. code-block:: bash

    # Change dataset to KIBA
    python -m ugtsdti.main data=tdc_davis data.train.params.name=KIBA data.val.params.name=KIBA

    # Increase MC-Dropout passes
    python -m ugtsdti.main model.params.fusion_cfg.params.mc_samples=10

    # Tune learning rate and batch size
    python -m ugtsdti.main trainer.params.lr=1e-4 trainer.params.batch_size=64

    # Student-only ablation with longer training
    python -m ugtsdti.main model=only_student trainer.params.epochs=200

    # Disable WandB
    python -m ugtsdti.main logging.wandb_enabled=false
