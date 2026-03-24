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
      - model: baseline.baseline.ug  # configs/model/baseline.baseline.ug.yaml
      - data: tdc_davis              # configs/data/tdc_davis.yaml
      - trainer: default_trainer     # configs/trainer/default_trainer.yaml

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

Naming convention: ``<teacher>.<student>.<fusion>.yaml``

Loss is NOT encoded in the config name — it is overridden via ``trainer.loss.name``.

**Available configs:**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Config file
     - Description
   * - ``_.baseline._.yaml``
     - Only student (baseline) — no teacher, no fusion
   * - ``baseline._._.yaml``
     - Only teacher (dummy embedding) — no student, no fusion
   * - ``gcn._._.yaml``
     - Only teacher (GCN) — no student, no fusion
   * - ``_.esm._.yaml``
     - Only student (ESM) — no teacher, no fusion
   * - ``baseline.baseline.ug.yaml``
     - Hybrid: dummy teacher + baseline student + UG fusion
   * - ``gcn.baseline.ug.yaml``
     - Hybrid: GCN teacher + baseline student + UG fusion ← **main**

**Example — gcn.baseline.ug.yaml:**

.. code-block:: yaml

    name: "hybrid_dti"
    params:
      student_cfg:
        name: "baseline_student"
        params:
          hidden_dim: 128
      teacher_cfg:
        name: "gcn_teacher"
        params:
          drug_feat_dim: 2048
          protein_feat_dim: 8000
          hidden_dim: 128
          num_layers: 2
          dropout: 0.3
      fusion_cfg:
        name: "ug_fusion"
        params:
          gate_hidden: 32
          mc_samples: 5

**Model config keys**:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Key
     - Description
   * - ``name``
     - Registry key passed to ``MODELS.build()``.
   * - ``params.student_cfg``
     - Nested config for student model; set to ``null`` to disable.
   * - ``params.teacher_cfg``
     - Nested config for teacher model; set to ``null`` to disable.
   * - ``params.fusion_cfg``
     - Nested config for ``UncertaintyGatedFusion``; set to ``null`` to disable.
   * - ``params.fusion_cfg.params.mc_samples``
     - Number of MC-Dropout passes at eval time. ``0`` disables uncertainty gating.
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
     - PyTDC split method: ``"cold_split"`` (S4) or ``"random_split"`` (S1).
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
      name: "bce"

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
     - ``"bce"``
     - Registry key for the loss function (``"bce"`` or ``"kd"``).
   * - ``loss.alpha``
     - ``0.5``
     - KD distillation weight (only used when ``loss.name="kd"``).

----

CLI Override Examples
---------------------

.. code-block:: bash

    # Change dataset to KIBA
    conda run -n ugtsdti python -m ugtsdti.main data=tdc_davis \
        data.train.params.name=KIBA data.val.params.name=KIBA

    # Increase MC-Dropout passes
    conda run -n ugtsdti python -m ugtsdti.main \
        model.params.fusion_cfg.params.mc_samples=10

    # Tune learning rate and batch size
    conda run -n ugtsdti python -m ugtsdti.main \
        trainer.params.lr=1e-4 trainer.params.batch_size=64

    # Student-only ablation with longer training
    conda run -n ugtsdti python -m ugtsdti.main \
        model=_.baseline._ trainer.params.epochs=200

    # KD loss with custom alpha
    conda run -n ugtsdti python -m ugtsdti.main \
        model=gcn.baseline.ug trainer.loss.name=kd "+trainer.loss.alpha=0.3"

    # Disable WandB
    WANDB_MODE=disabled conda run -n ugtsdti python -m ugtsdti.main model=gcn.baseline.ug
