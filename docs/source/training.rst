Training
========

UGTSDTI uses a standard supervised training loop augmented with
**Knowledge Distillation (KD)** and **early stopping** on validation AUROC.
All hyperparameters are controlled via Hydra YAML configs.

----

Loss Functions
--------------

Two loss functions are available, both registered in the ``LOSSES`` registry:

BCELoss (registry: ``"bce"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wraps ``nn.BCEWithLogitsLoss`` to accept the standard model output dict.
Default loss for ablation and single-branch modes.

.. autoclass:: ugtsdti.losses.bce.BCELoss
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

KDLoss (registry: ``"kd"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Knowledge Distillation loss — convex combination of task loss and distillation loss:

.. math::

    \mathcal{L} = (1 - \alpha)\,\mathcal{L}_{\text{BCE}}(\hat{y},\, y)
                + \alpha\,\mathcal{L}_{\text{MSE}}(\ell_s,\, \ell_t)

where:

- :math:`\hat{y}` — fused logit from UG fusion
- :math:`y` — ground-truth binary label
- :math:`\ell_s` — student logit
- :math:`\ell_t` — teacher logit
- :math:`\alpha \in [0, 1]` — distillation weight (default ``0.5``)

When only one branch is active (student-only or teacher-only mode),
``KDLoss`` falls back to plain BCE.

.. autoclass:: ugtsdti.losses.kd.KDLoss
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

----

Training Loop
-------------

.. mermaid::

    flowchart TD
        Start([Start]) --> Epoch[For each epoch]
        Epoch --> Train[_train_epoch\nforward + backward + clip_grad]
        Train --> ValCheck{epoch % val_check_interval == 0?}
        ValCheck -->|Yes| Eval[evaluate\nAUROC / AUPRC / F1 / MSE / CI]
        ValCheck -->|No| Next
        Eval --> ES{AUROC improved?}
        ES -->|Yes| Save[Save best_model.pt\nreset patience counter]
        ES -->|No| Inc[Increment no_improve_epochs]
        Save --> Next[Next epoch]
        Inc --> Patience{no_improve_epochs >= patience?}
        Patience -->|Yes| Stop([Early Stop])
        Patience -->|No| Next

**Key Trainer behaviours**:

- Gradient clipping: ``torch.nn.utils.clip_grad_norm_`` at ``grad_clip`` (default 1.0)
- Checkpoint: saves ``best_model.pt`` whenever validation AUROC improves
- WandB logging: all train/val metrics + LR per epoch (optional)

.. autoclass:: ugtsdti.core.trainer.Trainer
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

----

Evaluation Metrics
------------------

All metrics are computed by :func:`ugtsdti.core.metrics.compute_dti_metrics`
at the end of each validation epoch.

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Metric
     - Range
     - Notes
   * - AUROC
     - [0, 1] ↑
     - Area under ROC curve. Primary early-stopping target.
   * - AUPRC
     - [0, 1] ↑
     - Area under Precision-Recall curve. Better for imbalanced sets.
   * - F1
     - [0, 1] ↑
     - Binary F1 at threshold 0.5 on predicted probability.
   * - MSE
     - [0, ∞) ↓
     - Mean squared error on raw affinity values.
   * - CI
     - [0, 1] ↑
     - Concordance Index — fraction of correctly ranked pairs.

Continuous affinity values (e.g., pKd from DAVIS) are binarised at
``affinity_threshold`` (default **7.0**, corresponding to Kd ≤ 100 nM)
for classification metrics; raw values are used for MSE and CI.

.. autofunction:: ugtsdti.core.metrics.compute_dti_metrics
   :noindex:

----

Running an Experiment
---------------------

.. code-block:: bash

    # Hybrid GCN + UG fusion (default)
    conda run -n ugtsdti python -m ugtsdti.main model=gcn.baseline.ug data=tdc_davis

    # Student-only ablation
    conda run -n ugtsdti python -m ugtsdti.main model=_.baseline._ data=tdc_davis

    # Override hyperparameters inline
    conda run -n ugtsdti python -m ugtsdti.main trainer.params.lr=5e-4 trainer.params.epochs=50

    # KD loss with custom alpha
    conda run -n ugtsdti python -m ugtsdti.main model=gcn.baseline.ug \
        trainer.loss.name=kd "+trainer.loss.alpha=0.3"

    # Disable WandB
    WANDB_MODE=disabled conda run -n ugtsdti python -m ugtsdti.main model=gcn.baseline.ug data=tdc_davis

See :doc:`config` for the full list of configurable keys.
