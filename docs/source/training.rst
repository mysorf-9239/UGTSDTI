Training
========

Losses
------

Two losses are first-class in the current repo:

``BCELoss``
  standard supervised binary classification loss

``KDLoss``
  task BCE plus teacher-student distillation term

.. math::

    \mathcal{L} = (1 - \alpha)\,\mathcal{L}_{task} + \alpha\,\mathcal{L}_{distill}

where:

- :math:`\mathcal{L}_{task}` uses fused logits
- :math:`\mathcal{L}_{distill}` uses student and teacher logits
- :math:`\alpha` is the KD weight from config, not the UG gate weight

Training Loop
-------------

.. mermaid::

    flowchart TD
        Start([Start]) --> Compose["Resolve config"]
        Compose --> Build["Build data / model / loss / trainer"]
        Build --> Train["Trainer.fit()"]
        Train --> Val["Validation"]
        Val --> Check["Early stopping check"]
        Check --> Best["Load best checkpoint"]
        Best --> Test["Optional test evaluation"]

Research Logging
----------------

Evaluation currently logs:

- fused metrics
- student-only branch metrics
- teacher-only branch metrics
- `gate_alpha_*`
- `student_var_*`
- `teacher_var_*`
- `unk_drug_rate`
- `unk_target_rate`
- split audit metrics

This is important because the repo is intended for research interpretation, not only for final AUROC.

Running Experiments
-------------------

.. code-block:: bash

    # Hybrid GCN + UG + BCE on S4
    conda run -n ugtsdti python -m ugtsdti.main \
        model=hybrid teacher=gcn student=baseline fusion=ug \
        data=tdc_davis_s4 loss=bce

    # Hybrid GCN + UG + KD on S4
    conda run -n ugtsdti python -m ugtsdti.main \
        model=hybrid teacher=gcn student=baseline fusion=ug \
        data=tdc_davis_s4 loss=kd loss.params.alpha=0.3

    # Teacher-only ablation
    conda run -n ugtsdti python -m ugtsdti.main \
        model=hybrid teacher=gcn student=none fusion=none \
        data=tdc_davis_s2 loss=bce

Benchmarking
------------

Use the benchmark runner to execute experiment matrices:

.. code-block:: bash

    conda run -n ugtsdti python -m ugtsdti.benchmark

Artifacts include raw summaries, aggregated summaries, tidy reports, and protocol metadata.

API Reference
-------------

.. autoclass:: ugtsdti.core.trainer.Trainer
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: ugtsdti.losses.bce.BCELoss
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: ugtsdti.losses.kd.KDLoss
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
