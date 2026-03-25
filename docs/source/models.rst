Models
======

Model Stack
-----------

UGTSDTI is organized around one shell model and three slot types:

- shell: ``HybridDTIModel``
- teacher plugins
- student plugins
- fusion plugins

.. mermaid::

    flowchart LR
        Hybrid["HybridDTIModel"] --> Teacher["Teacher Plugin"]
        Hybrid --> Student["Student Plugin"]
        Hybrid --> Fusion["Fusion Plugin"]

Teacher Plugins
---------------

``baseline_teacher``
  Embedding-based transductive baseline.

``gcn_teacher``
  Graph-based teacher over DD/PP similarity graphs.

Student Plugins
---------------

``baseline_student``
  Lightweight inductive baseline from raw drug/target inputs.

``esm_student``
  Alternative student branch using a pretrained protein model setup.

Fusion Plugins
--------------

``ug_fusion``
  Uncertainty-gated fusion.

The gate uses:

.. math::

    \alpha = \sigma(\mathrm{MLP}([var_s, var_t]))

.. math::

    \ell_{fused} = \alpha \cdot \ell_t + (1 - \alpha) \cdot \ell_s

MC-Dropout Behavior
-------------------

Current code supports separate MC budgets:

- ``train_mc_samples``
- ``eval_mc_samples``

This means uncertainty can be used in both train and eval, with different compute budgets.

Output Schema
-------------

All model modes now return a standardized dict:

- ``logits``
- ``student_logits``
- ``teacher_logits``
- ``gate_alpha``
- ``student_var``
- ``teacher_var``

Non-applicable fields are ``None`` in single-branch modes.

Ablation Modes
--------------

.. mermaid::

    flowchart LR
        A["HybridDTIModel"] --> B{teacher / student / fusion}
        B --> C["student-only"]
        B --> D["teacher-only"]
        B --> E["hybrid"]

API Reference
-------------

.. autoclass:: ugtsdti.models.hybrid.HybridDTIModel
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: ugtsdti.models.student.baseline.BaselineStudent
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: ugtsdti.models.student.plm.ESMProteinStudent
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: ugtsdti.models.teacher.baseline.BaselineTeacher
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: ugtsdti.models.teacher.gcn_teacher.GCNTeacher
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: ugtsdti.models.fusion.ug.UncertaintyGatedFusion
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
