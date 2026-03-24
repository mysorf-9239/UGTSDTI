Models & Uncertainty Gating
===========================

UGTSDTI's model hierarchy is built around three cooperating components:
**BaselineStudent** (inductive), **BaselineTeacher** (transductive), and
**PairGateFusion** (uncertainty-aware blending). The top-level
**HybridDTIModel** orchestrates all three and supports three ablation modes.

----

Architecture Overview
---------------------

.. mermaid::

    graph TD
        Batch[Input Batch] --> S[BaselineStudent\nSMILES + FASTA]
        Batch --> T[BaselineTeacher\nTransductive IDs]

        subgraph MC-Dropout N passes eval only
            S -->|mc_logits| SVar[student_var\nmean_logit]
            T -->|mc_logits| TVar[teacher_var\nmean_logit]
        end

        SVar --> PG[PairGateFusion]
        TVar --> PG

        PG -->|α · logit_t + 1-α · logit_s| Out[Fused Logit]

        style PG fill:#f0e6ff,stroke:#9b59b6

----

BaselineStudent
---------------

*Registry key*: ``baseline_student``

An inductive encoder that operates purely from raw molecular features —
no graph convolutions, no pretrained transformers. Designed as a fast,
dependency-light baseline that generalises to unseen drugs and proteins.

**Drug branch**: projects raw RDKit atom features (7-dim) via a linear
layer, then applies ``global_mean_pool`` over the molecular graph.

**Protein branch**: embeds tokenised amino-acid IDs via ``nn.Embedding``,
mean-pools over the sequence (masked), then projects to ``hidden_dim``.

**Fusion**: concatenates drug and protein embeddings → Linear → ReLU →
Dropout → Linear(1).

.. code-block:: python

    from ugtsdti.models.student.baseline import BaselineStudent

    model = BaselineStudent(
        drug_raw_dim=7,
        prot_vocab_size=50,
        prot_embed_dim=32,
        hidden_dim=64,
        dropout=0.1,
    )
    out = model(batch)   # {"logits": Tensor[B, 1]}

.. autoclass:: ugtsdti.models.student.baseline.BaselineStudent
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

----

BaselineTeacher
---------------

*Registry key*: ``baseline_teacher``

A transductive encoder that looks up globally consistent hash-based node
embeddings for drugs and proteins. Mimics the role of a GCN-based teacher
that has seen the full similarity graph during pretraining.

**Drug / Protein branches**: ``nn.Embedding`` tables indexed by
``drug_index`` / ``target_index`` (MD5 hash mod large prime).

**Fusion**: same concat → MLP → logit pattern as the Student.

.. code-block:: python

    from ugtsdti.models.teacher.baseline import BaselineTeacher

    model = BaselineTeacher(num_drugs=100003, num_targets=100003, hidden_dim=64)
    out = model(batch)   # {"logits": Tensor[B, 1]}

.. autoclass:: ugtsdti.models.teacher.baseline.BaselineTeacher
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

----

PairGateFusion
--------------

*Registry key*: ``pairgate_fusion``

The core novelty of UGTSDTI. Computes a soft gate weight **α** from the
epistemic uncertainty pair ``(var_s, var_t)`` via a small MLP, then blends
student and teacher logits:

.. math::

    \alpha = \sigma\!\left(\text{MLP}\!\left([\,\sigma_s^2,\; \sigma_t^2\,]\right)\right)

.. math::

    \hat{y} = \alpha \cdot \ell_t + (1 - \alpha) \cdot \ell_s

When ``student_var`` / ``teacher_var`` are absent (ablation without
MC-Dropout), the gate falls back to a simple 50/50 average.

**Gate MLP**: Linear(2 → gate_hidden) → ReLU → Linear(gate_hidden → 1) → Sigmoid.

.. code-block:: python

    from ugtsdti.models.fusion.pairgate import PairGateFusion

    gate = PairGateFusion(gate_hidden=32, mc_samples=5)
    fused = gate(student_logits, teacher_logits, student_var, teacher_var)

.. autoclass:: ugtsdti.models.fusion.pairgate.PairGateFusion
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

----

HybridDTIModel
--------------

*Registry key*: ``hybrid_dti``

Top-level orchestration model. Builds sub-components from Hydra nested
configs via the ``MODELS`` registry and selects the forward path at runtime.

**Three operating modes** (determined by which sub-configs are present):

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Mode
     - Config keys
     - Behaviour
   * - ``hybrid``
     - student + teacher + fusion
     - MC-Dropout passes → PairGate → fused logit
   * - ``only_student``
     - student only
     - Direct student forward pass
   * - ``only_teacher``
     - teacher only
     - Direct teacher forward pass

**MC-Dropout guard**: MC passes run only during ``eval()`` mode.
During ``train()`` a single deterministic pass is used for efficiency.

.. mermaid::

    flowchart LR
        A{self.training?} -->|True| B[1 deterministic pass]
        A -->|False| C[N MC-Dropout passes]
        B --> D[PairGate fallback avg]
        C --> E[PairGate uncertainty gate]

.. autoclass:: ugtsdti.models.hybrid.HybridDTIModel
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
