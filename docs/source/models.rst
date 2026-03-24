Models & Uncertainty Gating
===========================

UGTSDTI's model hierarchy is built around three cooperating components:
**BaselineStudent** (inductive), **GCNTeacher** (transductive), and
**UncertaintyGatedFusion** (uncertainty-aware blending). The top-level
**HybridDTIModel** orchestrates all three and supports three ablation modes.

----

Architecture Overview
---------------------

.. mermaid::

    graph TD
        Batch[Input Batch] --> S[BaselineStudent\nSMILES + FASTA]
        Batch --> T[GCNTeacher\nDD/PP Graph Node IDs]

        subgraph MC-Dropout N passes eval only
            S -->|mc_logits| SVar[student_var\nmean_logit]
            T -->|mc_logits| TVar[teacher_var\nmean_logit]
        end

        SVar --> UG[UncertaintyGatedFusion]
        TVar --> UG

        UG -->|α · logit_t + 1-α · logit_s| Out[Fused Logit]

        style UG fill:#f0e6ff,stroke:#9b59b6

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
Dropout → Linear(1) → squeeze to ``(B,)``.

.. code-block:: python

    from ugtsdti.models.student.baseline import BaselineStudent

    model = BaselineStudent(
        drug_raw_dim=7,
        prot_vocab_size=50,
        prot_embed_dim=32,
        hidden_dim=64,
        dropout=0.1,
    )
    out = model(batch)   # {"logits": Tensor[B]}

.. autoclass:: ugtsdti.models.student.baseline.BaselineStudent
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

----

BaselineTeacher
---------------

*Registry key*: ``baseline_teacher``

A transductive encoder that looks up globally consistent sequential node
embeddings for drugs and proteins. Placeholder for ablation — no real
graph signal.

**Drug / Protein branches**: ``nn.Embedding`` tables indexed by
``drug_index`` / ``target_index`` (sequential 0..N-1).

**Fusion**: concat → MLP → logit → squeeze to ``(B,)``.

.. code-block:: python

    from ugtsdti.models.teacher.baseline import BaselineTeacher

    model = BaselineTeacher(num_drugs=1000, num_targets=1000, hidden_dim=64)
    out = model(batch)   # {"logits": Tensor[B]}

.. autoclass:: ugtsdti.models.teacher.baseline.BaselineTeacher
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

----

GCNTeacher
----------

*Registry key*: ``gcn_teacher``

The real Teacher encoder. Uses two GCN encoders operating on global
Drug-Drug (Tanimoto kNN) and Protein-Protein (k-mer cosine kNN) similarity
graphs. Node embeddings are looked up by sequential index.

Must call ``set_graphs(dd_graph, pp_graph)`` before ``forward()`` —
handled automatically by ``_wire_teacher_graphs()`` in ``main.py``.

.. code-block:: python

    from ugtsdti.models.teacher.gcn_teacher import GCNTeacher

    model = GCNTeacher(drug_feat_dim=2048, protein_feat_dim=8000, hidden_dim=128)
    model.set_graphs(dd_graph, pp_graph)
    out = model(batch)   # {"logits": Tensor[B]}

.. autoclass:: ugtsdti.models.teacher.gcn_teacher.GCNTeacher
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

----

UncertaintyGatedFusion
----------------------

*Registry key*: ``ug_fusion``

The core novelty of UGTSDTI. Computes a soft gate weight **α** from the
epistemic uncertainty pair ``(var_s, var_t)`` via a small MLP, then blends
student and teacher logits:

.. math::

    \alpha = \sigma\!\left(\text{MLP}\!\left([\,\sigma_s^2,\; \sigma_t^2\,]\right)\right)

.. math::

    \hat{y} = \alpha \cdot \ell_t + (1 - \alpha) \cdot \ell_s

When ``student_var`` / ``teacher_var`` are absent (training mode or ablation
without MC-Dropout), the gate falls back to a simple 50/50 average.

**Gate MLP**: Linear(2 → gate_hidden) → ReLU → Linear(gate_hidden → 1) → Sigmoid.

.. code-block:: python

    from ugtsdti.models.fusion.ug import UncertaintyGatedFusion

    gate = UncertaintyGatedFusion(gate_hidden=32, mc_samples=5)
    fused = gate(student_logits, teacher_logits, student_var, teacher_var)
    # fused: Tensor[B]

.. autoclass:: ugtsdti.models.fusion.ug.UncertaintyGatedFusion
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
     - Config
     - Behaviour
   * - ``hybrid``
     - student + teacher + fusion
     - MC-Dropout passes → UG gate → fused logit
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
        A{self.training?} -->|True| B[1 deterministic pass\nfallback 50/50 avg]
        A -->|False| C[N MC-Dropout passes\nUG gate from uncertainty]

.. autoclass:: ugtsdti.models.hybrid.HybridDTIModel
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
