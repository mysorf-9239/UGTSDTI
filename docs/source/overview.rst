Overview
========

The Cold-Start Problem
----------------------

Standard DTI benchmarks define four evaluation scenarios based on whether drugs and targets
appeared during training:

+----------+---------------+-----------------+-----------------------------------+
| Scenario | Drug in train | Target in train | Difficulty                        |
+==========+===============+=================+===================================+
| **S1**   | ✓             | ✓               | Easy — warm start                 |
+----------+---------------+-----------------+-----------------------------------+
| **S2**   | ✗             | ✓               | Medium — cold drug                |
+----------+---------------+-----------------+-----------------------------------+
| **S3**   | ✓             | ✗               | Medium — cold target              |
+----------+---------------+-----------------+-----------------------------------+
| **S4**   | ✗             | ✗               | Hard — fully cold, most realistic |
+----------+---------------+-----------------+-----------------------------------+

Graph-based (transductive) models excel at S1 but collapse at S4 because new nodes have no
embedding in the graph. Sequence-based (inductive) models generalise better but underperform
at S1 where graph context is available. UGTSDTI aims to handle all four scenarios with a
single adaptive model.

Architecture
------------

.. mermaid::

    flowchart LR
        subgraph Input
            A["Drug SMILES"]
            B["Drug Node ID / Protein Node ID"]
        end

        subgraph Student["Student Branch (Inductive)"]
            C["Sequence Encoder\nSMILES → PyG Graph\nFASTA → ESM Tokens"]
            D["logit_s (train) / ŷ_s (eval)"]
        end

        subgraph Teacher["Teacher Branch (Transductive)"]
            E["GNN Encoder\nDD + PP Similarity Graph\n(GAT / GCN)"]
            F["logit_t (train) / ŷ_t (eval)"]
        end

        subgraph MC["MC-Dropout (eval mode, N passes)"]
            G["ŷ_s = Mean(logit_s^1..N)\nvar_s = Var(logit_s^1..N)"]
            H["ŷ_t = Mean(logit_t^1..N)\nvar_t = Var(logit_t^1..N)"]
        end

        subgraph Fusion["PairGate Fusion"]
            I["Gate MLP\n[var_s, var_t] → α ∈ (0,1)"]
            J["ŷ = α · logit_t + (1−α) · logit_s"]
        end

        A --> C --> D --> G
        B --> E --> F --> H
        G --> I
        H --> I
        D --> J
        F --> J
        I --> J

Ablation modes
--------------

``HybridDTIModel`` natively supports four configurations via Hydra config:

.. mermaid::

    flowchart LR
        A["HybridDTIModel"] --> B{student_cfg\nteacher_cfg\nfusion_cfg}
        B -->|student only| C["only_student\nBaseline: sequence encoder alone"]
        B -->|teacher only| D["only_teacher\nBaseline: graph encoder alone"]
        B -->|all three| E["hybrid_baseline\nPairGate fusion · BCE loss"]
        B -->|all three + KD| F["hybrid_kd\nPairGate fusion · KDDualLoss"]

Research novelty
----------------

Adaptive fusion driven by epistemic uncertainty is not common in DTI literature. Most SOTA
models commit to one encoder type and do not handle the warm/cold transition automatically.
UGTSDTI's PairGate learns *when* to trust each branch based on how uncertain each branch is
about a given sample — a property that emerges naturally from MC-Dropout variance.

The key correctness property (Gal & Ghahramani, 2016): both the prediction logit and the
epistemic variance must come from the **same** set of stochastic forward passes:

.. math::

    \hat{y}_s = \frac{1}{N} \sum_{n=1}^{N} f_s^{(n)}(x), \quad
    \sigma^2_s = \frac{1}{N} \sum_{n=1}^{N} \left( f_s^{(n)}(x) - \hat{y}_s \right)^2

where :math:`f_s^{(n)}` denotes the :math:`n`-th stochastic forward pass of the student branch
with dropout active.
