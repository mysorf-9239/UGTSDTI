Overview
========

Research Goal
-------------

UGTSDTI asks whether a DTI system can adapt across warm-start and cold-start regimes
without committing to only one encoder family.

- The **teacher** is transductive and strongest when train-time entity structure is available.
- The **student** is inductive and remains usable when drugs or targets are unseen.
- The **UG gate** should learn when to trust each branch from uncertainty rather than from a fixed rule.

Cold-Start Scenarios
--------------------

+----------+---------------+-----------------+-------------------------+
| Scenario | Drug in train | Target in train | Meaning                 |
+==========+===============+=================+=========================+
| ``S1``   | ✓             | ✓               | warm-start              |
+----------+---------------+-----------------+-------------------------+
| ``S2``   | ✗             | ✓               | cold drug               |
+----------+---------------+-----------------+-------------------------+
| ``S3``   | ✓             | ✗               | cold target             |
+----------+---------------+-----------------+-------------------------+
| ``S4``   | ✗             | ✗               | fully cold              |
+----------+---------------+-----------------+-------------------------+

System Architecture
-------------------

.. mermaid::

    flowchart LR
        Cfg["Hydra Config\nmodel / teacher / student / fusion / loss / data"] --> Compose["Experiment Resolver"]
        Compose --> Model["HybridDTIModel"]
        Compose --> Data["TDCCachingDataset"]
        Compose --> Loss["BCELoss / KDLoss"]

        Model --> Student["Student Plugin"]
        Model --> Teacher["Teacher Plugin"]
        Model --> Fusion["Fusion Plugin"]

Research Architecture
---------------------

.. mermaid::

    flowchart LR
        Batch["Batch"] --> Student["Student\ninductive"]
        Batch --> Teacher["Teacher\ntransductive"]

        Student --> SLogit["student_logits"]
        Teacher --> TLogit["teacher_logits"]

        Student --> SVar["MC-Dropout\nstudent_var"]
        Teacher --> TVar["MC-Dropout\nteacher_var"]

        SVar --> Gate["UG Gate\nalpha = MLP(var_s, var_t)"]
        TVar --> Gate
        SLogit --> Fuse["fused logits"]
        TLogit --> Fuse
        Gate --> Fuse

Ablation Modes
--------------

.. mermaid::

    flowchart LR
        A["HybridDTIModel"] --> B{teacher / student / fusion}
        B --> C["teacher=none\nstudent=baseline\nfusion=none"]
        B --> D["teacher=gcn\nstudent=none\nfusion=none"]
        B --> E["teacher=gcn\nstudent=baseline\nfusion=ug"]
        B --> F["teacher=gcn\nstudent=baseline\nfusion=ug\nloss=kd"]

What Is Verified In Code
------------------------

The current codebase explicitly supports and audits:

- slot-based experiment configuration
- train/eval MC-Dropout budgets
- teacher train-based namespace with `UNK`
- runtime split audit
- benchmark protocol probing
- branch-wise and gate-wise evaluation metrics
