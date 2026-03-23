Knowledge Distillation (KD)
===========================

Training Strategy
-----------------
UGTSDTI employs offline Knowledge Distillation to regularize the Student Network's latent space, forcing the generic sequence embeddings to mimic the deterministic precision of the Graph Teacher.

Distillation Loss Diagram
-------------------------

.. mermaid::

    graph LR
        %% Inputs
        X[TDC Graph Batch] --> Teacher
        X --> Student

        %% Networks
        Teacher[Teacher Baseline] --> TLogits[Teacher Logits]
        Student[Student ESM/MPNN] --> SLogits[Student Logits]

        %% Fusion
        TLogits --> PairGate
        SLogits --> PairGate
        PairGate --> FLogits[Fused Logits]

        %% Targets
        Y[True Label]

        %% Losses
        TLogits -.->|MSE Distillation| SLogits
        FLogits -.->|BCE Main Task| Y

        Note over PairGate: KDDualLoss = (1 - a)*BCE + a*MSE
