Models & Uncertainty Gating
===========================

Teacher - Student
-----------------
UGTSDTI dynamically constructs isolated Neural Networks via the :code:`models/` subtree:

- **Teachers**: Extract rich molecular structures offline using Transductive lookups.
- **Students**: Employ Inductive Sequence mapping (Transformers) to scale generalizations.

Inference and Uncertainty Flow
------------------------------

.. mermaid::

    sequenceDiagram
        autonumber
        participant Input as Batch Graph/Seq
        participant T as Teacher Network
        participant S as Student Network
        participant PG as PairGate Fusion

        Input->>T: Forward Pass (Transductive Nodes)
        Input->>S: Forward Pass (Inductive Sequences)

        loop MC-Dropout 5 Times
            T->>T: Stochastic Execute
            S->>S: Stochastic Execute
        end

        T-->>PG: teacher_logits & teacher_var
        S-->>PG: student_logits & student_var

        Note over PG: Sigmoid(MLP(Vars)) gating
        PG-->>Input: Final Fused Logit Prediction

PairGate Fusion
---------------
The :code:`models/fusion/pairgate.py` interface enables **Uncertainty Gating**.
By utilizing **Monte Carlo Dropout**, it extracts standard deviations across multiple stochastic executions and fuses distributions proportionally relying on higher-confidence predictions.
