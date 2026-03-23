Architecture Map
================

`UGTSDTI` employs a strict Plugin-based hierarchy built on top of ``Hydra`` and a custom Python Registry (``@register``).

Registries Available
--------------------
- ``MODELS``: For Graph and Sequence Encoders
- ``DATASETS``: For PyTDC processing
- ``LOSSES``: For Knowledge Distillation bounds

All models evaluate within the immutable ``Trainer`` core.

System Data Flow Diagram
------------------------

.. mermaid::

    graph TD
        %% Configuration Flow
        subgraph Configurations [Hydra YAML Settings]
            C1[model=hybrid_baseline]
            C2[data=tdc_davis]
            C3[trainer=default_trainer]
        end

        %% Instantiation Engine
        subgraph Engine [Core Injection Registry]
            R[ugtsdti.core.registry]
        end

        %% Evaluator
        subgraph Execution [Trainer Loop]
            T[Trainer class]
            W[Weights & Biases]
        end

        Configurations -->|Instantiate Strings| R
        R -->|Returns PyTorch Objects| T
        T -->|Log Metrics| W
