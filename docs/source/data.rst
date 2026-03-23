Data Caching Pipeline
=====================

PyTDC Operations
----------------
UGTSDTI intercepts HuggingFace and RDKit logic directly inside the :code:`data/` module.
It accesses PyTDC (Therapeutics Data Commons) securely, pulling and automatically mapping splits like :code:`cold_split`.

MD5 Hash Checks
---------------
To avoid continuous molecular parsing, the datasets run an explicit parameter sweep converting the schema into an MD5 Hash string. Extracted `.pt` binaries are saved permanently, decreasing subsequent Dataloading loops exponentially.

Data Processing Flowchart
-------------------------

.. mermaid::

    graph LR
        subgraph Sources
            DB[(PyTDC Benchmark)]
        end

        subgraph Extractors
            S[SMILES Strings]
            F[Protein FASTA]
            R[RDKit Parser]
            E[ESM Tokenizer]
        end

        subgraph Output Array
            C[(MD5 Cache .pt)]
            Batch[PyG DataLoader]
        end

        DB --> S & F
        S --> R
        F --> E
        R --> C
        E --> C
        C --> Batch
