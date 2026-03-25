Data Pipeline
=============

Overview
--------

UGTSDTI data flow is built around three ideas:

- explicit `S1-S4` split protocols
- scenario-bundle caching
- train-based teacher namespace with `UNK` handling

.. mermaid::

    flowchart TD
        TDC["PyTDC dataset"] --> Split["Scenario split\nS1 / S2 / S3 / S4"]
        Split --> Cache["Scenario bundle cache\ntrain / valid / test"]
        Cache --> Drug["SMILES -> PyG graph"]
        Cache --> Target["FASTA -> token ids + mask"]
        Cache --> Vocab["Train-based vocab\n+ UNK indices"]
        Vocab --> Batch["Batch dict"]
        Drug --> Batch
        Target --> Batch

Current Data Subpackages
------------------------

- ``ugtsdti.data.datasets``: dataset classes
- ``ugtsdti.data.transforms``: feature extraction/tokenization
- ``ugtsdti.data.protocols``: split probing and runtime audit
- ``ugtsdti.data.graph_builder``: teacher similarity graphs

Batch Format
------------

``TDCCachingDataset.__getitem__`` returns a dict with:

+------------------+----------------------------------+
| Key              | Used by                          |
+==================+==================================+
| ``drug``         | student drug encoder             |
+------------------+----------------------------------+
| ``target_ids``   | student target encoder           |
+------------------+----------------------------------+
| ``target_mask``  | student target encoder           |
+------------------+----------------------------------+
| ``label``        | loss / metrics                   |
+------------------+----------------------------------+
| ``drug_index``   | teacher lookup                   |
+------------------+----------------------------------+
| ``target_index`` | teacher lookup                   |
+------------------+----------------------------------+

Teacher Namespace
-----------------

Teacher IDs are intentionally **train-based**.

- train entities define the shared teacher namespace
- validation/test entities unseen in train are mapped to `UNK`
- this preserves transductive semantics without leaking validation/test nodes into the teacher graph

Protocol Validation
-------------------

Split semantics are validated at two levels:

1. runtime audit via ``ugtsdti.data.protocols.audit``
2. direct PyTDC probing via ``ugtsdti.data.protocols.probe``

This is how the repo verifies that:

- `S1` overlaps on both drugs and targets
- `S2` is cold on drugs
- `S3` is cold on targets
- `S4` is cold on both

Caching
-------

The dataset now caches per scenario bundle instead of per isolated split.

Benefits:

- one `get_split()` build per scenario
- shared tokenizer work across train/valid/test
- explicit bundle metadata for provenance

Benchmark Scenarios
-------------------

+------------+-------------------------------+--------------------+
| Scenario   | Meaning                       | Hydra config       |
+============+===============================+====================+
| ``S1``     | warm-start                    | ``tdc_davis_s1``   |
+------------+-------------------------------+--------------------+
| ``S2``     | cold drug                     | ``tdc_davis_s2``   |
+------------+-------------------------------+--------------------+
| ``S3``     | cold target                   | ``tdc_davis_s3``   |
+------------+-------------------------------+--------------------+
| ``S4``     | fully cold                    | ``tdc_davis_s4``   |
+------------+-------------------------------+--------------------+
