Data Pipeline
=============

Overview
--------

Data is fetched automatically via `PyTDC <https://tdcommons.ai/>`_ and cached to
``data/cache/`` as ``.pt`` files after the first run. Subsequent runs load directly from
cache, avoiding re-running RDKit and ESM tokenisation.

.. mermaid::

    flowchart TD
        A["PyTDC API\nDAVIS / KIBA / BindingDB"] --> B["Split\nrandom_split → S1\ncold_split(Drug) → S2\ncold_split(Target) → S3\ncold_split(None) → S4"]
        B --> C["Negative Sampling\n(handled by PyTDC)"]
        C --> D{".pt cache exists?"}
        D -- Yes --> E["Load from data/cache/"]
        D -- No --> F["RDKit\nSMILES → PyG molecular graph\n7 atom features · 3 bond features"]
        F --> G["ESMSequenceTokenizer\nFASTA → input_ids, attention_mask"]
        G --> H["MD5 hash → drug_index, target_index\nfor Teacher transductive lookup"]
        H --> I["torch.save → data/cache/*.pt"]
        I --> E
        E --> J["PyG DataLoader → Trainer"]

Datasets
--------

+------------+----------------------+------------------------+--------+
| Dataset    | Affinity type        | Binarization threshold | Splits |
+============+======================+========================+========+
| DAVIS      | Kd (nM)              | pKd ≥ 7.0              | S1, S4 |
+------------+----------------------+------------------------+--------+
| KIBA       | Composite KIBA score | —                      | S1, S4 |
+------------+----------------------+------------------------+--------+
| BindingDB  | Kd (nM)              | pKd ≥ 7.0              | S1     |
+------------+----------------------+------------------------+--------+

Batch format
------------

``TDCCachingDataset.__getitem__`` returns a dict. ``PyG DataLoader`` collates these
recursively into a batched dict:

+------------------+-------------+------------+----------------------------------+
| Key              | Type        | Shape      | Used by                          |
+==================+=============+============+==================================+
| ``drug``         | PyG Data    | variable   | Student (drug branch)            |
+------------------+-------------+------------+----------------------------------+
| ``target_ids``   | LongTensor  | [seq_len]  | Student (protein branch)         |
+------------------+-------------+------------+----------------------------------+
| ``target_mask``  | LongTensor  | [seq_len]  | Student (protein branch)         |
+------------------+-------------+------------+----------------------------------+
| ``label``        | FloatTensor | [1]        | Trainer (loss)                   |
+------------------+-------------+------------+----------------------------------+
| ``drug_index``   | LongTensor  | [1]        | Teacher (transductive lookup)    |
+------------------+-------------+------------+----------------------------------+
| ``target_index`` | LongTensor  | [1]        | Teacher (transductive lookup)    |
+------------------+-------------+------------+----------------------------------+

``drug_index`` and ``target_index`` are shared train-based entity IDs for the teacher branch.
Validation/test entities unseen in train are mapped to an explicit ``UNK`` slot so the
teacher remains transductive while the student stays inductive.

Molecular graph features
------------------------

``smiles_to_graph`` (``ugtsdti/data/transforms/chemistry.py``) extracts OGB-standard features:

**Node features** (7 per atom):

1. Atomic number (vocabulary of 118 elements)
2. Total degree
3. Formal charge
4. Number of radical electrons
5. Hybridisation (SP, SP2, SP3, …)
6. Is aromatic
7. Is in ring

**Edge features** (3 per bond, both directions):

1. Bond type (SINGLE, DOUBLE, TRIPLE, AROMATIC, …)
2. Is conjugated
3. Is in ring

Negative sampling policy
------------------------

DAVIS (and most PyTDC DTI datasets) contains **only positive pairs** — no negative sampling
is applied here. If negative sampling is added in the future, it **must** occur *per-split*
(after ``dataset.get_split()`` returns the train/valid/test subsets). Global negative
sampling before the split would cause data leakage between splits, invalidating all metrics.

Correct order::

    split_dict = dataset.get_split(...)   # split first
    for split_name, split_df in split_dict.items():
        split_df = add_negatives(split_df)  # then sample negatives per split

Config reference
----------------

.. code-block:: yaml

    # configs/data/tdc_davis_s2.yaml
    train:
      name: tdc_caching_dataset
      params:
        name: DAVIS
        split: train
        split_type: cold_split
        column_name: Drug
        cache_dir: ./data/cache
        seed: 42

    val:
      name: tdc_caching_dataset
      params:
        name: DAVIS
        split: valid          # PyTDC key for validation split
        split_type: cold_split
        column_name: Drug
        cache_dir: ./data/cache
        seed: 42

Benchmark scenarios
-------------------

+------------+--------------------------------------+------------------+
| Scenario   | Meaning                              | Hydra data config |
+============+======================================+==================+
| ``S1``     | warm-start / random split            | ``tdc_davis_s1`` |
+------------+--------------------------------------+------------------+
| ``S2``     | cold drug                            | ``tdc_davis_s2`` |
+------------+--------------------------------------+------------------+
| ``S3``     | cold target                          | ``tdc_davis_s3`` |
+------------+--------------------------------------+------------------+
| ``S4``     | fully cold via PyTDC global cold split | ``tdc_davis_s4`` |
+------------+--------------------------------------+------------------+
