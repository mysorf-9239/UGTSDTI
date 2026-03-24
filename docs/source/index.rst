UGTSDTI Documentation
=====================

**UGTSDTI** — *Uncertainty-Gated Teacher–Student Learning for Drug–Target Interaction Prediction*.

UGTSDTI addresses the **Cold-Start Problem** in DTI prediction by combining two complementary
encoders via an uncertainty-aware fusion gate (**UG Fusion**):

- **Teacher** (graph-based, transductive): learns from global Drug-Drug and Protein-Protein
  similarity graphs. Strong on warm-start (S1), degrades on cold-start.
- **Student** (sequence-based, inductive): encodes directly from SMILES/FASTA. Generalises to
  unseen molecules.
- **UG Fusion** (Uncertainty-Gated): estimates epistemic uncertainty via MC-Dropout and
  dynamically weights each branch — when the Teacher is uncertain (cold-start), the Student
  is trusted more.

------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting-started
   overview
   data
   models
   training
   config
   modules/index

------

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
