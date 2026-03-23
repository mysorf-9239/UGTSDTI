Getting Started
===============

Installation
------------

Clone the repository and install through Conda:

.. code-block:: bash

    git clone https://github.com/mysorf-9239/UGTSDTI.git
    cd UGTSDTI
    make setup
    conda activate ugtsdti-full

Running the Baseline
--------------------

The architecture evaluates using Hydra parameter sweeps automatically:

.. code-block:: bash

    conda run -n ugtsdti python -m ugtsdti.main model=hybrid_baseline data=tdc_davis
