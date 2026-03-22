# UGTSDTI: Uncertainty-Gated Teacher–Student Learning for Drug–Target Interaction Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)

**UGTSDTI** is a research-grade, SOTA framework for Drug-Target Interaction (DTI) prediction. It dynamically fuses Sequence-based Protein Language Models (ESM) and Graph-based structured representation using an **Uncertainty-Gated Mechanism** (MC Dropout).

This framework is completely refactored from the ground up to support modern Deep Learning research pipelines.

---

## 🏗 Architecture Diagram
```mermaid
graph TD
    subgraph Configurations ["⚙️ Config System (Hydra)"]
        A[configs/default.yaml] --> M[model/hybrid.yaml]
        A --> D[data/s1.yaml]
        A --> T[trainer/default.yaml]
    end

    subgraph CoreEngine ["🧠 Core Engine (Frozen)"]
        R[Registry Pattern @register]
        Trainer[Trainer Loop]
        Metrics[DTI Metrics (CI, AUROC)]
        Logger[WandB + Loguru]
    end

    subgraph Plugins ["🧩 Customizable Plugins"]
        Model[Models: ESM, PairGate, GCN]
        Data[Datasets: PyTDC Caching]
    end

    Configurations --> R
    R --> |"Build from string"| Model
    R --> |"Build from string"| Data
    Model --> Trainer
    Data --> Trainer
    Metrics --> Trainer
    Trainer --> Logger
```

---

## 🌟 Modern SOTA Integration
- **Hydra Configuration:** Multi-level dict configs for true plug-and-play decoupling.
- **Weights & Biases (WandB):** Automated metric tracking and hyperparameter sweeping via `hydra-optuna`.
- **PyTDC (Therapeutics Data Commons):** Natively fetches DAVIS, KIBA, and BindingDB datasets with S1-S4 cold-split benchmarks.
- **Feature Caching (`.pt`):** Lightning fast dataloading bypassing repetitive RDKit or FASTA extraction.
- **Protein Language Models (PLMs):** Pre- интегрированный HuggingFace `transformers` (ESM-2, ProtBERT) directly into `@register_model`.

---

## 🚀 Getting Started

### 1. Installation
We use `conda-recipes/` to keep the root directory clean and provide different hardware configurations.
```bash
git clone git@github.com:mysorf-9239/UGTSDTI.git
cd UGTSDTI
make setup # Uses conda to create the 'ugtsdti-full' environment and installs pre-commits
conda activate ugtsdti-full
```

### 2. File Architecture

```text
UGTSDTI/
├── configs/             # Hydra Configs (model, data, trainer)
├── ugtsdti/             # Core Python Package
│   ├── core/            # Registry, Metrics, Trainer
│   ├── data/            # PyTDC & S1-S4 Handling
│   ├── models/          # Student (ESM), Teacher, Fusion (PairGate)
│   └── utils/           # Tracking, Checkpoints, Reproducibility
├── scripts/             # Useful bash scripts
├── examples/            # Example pipelines
├── conda-recipes/       # Environment files for isolated systems
└── pyproject.toml       # Mypy, Ruff, Black Configurations
```

### 3. Usage
Run the default experiment:
```bash
python -m ugtsdti.main model=hybrid_dti data=davis_s4
```

---

## ⚖ License
Licensed under the [MIT License](LICENSE).
