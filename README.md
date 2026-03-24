# UGTSDTI: Uncertainty-Gated Teacher–Student Learning for Drug–Target Interaction Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)

## Abstract

Drug-Target Interaction (DTI) prediction is a critical step in early-stage drug discovery.
A key challenge is the **Cold-Start Problem**: models trained on known drug-protein pairs fail to generalize to entirely
unseen drugs or targets (S2–S4 splits).

UGTSDTI addresses this by combining two complementary encoders via an uncertainty-aware fusion gate (**PairGate**):

- **Teacher** (graph-based, transductive): learns from global Drug-Drug and Protein-Protein similarity graphs. Strong on
  warm-start (S1), degrades on cold-start because new nodes have no graph context.
- **Student** (sequence-based, inductive): encodes directly from SMILES/FASTA. Generalizes to unseen molecules.
- **PairGate**: estimates epistemic uncertainty of each branch via MC-Dropout and dynamically weights their
  contributions — when the Teacher is uncertain (cold-start), the Student is trusted more.

The core novelty is this adaptive, uncertainty-driven fusion: most SOTA DTI models commit to one encoder type and do not
handle the warm/cold transition automatically.

---

## The Cold-Start Problem

Standard DTI benchmarks define four evaluation scenarios based on whether drugs and targets appeared during training:

| Scenario | Drug in train | Target in train | Difficulty                        |
|----------|:-------------:|:---------------:|-----------------------------------|
| **S1**   |       ✓       |        ✓        | Easy — warm start                 |
| **S2**   |       ✗       |        ✓        | Medium — cold drug                |
| **S3**   |       ✓       |        ✗        | Medium — cold target              |
| **S4**   |       ✗       |        ✗        | Hard — fully cold, most realistic |

Graph-based (transductive) models excel at S1 but collapse at S4 because new nodes have no embedding in the graph.
Sequence-based (inductive) models generalize better but underperform at S1 where graph context is available. UGTSDTI
aims to handle all four scenarios with a single adaptive model.

---

## Method

### Architecture

```mermaid
flowchart LR
    subgraph Input
        A["Drug SMILES"]
        B["Drug Node ID\nProtein Node ID"]
    end

    subgraph Student["Student Branch (Inductive)"]
        C["Sequence Encoder\nSMILES → PyG Graph\nFASTA → ESM Tokens"]
        D["logit_s (train) / ŷ_s (eval)"]
    end

    subgraph Teacher["Teacher Branch (Transductive)"]
        E["GNN Encoder\nDD + PP Similarity Graph\n(GAT / GCN)"]
        F["logit_t (train) / ŷ_t (eval)"]
    end

    subgraph MC["MC-Dropout (eval mode, N forward passes)"]
        G["ŷ_s = Mean(logit_s^1..N)\nvar_s = Var(logit_s^1..N)"]
        H["ŷ_t = Mean(logit_t^1..N)\nvar_t = Var(logit_t^1..N)"]
    end

    subgraph Fusion["PairGate Fusion"]
        I["Gate MLP\n[var_s, var_t] → α ∈ (0,1)"]
        J["ŷ = α · logit_t + (1−α) · logit_s"]
    end

    A --> C --> D --> G
    B --> E --> F --> H
    G --> I
    H --> I
    D --> J
    F --> J
    I --> J
```

### PairGate Fusion

Epistemic uncertainty is estimated via **Monte Carlo Dropout**: at eval time, the model runs `N` stochastic forward
passes with dropout active and computes both the mean prediction and variance across passes:

```
ŷ_s = Mean({logit_s^(1), ..., logit_s^(N)})   ← prediction logit (eval mode)
var_s = Var({logit_s^(1), ..., logit_s^(N)})   ← epistemic uncertainty

ŷ_t = Mean({logit_t^(1), ..., logit_t^(N)})
var_t = Var({logit_t^(1), ..., logit_t^(N)})
```

Both `ŷ` and `var` come from the **same** `mc_logits` tensor — this is the key correctness property
(Gal & Ghahramani, 2016). During training, a single forward pass is used (dropout active naturally via `model.train()`),
no extra MC passes are run.

The gate MLP maps the uncertainty pair to a scalar weight `α ∈ (0, 1)`:

```
α = σ(MLP([var_s, var_t]))
ŷ = α · ŷ_t + (1 − α) · ŷ_s
```

When the Teacher is uncertain (e.g., cold-start node absent from graph), `var_t` is high → `α → 0` → Student dominates.
When the Teacher is confident (warm-start), `α → 1` → Teacher dominates.

### Training Objective

Training uses **KDDualLoss**, a convex combination of task loss and knowledge distillation loss:

```
L = (1 − β) · L_task + β · L_distill

where:
  L_task    = BCE(ŷ, y)                  — supervised signal on fused output
  L_distill = MSE(logit_s, logit_t)      — student learns from teacher's logits
  β         = alpha hyperparameter ∈ [0, 1]
```

Note: `β` (KD weight) and `α` (gate weight) are independent — `β` is a fixed training hyperparameter, while `α` is
computed dynamically per sample from uncertainty estimates.

---

## Data Pipeline

```mermaid
flowchart TD
    A["PyTDC API\nDAVIS / KIBA / BindingDB"] --> B["Split\ncold_split → S4\nrandom_split → S1"]
    B --> C["Negative Sampling\n(handled by PyTDC)"]
    C --> D{".pt cache exists?"}
    D -- Yes --> E["Load from data/cache/"]
    D -- No --> F["RDKit\nSMILES → PyG molecular graph\n7 atom features · 3 bond features"]
    F --> G["ESMSequenceTokenizer\nFASTA → input_ids, attention_mask"]
    G --> H["MD5 hash → drug_index, target_index\nfor Teacher transductive lookup"]
    H --> I["torch.save → data/cache/*.pt"]
    I --> E
    E --> J["PyG DataLoader → Trainer"]
```

Data is fetched automatically via [PyTDC](https://tdcommons.ai/) and cached to `data/cache/` as `.pt` files after the
first run. Subsequent runs load directly from cache.

---

## Installation

```bash
git clone <repo>
cd UGTSDTI
conda env create -f conda-recipes/full.yaml
conda activate ugtsdti
pip install -e .
```

**Stack:** Python 3.10 · PyTorch 2.1+ · PyTorch Geometric · Hydra-core · WandB · PyTDC · RDKit · HuggingFace
Transformers · Loguru

---

## Usage

Configuration is managed via [Hydra](https://hydra.cc/). All parameters can be overridden at the command line.

```bash
# Student-only baseline
python -m ugtsdti.main model=only_student data=tdc_davis

# Teacher-only baseline
python -m ugtsdti.main model=only_teacher data=tdc_davis

# Hybrid: student + teacher + PairGate (BCE loss)
python -m ugtsdti.main model=hybrid_baseline data=tdc_davis

# Hybrid with Knowledge Distillation loss
python -m ugtsdti.main model=hybrid_baseline data=tdc_davis \
    trainer.loss.name=kd_dual_loss trainer.loss.alpha=0.5

# Full ablation suite (4 modes, WandB disabled)
bash scripts/run_baselines.sh
```

---

## Datasets

| Dataset   | Affinity type        | Binarization threshold | Splits |
|-----------|----------------------|------------------------|--------|
| DAVIS     | Kd (nM)              | pKd ≥ 7.0              | S1, S4 |
| KIBA      | Composite KIBA score | —                      | S1, S4 |
| BindingDB | Kd (nM)              | pKd ≥ 7.0              | S1     |

---

## Evaluation Metrics

| Metric | Type           | Description                                             |
|--------|----------------|---------------------------------------------------------|
| AUROC  | Classification | Area under ROC curve                                    |
| AUPRC  | Classification | Area under Precision-Recall curve                       |
| F1     | Classification | Binary F1 at threshold                                  |
| MSE    | Regression     | Mean squared error on raw affinity values               |
| CI     | Ranking        | Concordance Index — fraction of correctly ordered pairs |

---

## Ablation Modes

The framework natively supports four ablation configurations via `HybridDTIModel`:

```mermaid
flowchart LR
    A["HybridDTIModel"] --> B{student_cfg\nteacher_cfg\nfusion_cfg}
    B -->|student only| C["only_student\nBaseline: sequence encoder alone"]
    B -->|teacher only| D["only_teacher\nBaseline: graph encoder alone"]
    B -->|all three| E["hybrid_baseline\nPairGate fusion · BCE loss"]
    B -->|all three + KD| F["hybrid_kd\nPairGate fusion · KDDualLoss"]
```

---

## Project Structure

```
UGTSDTI/
├── configs/
│   ├── default.yaml              # Top-level Hydra defaults
│   ├── model/
│   │   ├── only_student.yaml
│   │   ├── only_teacher.yaml
│   │   └── hybrid_baseline.yaml
│   ├── data/
│   │   └── tdc_davis.yaml
│   └── trainer/
│       └── default_trainer.yaml
├── ugtsdti/
│   ├── main.py                   # Entry point (@hydra.main)
│   ├── core/                     # FROZEN: Registry, Trainer, Metrics
│   ├── data/
│   │   ├── datasets/
│   │   │   └── tdc_dataset.py    # TDCCachingDataset (PyTDC + disk cache)
│   │   └── transforms/
│   │       ├── chemistry.py      # smiles_to_graph (RDKit, OGB-standard features)
│   │       └── sequence.py       # ESMSequenceTokenizer
│   ├── models/
│   │   ├── hybrid.py             # HybridDTIModel — orchestration + MC-Dropout
│   │   ├── student/
│   │   │   └── baseline.py       # BaselineStudent (GlobalMeanPool, working)
│   │   ├── teacher/
│   │   │   └── baseline.py       # BaselineTeacher (nn.Embedding placeholder)
│   │   └── fusion/
│   │       └── pairgate.py       # PairGateFusion (gate MLP + uncertainty weighting)
│   ├── losses/
│   │   └── distillation.py       # KDDualLoss (BCE task + MSE distillation)
│   └── utils/                    # Logger, Seed
├── tests/                        # pytest suite
├── scripts/
│   ├── run_baselines.sh          # 4-mode ablation (WandB disabled)
│   └── run_experiments.sh        # Full experiment runs
└── .agent/                       # AI context, task tracking, research notes
```

---

## Citation

```bibtex
@misc{ugtsdti2026,
    title = {UGTSDTI: Uncertainty-Gated Teacher--Student Learning for
 Drug--Target Interaction Prediction},
    author = {},
    year = {2026},
    note = {Work in progress}
}
```

## License

[MIT License](LICENSE)
