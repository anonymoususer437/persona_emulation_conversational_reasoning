# Zero-Shot Persona Emulation from Conversational Data
### Latent Persona Synthesis (LPS) — EMNLP 2026

> **Paper:** *Zero-Shot Persona Emulation from Conversational Data*  


[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

**Latent Persona Synthesis (LPS)** predicts how a specific user would respond
to a novel question they have never been asked before — a task we call
**Zero-Shot Persona Emulation** — using only their longitudinal conversation
history with an LLM.

LPS addresses two failure modes of standard RAG:
- **Model Verbosity Bias** — model-generated text dominates retrieval, diluting the user's voice
- **Temporal Obsolescence** — old interactions are weighted equally to recent ones

It does so through two complementary frameworks:

| Framework | Approach | Best for |
|---|---|---|
| **LPS-Clustering (ELC)** | Time-decayed HDBSCAN clusters → Cognitive Anchors | Creative / personal domains |
| **LPS-Graph-RAG** | Temporally decayed knowledge graph + Leiden communities | Technical / professional domains |

Both share a **User-Prioritized Embedding** (agency coefficient α) and a
**Two-Stage Retrieval** pipeline (topic + tone).

---

## Results on WildChat-1M (4,200 users)

| Method | ROUGE-1 | ROUGE-L | BERTScore-F1 | PF |
|---|---|---|---|---|
| Zero-Shot | 18.4 | 14.2 | 0.831 | 2.1 |
| Top-K RAG | 27.3 | 22.1 | 0.861 | 2.8 |
| Persona-DB | 29.8 | 24.6 | 0.869 | 3.0 |
| PGraphRAG | 31.4 | 26.1 | 0.874 | 3.2 |
| **LPS-Clustering** | **33.6** | **27.8** | **0.883** | **3.4** |
| **LPS-Graph-RAG** | **35.7** | **29.4** | **0.891** | **3.6** |

---

## Repository Structure

```
lps_repo/
├── notebooks/
│   ├── 01_data_preparation.ipynb          # WildChat filtering & user cohort construction
│   ├── 02_user_prioritized_embeddings.ipynb  # Agency-weighted embedding (α)
│   ├── 03_elc_framework.ipynb             # ELC: HDBSCAN + temporal decay
│   ├── 04_graph_rag_framework.ipynb       # Graph-RAG: triplets + Leiden + edge decay
│   ├── 05_two_stage_retrieval.ipynb       # Tier 1 (topic) + Tier 2 (tone κ)
│   ├── 06_evaluation.ipynb               # ROUGE, BERTScore, TC, PF evaluation
│   └── 07_ablation_studies.ipynb         # α, λ, retrieval strategy ablations
├── src/
│   ├── embeddings.py                     # UserPrioritizedEmbedder
│   ├── clustering.py                     # ELC: HDBSCAN + CognitiveAnchor
│   ├── graph_rag.py                      # P-Graph: KG construction + Leiden
│   ├── retrieval.py                      # TwoStageRetriever
│   ├── generation.py                     # DualStreamGenerator
│   └── evaluation.py                     # Metrics
├── configs/
│   └── default.yaml                      # Hyperparameters
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/your-username/lps-persona-emulation
cd lps-persona-emulation
pip install -r requirements.txt
jupyter notebook notebooks/01_data_preparation.ipynb
```

---

