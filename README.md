# AML Project: Part-of-Speech Tagging on UD English-EWT

![Python](https://img.shields.io/badge/python-3.13%2B-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/managed%20with-uv-6C47FF?logo=uv&logoColor=white)
![Tests](https://img.shields.io/badge/tests-31%20passed-2ECC71)
![Phase 1](https://img.shields.io/badge/Phase%201-HMM%20Complete-0984E3)

Exploratory analysis, statistical modeling, and deep sequence labeling for Universal POS (UPOS) tagging over the Universal Dependencies English Web Treebank (UD English-EWT). 

This repository implements:
1. **Comprehensive EDA**: Corpus statistics, sentence lengths, Zipfian tag skew, unknown-token distributions, and lexical ambiguity.
2. **Phase 1 Generative Baseline**: A mathematically rigorous first-order Hidden Markov Model (HMM) with add-$\alpha$ smoothing, sentence boundaries, log-space Viterbi decoding, and an evaluation protocol distinguishing model vocabulary from pure empirical training lexicon.
3. **Phase 2 (Upcoming)**: Deep neural sequence models (BiLSTM) and hybrid extensions.

---

## Project & Team Specifications

- **Project**: **Project 16**: Part-of-Speech Tagging: Hidden Markov Model vs Bidirectional LSTM Tagger
- **Team ID**: **T058** | **Section**: **A**
- **Degree Program**: **B. Tech in Computer Science & Artificial Intelligence**
- **Institution**: Newton School of Technology, Rishihood University

| # | Enrollment Number | Full Name | Institutional Email |
| :-: | :---: | :--- | :--- |
| 1 | `2401010391` | **Rohan Singh** | `rohan.singh2024@nst.rishihood.edu.in` |
| 2 | `2401020088` | **Aniket Pathak** | `aniket.pathak2024@nst.rishihood.edu.in` |
| 3 | `2401010361` | **Pushpendra Singh Parihar** | `pushpendra.parihar2024@nst.rishihood.edu.in` |
| 4 | `2401010209` | **Kapil Karan Mathur** | `kapil.mathur2024@nst.rishihood.edu.in` |
| 5 | `2401010045` | **Agrim Kumar Malhotra** | `agrim.malhotra2024@nst.rishihood.edu.in` |

---

## Key Highlights

- **Sentence-Preserving CoNLL-U Parser**: Strict 10-column validation extracting `FORM` and `UPOS` while properly filtering comments, multiword tokens (`1-2`), and empty nodes (`2.1`).
- **First-Order HMM POS Tagger**:
  - Sentence boundary modeling via dedicated `<START>` and `<END>` context markers (Brants, 2000).
  - Add-$\alpha$ smoothing across state transitions and emission distributions.
  - Numerically stable log-space Viterbi decoding with dynamic programming backpointer trellis (Rabiner, 1989; Jurafsky & Martin, 2026).
- **Strict Vocabulary & Evaluation Protocol**:
  - Maintains two distinct vocabulary representations: model vocabulary $V_{\text{model}}$ (for probability tables, containing `<UNK>` and frequency-pruned tokens) and empirical training vocabulary $V_{\text{original}}$ (19,674 distinct types, strictly excluding `<UNK>`).
  - Unknown-word evaluation strictly tests tokens unseen in $V_{\text{original}}$ (Sokolova & Lapalme, 2009).
- **Split Discipline**:
  - Hyperparameter tuning ($\alpha \in \{0.01, 0.1, 1.0\}$) performed exclusively on the Development set; Test set evaluated strictly once to prevent data leakage (Manning et al., 2008).
- **Publication-Grade Visualizations**:
  - High-DPI heatmaps for the transition probability matrix $A$, step-by-step Viterbi trellis with optimal path overlay, dual-axis tuning curves, and class-normalized confusion matrices.
  - Interactive demonstration using iconic movie dialogues from David Fincher films (*Fight Club*, *The Social Network*, *Se7en*).

---

## Repository Layout

```text
aml-project/
├── data/                         # Official UD English EWT CoNLL-U splits
│   ├── train.conllu              # Train split (12,544 sents, 204,578 tokens)
│   ├── dev.conllu                # Dev split (2,001 sents, 25,148 tokens)
│   └── test.conllu               # Test split (2,077 sents, 25,094 tokens)
├── experiments/                  # Reproducible experiment runners
│   └── phase1_hmm.py             # Phase 1 HMM tuning and test evaluation
├── notebooks/                    # Self-explanatory Jupyter notebooks
│   ├── eda.ipynb                 # Exploratory data analysis & corpus statistics
│   └── phase1_hmm.ipynb          # End-to-end HMM walkthrough, math, and interactive demo
├── references/                   # Standardized academic literature
│   ├── phase1/                   # Phase 1 papers (Church, Rabiner, Brants, Sokolova, etc.)
│   └── phase2/                   # Phase 2 papers (BiLSTM, word embeddings, hybrid models)
├── src/                          # Modular core package
│   ├── dataset_loader.py         # CoNLL-U loader and UPOS sequence extractors
│   ├── dataset_registry.py       # Dataset path registry and integrity checks
│   ├── evaluation.py             # Accuracy, Macro F1, and unknown-word metrics
│   ├── hmm_model.py              # First-order HMM with log-space Viterbi tagger
│   └── preprocessing.py          # Vocabularies, boundaries, and OOV normalizers
├── tests/                        # Comprehensive unit and integration test suite
│   ├── test_dataset_loader.py    # Basic loader sanity checks
│   ├── test_dataset_validation.py# CoNLL-U format & edge case validation (9 tests)
│   ├── test_imports.py           # Dependency verification (1 test)
│   └── test_phase1_hmm.py        # HMM model, Viterbi, and metric invariants (18 tests)
├── Makefile                      # Standardized project tasks
├── pyproject.toml                # Project dependencies and packaging metadata
├── uv.lock                       # Deterministic dependency lockfile
└── memory.md                     # Persistent project state, experiments, and handoffs
```

---

## Phase 1 Experimental Results

Hyperparameter tuning over add-$\alpha$ smoothing was conducted on the development split, selecting $\alpha^* = 0.1$. The model was then evaluated on the isolated test set:

| Evaluation Split | Smoothing $\alpha$ | Token Accuracy | Macro-averaged $F_1$ | Unknown-Word Accuracy | OOV Token Count |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Development** | 0.01 | 87.13% | 0.7716 | 24.92% | 2,087 (8.30%) |
| **Development** | **0.10** *(Selected)* | **87.41%** | **0.7817** | **29.52%** | **2,087 (8.30%)** |
| **Development** | 1.00 | 84.52% | 0.7491 | 33.45% | 2,087 (8.30%) |
| **Test (Final)** | **0.10** | **87.59%** | **0.7944** | **30.98%** | **2,292 (9.13%)** |

### Top Confusion Error Patterns (Test Set)
1. `NOUN -> PROPN`: 161 errors *(Capitalization and nominal ambiguity)*
2. `PROPN -> NOUN`: 153 errors *(Common nouns in named entity phrases)*
3. `PROPN -> X`: 136 errors *(Web/foreign tokens confused with proper nouns)*
4. `PROPN -> PUNCT`: 118 errors *(Special symbols and abbreviations in names)*
5. `NOUN -> X`: 112 errors *(Informal slang and web-domain tokens)*

---

## Getting Started

### Prerequisites
- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- GNU Make

### 1. Environment Setup
Clone the repository and synchronize the locked dependencies:
```bash
make setup
```

### 2. Run Test Suite
Run all unit and validation tests (31 tests):
```bash
make test
```

### 3. Run Phase 1 Experiment
Execute the complete HMM training, Dev hyperparameter selection, and Test evaluation pipeline:
```bash
uv run python experiments/phase1_hmm.py
```

### 4. Launch Notebooks
Launch the JupyterLab workspace:
```bash
make notebook
```
- Open `notebooks/eda.ipynb` for the exploratory data analysis.
- Open `notebooks/phase1_hmm.ipynb` for the interactive model walkthrough and David Fincher dialogue tagger.

---

## Theoretical References

1. **Nivre et al. (2020)**: Universal Dependencies v2: An Even More Useful Treebank.
2. **Silveira et al. (2014)**: A Gold Standard Dependency Corpus for English (English Web Treebank).
3. **Church (1988)**: A Stochastic Parts Program and Noun Phrase Parser for Unrestricted Text.
4. **Rabiner (1989)**: A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition.
5. **Brants (2000)**: TnT — A Statistical Part-of-Speech Tagger.
6. **Jurafsky & Martin (2026)**: Sequence Labeling for POS Tagging (Speech and Language Processing, Ch. 18).
7. **Manning, Raghavan, & Schütze (2008)**: Introduction to Information Retrieval (Ch. 8: Evaluation).
8. **Sokolova & Lapalme (2009)**: A Systematic Analysis of Performance Measures for Classification Tasks.
