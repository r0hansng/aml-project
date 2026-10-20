# Project Memory

## Project Overview

This project implements Part-of-Speech (POS) tagging and sequence labeling models on the Universal Dependencies English Web Treebank (UD English-EWT) dataset.

The project is structured into two main phases:
1. **Phase 1 (Machine Learning Baseline — Completed)**:
   - Comprehensive exploratory data analysis (`notebooks/eda.ipynb`).
   - Generative First-Order Hidden Markov Model (HMM) with add-$\alpha$ smoothing and sentence boundaries (`src/hmm_model.py`, `src/preprocessing.py`).
   - Log-space Viterbi dynamic programming decoding with backpointer path reconstruction.
   - Rigorous evaluation protocol maintaining two distinct vocabulary concepts ($V_{\text{original}}$ vs. $V_{\text{model}}$) and reporting Token Accuracy, Macro $F_1$, per-tag metrics, unknown-word accuracy, and confusion matrices (`src/evaluation.py`).
   - Hyperparameter tuning and test execution scripts (`experiments/phase1_hmm.py`).
   - End-to-end self-explanatory demonstration notebook with publication-grade visualizations and interactive tagger using David Fincher film dialogues (`notebooks/phase1_hmm.ipynb`).
   - 31 automated tests passing (`tests/test_phase1_hmm.py`, `tests/test_dataset_validation.py`, etc.).
2. **Phase 2 (Deep Learning Sequence Models — Pending)**:
   - BiLSTM baseline with pre-trained word embeddings and character/subword representations.
   - Hybrid ML/DL architectural exploration and ablation studies.

### Team Specifications

| Property | Value |
| :--- | :--- |
| **Team ID** | **T058** |
| **Section** | **A** |
| **Project** | **Project 16**: Part-of-Speech Tagging: Hidden Markov Model vs Bidirectional LSTM Tagger |
| **Degree** | **B. Tech in Computer Science & Artificial Intelligence** |
| **Institution** | Newton School of Technology, Rishihood University |

#### Team Members
1. **Rohan Singh** (`2401010391`) — `rohan.singh2024@nst.rishihood.edu.in`
2. **Aniket Pathak** (`2401020088`) — `aniket.pathak2024@nst.rishihood.edu.in`
3. **Pushpendra Singh Parihar** (`2401010361`) — `pushpendra.parihar2024@nst.rishihood.edu.in`
4. **Kapil Karan Mathur** (`2401010209`) — `kapil.mathur2024@nst.rishihood.edu.in`
5. **Agrim Kumar Malhotra** (`2401010045`) — `agrim.malhotra2024@nst.rishihood.edu.in`

---

## Complete Project File Map

```text
aml-project/
├── data/                         # Official UD English EWT CoNLL-U splits
│   ├── train.conllu              # Train split (12,544 sents, 204,578 tokens)
│   ├── dev.conllu                # Dev split (2,001 sents, 25,148 tokens)
│   └── test.conllu               # Test split (2,077 sents, 25,094 tokens)
├── experiments/                  # Experiment runner scripts
│   └── phase1_hmm.py             # Phase 1 HMM tuning & isolated test evaluation
├── notebooks/                    # Interactive analysis notebooks
│   ├── eda.ipynb                 # Exploratory data analysis & corpus statistics
│   └── phase1_hmm.ipynb          # End-to-end HMM walkthrough, math, and interactive demo
├── references/                   # Standardized academic literature
│   ├── phase1/                   # Phase 1 reference papers (01_nivre ... 10_sokolova)
│   │   ├── paper_implementation_guide.md
│   │   └── summary_phase1.md
│   └── phase2/                   # Phase 2 reference papers (01_graves ... 06_vaswani)
│       ├── paper_implementation_guide.md
│       └── summary_phase2.md
├── src/                          # Modular core package
│   ├── dataset_loader.py         # CoNLL-U parser, token extraction, and UPOS validation
│   ├── dataset_registry.py       # Official dataset paths & verification
│   ├── evaluation.py             # Token accuracy, Macro F1, unknown accuracy, confusion matrix
│   ├── hmm_model.py              # First-order HMM with log-space Viterbi decoder
│   └── preprocessing.py          # HMMVocabulary, boundary tags, and UNK normalization
├── tests/                        # Comprehensive automated test suite
│   ├── test_dataset_loader.py    # Basic loader sanity checks
│   ├── test_dataset_loader_unit.py# Unit tests for loader edge cases
│   ├── test_dataset_validation.py# CoNLL-U format & edge case validation (9 tests)
│   ├── test_imports.py           # Dependency smoke test (1 test)
│   └── test_phase1_hmm.py        # HMM model, Viterbi, and metric invariants (18 tests)
├── Makefile                      # Standardized project tasks (setup, test, notebook)
├── pyproject.toml                # Project dependencies and packaging metadata
├── uv.lock                       # Deterministic dependency lockfile
├── README.md                     # User-facing documentation and experiment results
├── project_instructions.md       # Institutional course requirements
├── project_specs.md              # Technical specification
└── memory.md                     # Persistent project state, experiments, and handoffs
```

---

## Phase 1 Architecture and Technical Specifications

### 1. Two-Vocabulary Separation Invariant
To ensure strict scientific integrity in unknown-word evaluation (Sokolova & Lapalme, 2009; Brants, 2000), the codebase maintains two distinct vocabulary concepts:
- **$V_{\text{original}}$ (Empirical Training Vocabulary)**:
  - Extracted via `extract_original_vocabulary(train_sentences)`.
  - Exactly 19,674 distinct word types.
  - **Strict invariant**: Must never contain `<UNK>`.
  - Passed to `evaluate_predictions(..., training_vocabulary=V_original)` to identify true out-of-vocabulary tokens in development and test sets.
- **$V_{\text{model}}$ (Model Vocabulary)**:
  - Stored in `HMMVocabulary.words`.
  - Contains all words meeting the frequency threshold (default: $\ge 1$) plus the special out-of-vocabulary placeholder `<UNK>`.
  - Used for parameter matrix sizing and emission smoothing: $|V_{\text{model}}| = 19,675$.

### 2. Generative First-Order HMM Formulation
- Joint probability factorization (Church, 1988; Rabiner, 1989):
  $$P(W, T) = P(t_1 \mid \langle\text{START}\rangle) P(w_1 \mid t_1) \prod_{k=2}^n P(t_k \mid t_{k-1}) P(w_k \mid t_k) P(\langle\text{END}\rangle \mid t_n)$$
- Sentence boundaries: `START_TAG = "<START>"`, `END_TAG = "<END>"`.
- Add-$\alpha$ Laplace smoothing (Brants, 2000):
  $$P(t_j \mid t_i) = \frac{C(t_i, t_j) + \alpha}{C(t_i) + \alpha \cdot |S_{\text{next}}|}$$
  $$P(w \mid t) = \frac{C(t, w) + \alpha}{C(t) + \alpha \cdot |V_{\text{model}}|}$$
- Numerically stable log-space Viterbi decoding with dynamic programming backpointer trellis (Jurafsky & Martin, 2026):
  $$v_t(j) = \max_{i \in S} \left[ v_{t-1}(i) + \log P(t_j \mid t_i) \right] + \log P(w_t \mid t_j)$$
  Backpointer: $bp_t(j) = \arg\max_{i \in S} \left[ v_{t-1}(i) + \log P(t_j \mid t_i) \right]$.

---

## Phase 1 Experimental Results (UD English-EWT)

### Hyperparameter Tuning on Development Set (Manning et al., 2008)
| Smoothing $\alpha$ | Token Accuracy | Macro $F_1$ | Unknown-Word Accuracy | OOV Tokens |
| :---: | :---: | :---: | :---: | :---: |
| $\alpha = 0.01$ | 87.13% | 0.7716 | 24.92% | 2,087 (8.30%) |
| **$\alpha = 0.10$** *(Best)* | **87.41%** | **0.7817** | **29.52%** | **2,087 (8.30%)** |
| $\alpha = 1.00$ | 84.52% | 0.7491 | 33.45% | 2,087 (8.30%) |

*Decision*: $\alpha = 0.1$ selected based on highest Development Macro $F_1$ ($0.7817$).

### Final Evaluation on Isolated Test Set (Sokolova & Lapalme, 2009)
Evaluated strictly once on `test.conllu` (2,077 sentences, 25,094 tokens):
- **Overall Token Accuracy**: `87.59%`
- **Macro-averaged $F_1$**: `0.7944`
- **Unknown-Word Token Accuracy**: `30.98%` (670 / 2,292 tokens correct)
- **Total Unknown Tokens**: `2,292` ($9.13\%$ unseen token rate)

### Per-Tag Performance Breakdown (Test Set)
| UPOS Tag | Precision | Recall | $F_1$-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `CCONJ` | 0.97 | 0.98 | 0.98 | 736 |
| `PRON` | 0.93 | 0.97 | 0.95 | 2,164 |
| `PUNCT` | 0.93 | 0.98 | 0.95 | 3,096 |
| `DET` | 0.91 | 0.97 | 0.94 | 1,897 |
| `AUX` | 0.89 | 0.98 | 0.93 | 1,543 |
| `ADP` | 0.89 | 0.94 | 0.92 | 2,025 |
| `PART` | 0.86 | 0.93 | 0.90 | 649 |
| `VERB` | 0.91 | 0.86 | 0.89 | 2,605 |
| `NOUN` | 0.92 | 0.82 | 0.87 | 4,123 |
| `ADJ` | 0.87 | 0.84 | 0.85 | 1,788 |
| `ADV` | 0.81 | 0.85 | 0.83 | 1,191 |
| `NUM` | 0.78 | 0.69 | 0.73 | 542 |
| `PROPN` | 0.83 | 0.61 | 0.70 | 2,075 |
| `SCONJ` | 0.70 | 0.69 | 0.69 | 384 |
| `INTJ` | 0.58 | 0.79 | 0.66 | 121 |
| `SYM` | 0.50 | 0.80 | 0.62 | 113 |
| `X` | 0.05 | 0.52 | 0.10 | 42 |

### Top Confusion Error Patterns
1. `NOUN -> PROPN`: 161 errors *(Capitalization and nominal ambiguity)*
2. `PROPN -> NOUN`: 153 errors *(Common nouns within proper named phrases)*
3. `PROPN -> X`: 136 errors *(Web and foreign tokens confused with proper nouns)*
4. `PROPN -> PUNCT`: 118 errors *(Special symbols and abbreviations in names)*
5. `NOUN -> X`: 112 errors *(Informal slang, typos, and web-domain tokens)*

---

## Interactive Demonstrations (David Fincher Dialogues)

The interactive tagger (`notebooks/phase1_hmm.ipynb` Cell 26) evaluates iconic dialogues from David Fincher films:
1. ***Fight Club* (1999)**:
   - *"The first rule of Fight Club is : you do not talk about Fight Club ."*
   - Model correctly infers `Fight` $\rightarrow$ `PROPN` (unseen token `<UNK>`) based on capitalization and local transition to `Club`.
2. ***The Social Network* (2010)**:
   - *"If you guys were the inventors of Facebook , you would have invented Facebook ."*
   - Model correctly tags `If` $\rightarrow$ `SCONJ`, `inventors` $\rightarrow$ `NOUN` (`<UNK>` token correctly resolved via preceding determiner `the`), `Facebook` $\rightarrow$ `PROPN`, `invented` $\rightarrow$ `VERB`.
3. ***Se7en* (1995)**:
   - *"What's in the box ?"*
   - Model correctly parses contractions and interrogative structure: `What` $\rightarrow$ `PRON`, `'` $\rightarrow$ `PUNCT`, `s` $\rightarrow$ `AUX`, `in` $\rightarrow$ `ADP`, `the` $\rightarrow$ `DET`, `box` $\rightarrow$ `NOUN`.
4. ***Fight Club* (1999)**:
   - *"The things you own end up owning you ."*
   - Model correctly predicts syntactic transitions: `things` $\rightarrow$ `NOUN`, `own` $\rightarrow$ `ADJ`, `end` $\rightarrow$ `NOUN`, `up` $\rightarrow$ `ADV`, `owning` $\rightarrow$ `VERB`.

---

## Session Log

#### Session 1 — Initial EDA
- Completed exploratory analysis in `notebooks/eda.ipynb`.
- Loaded official UD English EWT dataset splits.
- Calculated dataset statistics, sentence lengths, POS distributions, unknown vocabulary, and word/POS ambiguity.
- Identified the necessity of unknown-word handling and contextual modeling.

#### Session 2 — Requirements & Governance Documentation
- Added `project_instructions.md` and `project_specs.md`.
- Recorded evaluation requirements for Phase 1 ML and Phase 2 DL.
- Established mandatory updates to `memory.md`.

#### Session 3 — Reference Standardization & Phase 1 HMM Pipeline
- Standardized PDF filenames across `references/phase1/` (01 to 10) and `references/phase2/` (01 to 06).
- Implemented `src/preprocessing.py`, `src/hmm_model.py`, `src/evaluation.py`, and `experiments/phase1_hmm.py`.
- Enforced strict separation of $V_{\text{original}}$ (without `<UNK>`) and $V_{\text{model}}$ (with `<UNK>`).
- Built 18 unit tests in `tests/test_phase1_hmm.py`; total 31 tests passing across the project.
- Verified test set performance: Token Accuracy `87.59%`, Macro $F_1$ `0.7944`, Unknown Accuracy `30.98%`.

#### Session 4 — Code Refactoring, Publication Visualizations & Fincher Dialogues
- Refactored all source files (`src/`, `experiments/`, `tests/`) to clean, idiomatic Python with self-explanatory names, textbook Viterbi backpointer decoding, and high-signal comments only at mathematical steps.
- Upgraded all graphs in `notebooks/eda.ipynb` and `notebooks/phase1_hmm.ipynb` to 150 DPI publication quality (custom color palettes, optimal Viterbi path border overlays, dual-axis tuning curves, and per-class recall heatmaps).
- Replaced placeholder interactive tagger inputs with iconic David Fincher film dialogues (*The Social Network*, *Fight Club*, *Se7en*).
- Fully executed and saved both notebooks with all cell outputs rendered.
- Updated `README.md` and `memory.md` to reflect full project state.

#### Session 5 — Phase 1 Project Report (Markdown & IEEE LaTeX)
- Generated comprehensive Phase 1 Project Report in `reports/phase1_project_report.md` with:
  - System architecture flowcharts and Bayesian network representations in Mermaid.
  - Mathematical formalization of first-order joint decomposition, add-$\alpha$ Laplace smoothing, and log-space Viterbi dynamic programming trellis.
  - Complete corpus statistics and Zipfian tag skew tables.
  - Quantitative tuning and test performance tables (Token Acc 87.59%, Macro F1 0.7944, Unknown Acc 30.98%).
  - Detailed per-tag classification report across all 17 UPOS classes.
  - Linguistic confusion matrix error analysis (`NOUN` $\leftrightarrow$ `PROPN`, `PROPN` $\rightarrow$ `X`).
  - Qualitative dialogue inference evaluation on David Fincher film quotes (*The Social Network*, *Fight Club*, *Se7en*).
  - Theoretical roadmap and hybrid architectural proposals for Phase 2.
  - Team division of responsibilities (5 members) and 4 core viva voce question-and-answer pairs.
- Generated matching IEEE Conference formatted LaTeX report in `reports/phase1_project_report.tex` adhering strictly to course evaluation requirements.

---

## Current Pending Tasks (Transition to Phase 2)

1. **Phase 2 Deep Learning Architecture Planning**:
   - BiLSTM baseline implementation for sequence labeling.
   - Word representation strategy: GloVe / FastText pre-trained embeddings + character-level CNN/LSTM or subword embeddings.
   - `<UNK>` token handling and out-of-vocabulary generalization.
2. **Hybrid ML/DL Exploration**:
   - HMM-informed feature injection, transition matrix constraint priors, or ensemble decoding.
   - Theoretical justification and empirical value demonstration as required by project instructions.
3. **Phase 2 Ablation Studies**:
   - Effect of bidirectional context (Forward LSTM vs. BiLSTM).
   - Effect of character-level subword representations on unknown-word accuracy.
   - Effect of pre-trained vs. randomly initialized embeddings.
4. **Phase 2 Presentation & Report Drafting**:
   - Compile Phase 1 vs. Phase 2 comparative benchmark table.
   - Finalize IEEE report with Deep Learning and Hybrid ablation results.

