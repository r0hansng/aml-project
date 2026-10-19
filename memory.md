# Project Memory

## Project Overview

This project performs exploratory data analysis and is intended to support POS-tagging models using the Universal Dependencies English Web Treebank (UD English EWT) dataset.

The current completed work is documented in:

- `notebooks/eda.ipynb`

The notebook loads CoNLL-U datasets, computes dataset statistics, analyzes sentence lengths and POS-tag distributions, measures unknown vocabulary, and investigates word/POS ambiguity.

---

## Project Structure

```text
aml-project/
├── data/
│   ├── train.conllu
│   ├── dev.conllu
│   └── test.conllu
├── notebooks/
│   └── eda.ipynb
├── experiments/
│   └── validate_dataset.py
├── src/
│   ├── dataset_loader.py
│   └── dataset_registry.py
├── tests/
│   ├── test_dataset_loader.py
│   ├── test_dataset_validation.py
│   └── test_imports.py
└── memory.md
```

### `data/`

Contains the UD English EWT dataset splits in CoNLL-U format.

- `train.conllu`
  - Training data.
  - Loaded into the `train` variable.
- `dev.conllu`
  - Development/validation data.
  - Loaded into the `dev` variable.
- `test.conllu`
  - Test data.
  - Loaded into the `test` variable.

Expected relative paths from the notebook:

```python
data_dir = Path("../data")
```

### `src/dataset_loader.py`

Provides the `load_conllu` function used to parse CoNLL-U files.

Import used by the notebook:

```python
from src.dataset_loader import load_conllu
```

The notebook adds the project root to `sys.path` because it is executed from the `notebooks/` directory:

```python
sys.path.insert(0, str(Path("..").resolve()))
```

The loader is expected to return sentences where each sentence is a list of `(word, tag)` tuples.

The loader currently:

- requires ten CoNLL-U columns;
- extracts `FORM` and `UPOS`;
- preserves sentence boundaries;
- ignores comments;
- skips multiword-token IDs such as `1-2`;
- skips empty-node IDs such as `3.1`;
- rejects malformed token IDs;
- supports Unix and Windows line endings;
- validates sentence structure and official UPOS tags.

The project uses the POS-tagging subset of the UD annotation scheme. It does not preserve all CoNLL-U fields such as `LEMMA`, `FEATS`, `HEAD`, or `DEPREL` because Phase 1 does not require dependency parsing.

### `src/dataset_registry.py`

Defines the official dataset paths relative to the project root:

```python
DATASET_SPLITS = {
  "train": DATA_DIRECTORY / "train.conllu",
  "dev": DATA_DIRECTORY / "dev.conllu",
  "test": DATA_DIRECTORY / "test.conllu",
}
```

`validate_dataset_files()` raises `FileNotFoundError` when an official split is missing.

### `experiments/validate_dataset.py`

Loads the registered train, development, and test files. The script adds the project root to `sys.path`, so it works when executed directly:

```bash
python experiments/validate_dataset.py
```

### `tests/test_dataset_validation.py`

Verifies:

- official split files exist;
- expected sentence counts remain unchanged;
- expected token counts remain unchanged;
- official UPOS tags are valid;
- comments, multiword tokens, empty nodes, and Windows line endings are handled;
- malformed column counts and token IDs are rejected;
- invalid UPOS tags and empty datasets are rejected.

### `notebooks/eda.ipynb`

Main exploratory analysis notebook.

It imports:

- `pathlib.Path`
- `collections.Counter`
- `collections.defaultdict`
- `statistics`
- `pandas`
- `matplotlib.pyplot`

---

## Dataset Loading

The notebook verifies that all three data files exist:

```python
print(train_path.exists())
print(dev_path.exists())
print(test_path.exists())
```

The datasets are loaded with:

```python
train = load_conllu(train_path)
dev = load_conllu(dev_path)
test = load_conllu(test_path)
```

Each split is flattened into a token list using:

```python
def get_tokens(sentences):
    return [token for sentence in sentences for token in sentence]
```

The resulting variables are:

- `train_tokens`
- `dev_tokens`
- `test_tokens`

Each token is represented as:

```python
(word, POS_tag)
```

---

## EDA Results

### Dataset Sizes

| Split | Sentences | Tokens |
|---|---:|---:|
| Train | 12,544 | 204,578 |
| Dev | 2,001 | 25,148 |
| Test | 2,077 | 25,094 |

The training split is substantially larger than the development and test splits.

### Vocabulary and POS Tags

The notebook calculates:

- Number of tokens
- Vocabulary size
- Number of POS tags

The training vocabulary contains:

- `19,674` distinct word types

### Sentence Lengths

Mean sentence lengths:

| Split | Mean Length |
|---|---:|
| Train | 16.31 |
| Dev | 12.57 |
| Test | 12.08 |

Training sentences are approximately four tokens longer on average than development and test sentences.

The notebook also calculates:

- Median sentence length
- Minimum sentence length
- Maximum sentence length
- Standard deviation

A histogram is generated for the training sentence-length distribution.

### POS-Tag Distribution

The notebook counts UPOS tags independently for each split and displays them in a table.

For the training set:

- `NOUN` is the most frequent POS tag, with `34,751` tokens.
- `X` is the least frequent POS tag, with `400` tokens.

Therefore, the POS-tag distribution is imbalanced.

A bar chart is generated for the training POS-tag distribution.

---

## Unknown-Word Analysis

The training vocabulary is defined as:

```python
train_vocab = set(word for word, tag in train_tokens)
```

Development and test vocabulary items not present in the training vocabulary are identified using set subtraction.

Results:

| Split | Unseen Word Types | Unknown Token Rate |
|---|---:|---:|
| Dev | 1,661 | 8.30% |
| Test | 1,836 | 9.13% |

Unknown words represent a meaningful part of evaluation data.

Implications:

- A model cannot rely only on direct word-to-tag memorization.
- HMM models need an unknown-word strategy.
- Neural models need an unknown-token representation or character/subword features.
- Character-level features may help generalize to unseen words.

---

## Word/POS Ambiguity Analysis

The notebook creates a mapping from each word to all POS tags observed with that word:

```python
word_tags = defaultdict(set)
```

Words occurring with multiple POS tags are considered ambiguous.

The analysis found:

- `1,973` word types occur with more than one POS tag.

Examples include:

- `to`
- `a`
- `in`
- `on`
- `run`

The notebook displays the top 30 ambiguous words, sorted by number of observed tags.

### Frequency-Aware Ambiguity

A second analysis counts how frequently each word occurs with each POS tag:

```python
word_tag_counts = defaultdict(Counter)
```

This shows that ambiguity is often unbalanced. For example, a word may occur mostly with one tag while appearing rarely with other tags.

This demonstrates that:

- A simple word-to-single-tag dictionary loses useful frequency information.
- Context is necessary for correct tagging.
- Sequence models can use neighboring tokens to resolve ambiguity.

---

## Important Notebook Variables

### Paths

- `data_dir`
- `train_path`
- `dev_path`
- `test_path`

### Loaded datasets

- `train`
- `dev`
- `test`

### Flattened tokens

- `train_tokens`
- `dev_tokens`
- `test_tokens`

### Vocabulary sets

- `train_vocab`
- `dev_vocab`
- `test_vocab`

### POS counts

- `train_pos_counts`
- `dev_pos_counts`
- `test_pos_counts`

### Sentence lengths

- `train_lengths`
- `dev_lengths`
- `test_lengths`

### Analysis tables

- `statistics_df`
- `sentence_length_df`
- `upos_counts_df`
- `unknown_token_df`
- `ambiguous_words_df`
- `frequency_ambiguity_df`

---

## Library Dependencies

The notebook currently uses:

```python
from pathlib import Path
from collections import Counter, defaultdict
import statistics as stats

import pandas as pd
import matplotlib.pyplot as plt
```

Required external libraries:

- `pandas`
- `matplotlib`

The project also requires a Python environment capable of running Jupyter notebooks.

The local project module dependency is:

```python
src.dataset_loader.load_conllu
```

---

## Execution Notes

The notebook assumes it is executed from the `notebooks/` directory or in an environment where:

```python
Path("../data")
```

resolves to the project’s `data/` directory.

If the notebook is executed from another working directory, the relative paths may fail. In that case, use an absolute project-root path or determine the project root dynamically.

The notebook currently verifies data-file existence before loading the files.

---

## Current Conclusions

1. The training split is the largest dataset split.
2. POS tags are not evenly distributed.
3. Training sentences are longer on average than development and test sentences.
4. Development and test data contain unseen word types.
5. Unknown words account for approximately 8–9% of evaluation tokens.
6. Many words occur with multiple POS tags.
7. Word/POS ambiguity is frequently imbalanced across tags.
8. A direct word-to-tag lookup is insufficient.
9. Future HMM and BiLSTM models need unknown-word handling.
10. Contextual and potentially character-level features are important for POS tagging.

---

## Recommended Next Steps

1. Define the training-only vocabulary and POS-tag mapping for the HMM.
2. Define special symbols such as `<START>`, `<END>`, and `<UNK>`.
3. Implement an HMM baseline with:
   - Transition probabilities
   - Emission probabilities
   - Smoothing
   - Unknown-word handling
4. Add Viterbi decoding and tests for sequence scores and backpointers.
5. Create evaluation utilities for:
  - Token-level accuracy
  - Per-tag precision, recall, and F1
  - Macro F1
  - Unknown-word accuracy
  - Confusion matrix
6. Keep the test set isolated until final evaluation.
7. Record model configurations, random seeds, and results for reproducibility.
8. Implement Phase 2 only after Phase 1 is complete. The BiLSTM baseline should include:
   - Word embeddings
   - `<UNK>` token handling
   - Optional character-level features

The detailed literature and implementation guidance is stored in:

- `references/phase1/summary_phase1.md`
- `references/phase2/summary_phase2.md`

Run the verified project checks with:

```bash
make test
```

The current test target passes the loader script, dataset validation tests, and dependency import tests.

---

## Project Requirements and Governance

### `project_instructions.md`

This file contains the official CSAI project requirements.

#### Team Requirements

- The team must contain exactly five members.
- Members must belong to the same section.
- Team and project selection become final after the stated deadlines.
- Each member is individually assessed during the viva voce.

#### Phase 1 — Machine Learning Evaluation

Required areas:

- Literature review
- Dataset quality and EDA
- Feature engineering
- Theoretical rigor
- Model application
- GitHub repository and code quality
- Presentation
- Individual viva voce
- LaTeX/IEEE project report

#### Phase 2 — Deep Learning Evaluation

Required areas:

- Architecture logic
- Hybrid innovation
- Ablation studies
- Deep learning literature review
- Dataset and regularization
- Technical validation
- Theoretical rigor
- GitHub repository and code quality
- Presentation
- Individual viva voce
- LaTeX/IEEE project report

A hybrid ML/DL approach qualifies for additional marks only if it is technically justified, successfully implemented, and demonstrates measurable value.

Every team member must understand the algorithms, implementation, experiments, results, theory, and their individual contribution.

---

## Project-Specific Requirements

### `project_specs.md`

`project_specs.md` is the authoritative project-specific specification and must be consulted before implementing models or experiments.

Future sessions must document from this file:

- Problem statement
- Required inputs and outputs
- Dataset requirements
- Required ML models
- Required DL models
- Hybrid-approach requirements
- Evaluation metrics
- Required experiments
- Deliverables
- Report and presentation requirements
- Constraints and grading criteria

The complete contents of `project_specs.md` must be inspected and summarized here before model implementation begins.

---

## Complete Project File Map

```text
aml-project/
├── data/
│   ├── train.conllu
│   ├── dev.conllu
│   └── test.conllu
├── notebooks/
│   └── eda.ipynb
├── src/
│   └── dataset_loader.py
├── memory.md
├── project_instructions.md
└── project_specs.md
```

### File Relationships

- `project_instructions.md` defines institutional evaluation and submission requirements.
- `project_specs.md` defines project-specific technical requirements.
- `data/*.conllu` provides the dataset consumed by `src/dataset_loader.py`.
- `src/dataset_loader.py` parses the dataset for `notebooks/eda.ipynb` and future models.
- `notebooks/eda.ipynb` performs EDA required for Phase 1.
- `memory.md` records project state, decisions, results, requirements, and session handoffs.

---

## Session Handoff Protocol

Every session must update this file before completion.

Each entry must record:

1. Date.
2. Objective.
3. Files created, modified, or deleted.
4. Implementation changes.
5. Commands, notebooks, or experiments executed.
6. Results and metrics.
7. Errors and resolutions.
8. Decisions to preserve.
9. Pending tasks.
10. Deviations from `project_instructions.md` or `project_specs.md`.
11. The exact next recommended action.

### Session Log

#### Initial EDA Session

- Completed EDA in `notebooks/eda.ipynb`.
- Loaded all UD English EWT dataset splits.
- Calculated dataset statistics, sentence lengths, POS distributions, unknown vocabulary, and word/POS ambiguity.
- Identified the need for unknown-word handling and contextual modeling.
- No ML or DL model has been implemented yet.

#### Requirements Documentation Session

- Added `project_instructions.md` to the project context.
- Recorded Phase 1 ML and Phase 2 DL evaluation requirements.
- Recorded hybrid-approach, repository, report, presentation, and viva requirements.
- Added `project_specs.md` as the authoritative technical specification.
- Established mandatory updates to this file after every session.

---

## Current Pending Tasks

1. Inspect and summarize `project_specs.md`.
2. Verify `src/dataset_loader.py`.
3. Add parser unit tests.
4. Implement an HMM baseline.
5. Implement a BiLSTM model.
6. Add unknown-word handling.
7. Add evaluation metrics and confusion matrices.
8. Plan and justify a hybrid approach if required.
9. Add DL and hybrid ablation studies.
10. Record reproducible experiment configurations and results.

### Required Future Entry Format

```markdown
#### Session — YYYY-MM-DD

**Objective:**  
...

**Files changed:**  
- `path/to/file`: ...

**Work completed:**  
- ...

**Experiments and results:**  
- ...

**Problems and resolutions:**  
- ...

**Decisions to preserve:**  
- ...

**Requirements impact:**  
- Confirmed compliance with `project_instructions.md` / `project_specs.md`.
- Or document any deviation.

**Pending tasks:**  
- ...

**Next recommended action:**  
...
```

