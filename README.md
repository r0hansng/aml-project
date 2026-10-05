# AML Project

![Python](https://img.shields.io/badge/python-3.13%2B-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/managed%20with-uv-6C47FF?logo=uv&logoColor=white)
![Status](https://img.shields.io/badge/status-in%20development-F2C94C)

Exploratory analysis and NLP experimentation over the Universal Dependencies
English Web Treebank (UD English EWT). The project focuses on reproducible
dataset handling, corpus statistics, POS-tag distributions, vocabulary
coverage, unknown-token analysis, and word/tag ambiguity.

## Highlights

- Sentence-preserving CoNLL-U loading with `FORM` and `UPOS` extraction
- Reproducible train, development, and test split handling
- Dataset size, sentence-length, POS, vocabulary, and ambiguity analysis
- Unknown word-type and unknown-token coverage measurements
- Notebook-first workflow with scriptable setup and validation commands

## Repository Layout

```text
aml-project/
├── data/                 # Official UD English EWT splits
├── experiments/          # Experiment workspaces
├── notebooks/            # Exploratory analysis notebooks
├── reports/              # Generated reports and findings
├── src/                  # Reusable project code
├── tests/                # Import and loader validation
├── Makefile              # Setup, test, and notebook commands
├── pyproject.toml        # Project metadata and dependencies
└── uv.lock               # Reproducible dependency lockfile
```

## Dataset

The `data/` directory contains the official UD English EWT train, development,
and test splits. Original sentence boundaries and annotations are preserved;
the official split is not regenerated or randomly repartitioned.

- Source: [Universal Dependencies English EWT](https://github.com/UniversalDependencies/UD_English-EWT)
- Revision: `4a4d77f599ea53cc405f85d0cec4b2f14f81d42b`
- Repository context: `r2.18-11-g4a4d77f5`

## Requirements

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- GNU Make

## Quick Start

Set up the environment from the repository root:

```bash
make setup
```

This synchronizes the project environment from `pyproject.toml` and
`uv.lock`. To activate the environment in the current shell:

```bash
source .venv/bin/activate
```

Run the loader validation and import smoke test:

```bash
make test
```

Launch the exploratory notebook environment:

```bash
make notebook
```

## Analysis Workflow

The main exploratory notebook is [`notebooks/eda.ipynb`](notebooks/eda.ipynb).
It loads the three official splits and reports:

- Sentence and token counts
- Vocabulary size and training-only vocabulary coverage
- Sentence-length statistics and visualization
- UPOS tag frequencies
- Development and test unknown-token rates
- Word-level POS ambiguity and tag-frequency distributions

The notebook is designed to keep the sentence-preserving representation
available while creating flattened token views only for token-level statistics.

## Development Commands

| Command | Purpose |
| --- | --- |
| `make setup` | Synchronize the locked project environment |
| `make test` | Run loader validation and dependency import tests |
| `make notebook` | Start JupyterLab |

For direct, project-managed Python execution, use `uv run`, for example:

```bash
uv run python tests/test_dataset_loader.py
```
