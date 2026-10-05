# AML Project

## Dataset

The `data/` directory contains the official Universal Dependencies English EWT
train, development, and test splits. The original sentence boundaries and
annotations are preserved.

Source repository: <https://github.com/UniversalDependencies/UD_English-EWT>

Dataset revision: `4a4d77f599ea53cc405f85d0cec4b2f14f81d42b` (`r2.18-11-g4a4d77f5`)

## Prerequisites

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- GNU Make

## Setup

From the project directory, run:

```bash
make setup
```

This creates the project virtual environment and installs the dependencies from
`uv.lock`.

To activate the environment in your current shell:

```bash
source .venv/bin/activate
```

## Verify the setup

Run the import smoke test with:

```bash
make test
```

## Start JupyterLab

To open the notebooks in JupyterLab:

```bash
make notebook
```
