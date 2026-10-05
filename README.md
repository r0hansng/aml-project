# AML Project

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
