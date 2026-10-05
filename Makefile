.PHONY: setup test notebook

setup:
	uv sync

test:
	uv run python tests/test_dataset_loader.py
	uv run python -m unittest discover -s tests -p 'test_imports.py' -v

notebook:
	uv run jupyter lab