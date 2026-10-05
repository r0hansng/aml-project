.PHONY: setup test notebook

setup:
	uv sync

test:
	uv run python -m unittest discover -s tests -v

notebook:
	uv run jupyter lab