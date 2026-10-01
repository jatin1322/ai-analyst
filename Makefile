.PHONY: install test lint fmt clean

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests

fmt:
	ruff format src tests

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} +
