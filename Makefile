.PHONY: lint lint-fix format check test test-cov pre-commit

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check --fix .

format:
	uv run ruff format .

test:
	uv run pytest

test-cov:
	uv run pytest --cov=ollama_adapter --cov-report=term-missing

check:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy ollama_adapter/
	uv run pytest

pre-commit: format lint-fix check
