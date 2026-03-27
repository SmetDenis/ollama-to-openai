.PHONY: lint lint-fix format check

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check --fix .

format:
	uv run ruff format .

check:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy ollama_adapter/
