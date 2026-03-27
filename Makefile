.PHONY: help lint lint-fix format check test test-cov pre-commit

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

lint: ## Run linter (ruff check)
	uv run ruff check .

lint-fix: ## Run linter with auto-fix
	uv run ruff check --fix .

format: ## Format code (ruff format)
	uv run ruff format .

test: ## Run tests
	uv run pytest

test-cov: ## Run tests with coverage report
	uv run pytest --cov=ollama_adapter --cov-report=term-missing

check: ## Run all checks (format, lint, mypy, tests)
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy ollama_adapter/
	uv run pytest

pre-commit: format lint-fix check ## Format, fix, then run all checks
