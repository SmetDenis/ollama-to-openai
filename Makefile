.PHONY: lint lint-fix format check

lint:
	ruff check .

lint-fix:
	ruff check --fix .

format:
	ruff format .

check:
	ruff format --check .
	ruff check .
