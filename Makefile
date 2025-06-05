.PHONY: help install format lint typecheck security test all

help:
	@echo "Makefile targets:"
	@echo "  make install      - Install dependencies"
	@echo "  make format       - Run black to autoformat code"
	@echo "  make lint         - Run flake8 for code linting"
	@echo "  make typecheck    - Run mypy for static type checks"
	@echo "  make security     - Run bandit for security scanning"
	@echo "  make test         - Run unittests in ./tests"
	@echo "  make all          - Run all quality checks"

install:
	pip install -r requirements.txt -r code_quality/requirements-code-quality.txt

format:
	black .

lint:
	flake8 .

typecheck:
	mypy .

security:
	bandit -r . -x tests

test:
	python -m unittest discover -s tests -p '*.py'

all: format lint typecheck security test
