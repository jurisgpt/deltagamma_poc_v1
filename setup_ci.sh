#!/bin/bash
pip install -r requirements.txt -r code_quality/requirements-code-quality.txt
flake8 .
black . --check
mypy .
bandit -r . -x tests

