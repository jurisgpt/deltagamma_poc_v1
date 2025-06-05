# Code Quality Checker

This directory contains tools for checking code quality in the project using various Python static analysis tools.

## Setup

1. Install the required tools:
   ```bash
   pip install -r requirements-code-quality.txt
   ```

## Usage

Run the code quality checker from the project root:

```bash
python code_quality/CodeQualityChecker.py
```

### Command Line Options

- `--path`: Path to the directory containing Python files to check (default: current directory)
- `--exclude`: Directories to exclude from checking (default: ['venv', '.git', '__pycache__', '.pytest_cache', '.mypy_cache'])
- `--output`: Output report file (default: CodeQualityReport.txt)

Example:
```bash
python code_quality/CodeQualityChecker.py --path . --exclude venv test_data --output my_report.txt
```

## Tools Used

- **pylint**: Static code analysis for Python that looks for programming errors and helps enforce a coding standard
- **mypy**: Optional static type checker for Python
- **black**: Python code formatter (check mode only)
- **flake8**: A wrapper around PyFlakes, pycodestyle, and McCabe script
- **pycodestyle**: Style guide checker for Python code (PEP 8)
- **pydocstyle**: Python docstring style checker (PEP 257)
- **bandit**: Security linter for Python code

## Report

The tool generates a comprehensive report in `CodeQualityReport.txt` (or the specified output file) with the results from all tools.

## Continuous Integration

To integrate this into your CI/CD pipeline, you can add a step like this to your workflow:

```yaml
- name: Run Code Quality Checks
  run: |
    pip install -r code_quality/requirements-code-quality.txt
    python code_quality/CodeQualityChecker.py
```
