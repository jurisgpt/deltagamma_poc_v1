# DeltaGamma POC v1

![CI/CD Status](https://github.com/jurisgpt/deltagamma_poc_v1/actions/workflows/code_quality.yml/badge.svg)

A PyTorch-based framework for biological knowledge graph analysis and drug-disease association prediction.

## 🚀 Features

- Graph-based feature extraction using Graph Attention Networks (GAT)
- Drug-disease association prediction
- Automated code quality checks and formatting
- Security scanning for Python code
- Type checking with mypy

## 🛠 Development Setup

### Prerequisites

- Python 3.11+
- pip 20.0+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/deltagamma_poc_v1.git
   cd deltagamma_poc_v1
   ```

2. **Set up a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r code_quality/requirements-code-quality.txt
   ```

4. **Install pre-commit hooks** (for automated code formatting and checks)
   ```bash
   pre-commit install
   ```

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=.


# Run specific test file
pytest tests/test_module.py
```

## 🔍 Code Quality

### Pre-commit Hooks

Pre-commit hooks automatically run on each commit to ensure code quality:
- Black code formatting
- isort import sorting
- Flake8 linting
- mypy type checking
- Bandit security scanning
- pyupgrade for modern Python syntax
- pycln for removing unused imports
- detect-secrets for finding secrets in code

### Manual Checks

```bash
# Format code with Black
black .


# Sort imports with isort
isort .


# Run linter
flake8 .


# Run type checker
mypy .


# Run security scanner
bandit -r . -c .bandit
```

## 🤖 CI/CD

This project uses GitHub Actions for continuous integration. The following checks run on every push and pull request:

- Python 3.11 compatibility
- Test suite execution
- Code coverage reporting
- Code quality checks
- Security scanning

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
