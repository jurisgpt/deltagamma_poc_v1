#!/bin/bash

# Precommit script to fix code quality issues before commits
set -e

echo "🔧 Running precommit checks and fixes..."

# Install code quality tools if not available
if ! command -v black &> /dev/null; then
    echo "📦 Installing code quality tools..."
    pip install black flake8 mypy bandit
fi

echo "🎨 Fixing code formatting with black..."
black .

echo "🔍 Running type checking with mypy..."
mypy . || {
    echo "❌ MyPy type checking failed. Please fix type annotation issues."
    exit 1
}

echo "🧹 Running linting with flake8..."
flake8 . || {
    echo "❌ Flake8 linting failed. Please fix code style issues."
    exit 1
}

echo "✅ All precommit checks passed!"
