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

echo "✅ Precommit checks completed!"
