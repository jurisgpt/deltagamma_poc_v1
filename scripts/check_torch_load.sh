#!/bin/bash

# Script to check for unsafe torch.load() usage
# Exit code 0 means no unsafe usage found
# Exit code 1 means unsafe usage found

echo "🔍 Scanning for unsafe torch.load() usage..."

# Find all Python files and check for torch.load() usage, excluding venv/, .venv/, and test/ directories
UNSAFE_USAGE=$(grep -r "torch\\.load(" --include="*.py" \
    --exclude-dir=venv \
    --exclude-dir=.venv \
    --exclude-dir=test \
    --exclude-dir=tests \
    --exclude-dir=__pycache__ \
    . | grep -v "pickle_module" | grep -v "map_location" | grep -v "weights_only")

if [ -z "$UNSAFE_USAGE" ]; then
    echo "✅ No unsafe torch.load() usage found"
    exit 0
else
    echo "❌ Potentially unsafe torch.load() usage found:"
    echo "$UNSAFE_USAGE"
    echo ""
    echo "💡 Recommendation: Use torch.load() with weights_only=True for security"
    echo "Example: torch.load('model.pt', weights_only=True)"
    echo ""
    echo "If you need to load untrusted data, make sure to use a safe unpickler"
    echo "and validate the input data carefully."
    exit 1
fi
