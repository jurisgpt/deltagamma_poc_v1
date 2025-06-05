#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

# 🔧 Configuration
TORCH_SAFE_VERSION="2.7.1-rc1"
TORCH_VULNERABLE_MAX="2.5.1"
TORCH_TARGET_REQUIREMENTS_FILE="requirements.txt"
SCAN_TARGETS=("embedding_extractor.py" "train_model.py" "graph_builder.py" "main.py")
TEMP_FREEZE_BEFORE="freeze_before_upgrade.txt"
TEMP_FREEZE_AFTER="freeze_after_upgrade.txt"

# 🎯 Step 1: Show Current Torch Version
echo "🧠 Current torch version:"
python -c "import torch; print(torch.__version__)"

# 🔎 Step 2: Check if torch is already patched
IS_PATCHED=$(python -c "import torch; from packaging import version; print(version.parse(torch.__version__) >= version.parse('$TORCH_SAFE_VERSION'))")

if [ "$IS_PATCHED" = "True" ]; then
    echo "✅ torch >= $TORCH_SAFE_VERSION already installed. Skipping upgrade."
else
    echo "⚠️ torch < $TORCH_SAFE_VERSION detected. Proceeding with secure upgrade..."
    echo "📦 Upgrading torch to version $TORCH_SAFE_VERSION"
    pip install --upgrade "torch==$TORCH_SAFE_VERSION"
fi

# 💾 Step 3: Freeze Environment for Traceability
echo "📄 Saving pre-upgrade package list..."
pip freeze > "$TEMP_FREEZE_BEFORE"

# 📁 Step 4: Verify torch.load usage (vulnerability check)
echo "🔍 Scanning for unsafe torch.load usage..."
echo "----------------------------------------"

VULNERABLE_LINES=0
for file in "${SCAN_TARGETS[@]}"; do
    if grep -q "torch.load" "$file"; then
        echo "⚠️  torch.load found in: $file"
        grep -n "torch.load" "$file" | grep -v "load_state_dict" | grep -v "jit.load" | tee /tmp/torch_load_findings.txt
        VULNERABLE_LINES=$(wc -l < /tmp/torch_load_findings.txt)
    fi
done

if [[ "$VULNERABLE_LINES" -gt 0 ]]; then
    echo "🚨 Potential unsafe use of torch.load detected. You MUST review these lines manually!"
    echo "Use safe alternatives like torch.jit.load or torch.load + load_state_dict"
    exit 1
else
    echo "✅ No unsafe torch.load patterns detected."
fi

# ✅ Step 5: Sanity run of primary scripts
echo "⚙️ Running main scripts for runtime verification..."

if python build_test_graph.py && python embedding_extractor.py && python train_model.py; then
    echo "✅ All main scripts executed successfully after upgrade."
else
    echo "❌ One or more scripts failed post-upgrade."
    exit 2
fi

# 📄 Step 6: Freeze and diff post-upgrade state
pip freeze > "$TEMP_FREEZE_AFTER"
echo "🧾 Comparing pre/post-upgrade package state..."
diff -u "$TEMP_FREEZE_BEFORE" "$TEMP_FREEZE_AFTER" || true

# 🧼 Cleanup
rm -f "$TEMP_FREEZE_BEFORE" "$TEMP_FREEZE_AFTER" /tmp/torch_load_findings.txt

echo "🎉 Upgrade + validation completed securely."


