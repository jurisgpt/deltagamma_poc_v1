#!/usr/bin/env python3
import re
import sys
from pathlib import Path


def fix_bandit_yaml(file_path):
    """
    Fix YAML formatting in a Bandit configuration file.
    Converts comma-separated entries to proper YAML list format.
    """
    try:
        # Read the file content
        with open(file_path) as f:
            content = f.read()

        # Create a backup of the original file
        backup_path = str(file_path) + ".bak"
        with open(backup_path, "w") as f:
            f.write(content)
        print(f"Created backup at: {backup_path}")

        # Fix the YAML list formatting
        # Pattern to match:
        # 1. Start of line with optional whitespace
        # 2. B followed by digits
        # 3. Optional comma and whitespace
        # 4. Comment
        pattern = r"^(\s*)(B\d+)(?:\s*,\s*)?(\s*#.*)?$"

        def replace_match(match):
            # Reconstruct the line with proper YAML list format
            indent = match.group(1)
            code = match.group(2)
            comment = match.group(3) or ""
            return f"{indent}- {code}{comment}"

        # Apply the replacement
        fixed_content = re.sub(pattern, replace_match, content, flags=re.MULTILINE)

        # Write the fixed content back to the file
        with open(file_path, "w") as f:
            f.write(fixed_content)

        print(f"Successfully fixed YAML formatting in: {file_path}")
        return True

    except Exception as e:
        print(f"Error: {str(e)}")
        return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fix_bandit_yaml.py <path_to_bandit_config>")
        sys.exit(1)

    file_path = Path(sys.argv[1]).expanduser()
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    success = fix_bandit_yaml(file_path)
    sys.exit(0 if success else 1)
