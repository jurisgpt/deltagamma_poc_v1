#!/usr/bin/env python3
"""
Code Quality Checker

This script performs various code quality checks on Python source files using multiple tools:
- pylint: Static code analysis for Python that looks for programming errors, helps enforce a coding standard
- mypy: Optional static type checker for Python
- black: Python code formatter
- flake8: A wrapper around PyFlakes, pycodestyle, and McCabe script
- pycodestyle: Style guide checker for Python code
- pydocstyle: Python docstring style checker
- bandit: Security linter for Python code
"""

import argparse
import os
import subprocess  # nosec - Used for running code quality tools as subprocesses
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Default configuration
DEFAULT_EXCLUDES = ["venv", ".git", "__pycache__", ".pytest_cache", ".mypy_cache"]
REPORT_FILE = "CodeQualityReport.txt"
PYTHON_FILES = "*.py"


def run_command(cmd: List[str], cwd: Optional[str] = None) -> Tuple[int, str]:
    """
    Run a shell command and return the exit code and output.

    Note: This function uses subprocess.run() with shell=False (default) and doesn't use
    untrusted input for the command. All commands are hardcoded in the script.
    """
    try:
        # Using subprocess.run() with a list of arguments and shell=False is safe
        # as long as the command is not constructed from untrusted input.
        result = subprocess.run(  # nosec - Command is hardcoded, not from user input
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return result.returncode, result.stdout
    except Exception as e:
        return 1, str(e)


def find_python_files(
    directory: str, exclude: List[str] = DEFAULT_EXCLUDES
) -> List[str]:
    """Find all Python files in the given directory, excluding specified patterns."""
    exclude = exclude or DEFAULT_EXCLUDES

    python_files = []
    for root, _, files in os.walk(directory):
        # Skip excluded directories
        if any(excl in root for excl in exclude):
            continue

        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))

    return python_files


def run_pylint(files: List[str], report_file: str) -> Tuple[int, str]:
    """Run pylint on the given files."""
    if not files:
        return 0, "No Python files found to analyze with pylint"

    cmd = ["pylint", "--output-format=text"] + files
    return run_command(cmd)


def run_mypy(files: List[str], report_file: str) -> Tuple[int, str]:
    """Run mypy on the given files."""
    if not files:
        return 0, "No Python files found to analyze with mypy"

    cmd = ["mypy"] + files
    return run_command(cmd)


def run_black_check(files: List[str], report_file: str) -> Tuple[int, str]:
    """Check if files are formatted with black."""
    if not files:
        return 0, "No Python files found to check with black"

    cmd = ["black", "--check"] + files
    return run_command(cmd)


def run_flake8(files: List[str], report_file: str) -> Tuple[int, str]:
    """Run flake8 on the given files."""
    if not files:
        return 0, "No Python files found to analyze with flake8"

    cmd = ["flake8"] + files
    return run_command(cmd)


def run_pycodestyle(files: List[str], report_file: str) -> Tuple[int, str]:
    """Run pycodestyle on the given files."""
    if not files:
        return 0, "No Python files found to analyze with pycodestyle"

    cmd = ["pycodestyle"] + files
    return run_command(cmd)


def run_pydocstyle(files: List[str], report_file: str) -> Tuple[int, str]:
    """Run pydocstyle on the given files."""
    if not files:
        return 0, "No Python files found to analyze with pydocstyle"

    cmd = ["pydocstyle"] + files
    return run_command(cmd)


def run_bandit(files: List[str], report_file: str) -> Tuple[int, str]:
    """Run bandit on the given files."""
    if not files:
        return 0, "No Python files found to analyze with bandit"

    cmd = ["bandit", "-r"] + files
    return run_command(cmd)


def write_report(report_file: str, results: Dict[str, Tuple[int, str]]) -> None:
    """Write the code quality report to a file."""
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(
            f"Code Quality Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write("=" * 50 + "\n\n")

        total_issues = 0

        for tool, (exit_code, output) in results.items():
            f.write(f"{tool.upper()} Results (Exit Code: {exit_code}):\n")
            f.write("-" * 40 + "\n")
            f.write(output)
            f.write("\n\n")

            # Count issues (non-zero exit code indicates issues for most tools)
            if exit_code != 0 and output.strip():
                total_issues += 1

        f.write("\n" + "=" * 50 + "\n")
        f.write(f"Total tools with issues: {total_issues} out of {len(results)}\n")
        f.write("=" * 50 + "\n")


def main() -> int:
    """Main function to run all code quality checks."""
    parser = argparse.ArgumentParser(
        description="Run code quality checks on Python files."
    )
    parser.add_argument(
        "--path",
        type=str,
        default=".",
        help="Path to the directory containing Python files to check",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=DEFAULT_EXCLUDES,
        help="Directories to exclude from checking",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=REPORT_FILE,
        help=f"Output report file (default: {REPORT_FILE})",
    )

    args = parser.parse_args()

    # Find all Python files
    python_files = find_python_files(args.path, args.exclude)

    if not python_files:
        print(f"No Python files found in {args.path}")
        return 1

    print(f"Found {len(python_files)} Python files to analyze...")

    # Run all code quality tools
    results = {}

    print("Running pylint...")
    results["pylint"] = run_pylint(python_files, args.output)

    print("Running mypy...")
    results["mypy"] = run_mypy(python_files, args.output)

    print("Running black...")
    results["black"] = run_black_check(python_files, args.output)

    print("Running flake8...")
    results["flake8"] = run_flake8(python_files, args.output)

    print("Running pycodestyle...")
    results["pycodestyle"] = run_pycodestyle(python_files, args.output)

    print("Running pydocstyle...")
    results["pydocstyle"] = run_pydocstyle(python_files, args.output)

    print("Running bandit...")
    results["bandit"] = run_bandit(python_files, args.output)

    # Write the report
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    write_report(report_path, results)

    print(f"\nCode quality check completed. Report saved to: {report_path}")

    # Return non-zero if any tool found issues
    if any(exit_code != 0 for exit_code, _ in results.values()):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
