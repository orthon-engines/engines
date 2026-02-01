"""
PRISM Classification Audit

Scans codebase for classification logic that should not exist.
PRISM computes. PRISM never classifies.

Usage:
    python -m prism.scripts.audit_classification [--fix]
"""

import re
import sys
from pathlib import Path
from typing import List
from dataclasses import dataclass


@dataclass
class Violation:
    file: str
    line_num: int
    line: str
    violation_type: str
    severity: str  # 'error' or 'warning'


# ============================================================
# PATTERNS TO FLAG
# ============================================================

# Classification function names
CLASSIFICATION_FUNCTIONS = [
    r'def classify_',
    r'def get_.*_type\(',
    r'def get_.*_class\(',
    r'def get_.*_category\(',
]

# Classification column names in output
CLASSIFICATION_COLUMNS = [
    r"['\"]trajectory_type['\"]",
    r"['\"]stability_class['\"]",
    r"['\"]regime_type['\"]",
    r"['\"]health_status['\"]",
    r"['\"]anomaly_type['\"]",
    r"['\"]fault_type['\"]",
    r"['\"]is_chaotic['\"]",
    r"['\"]is_stable['\"]",
    r"['\"]is_healthy['\"]",
    r"['\"]is_anomaly['\"]",
    r"['\"]is_converging['\"]",
    r"['\"]is_diverging['\"]",
    r"['\"]is_collapsing['\"]",
    r"['\"]is_aligning['\"]",
    r"['\"]collapse_detected['\"]",
]

# Enums that indicate classification
CLASSIFICATION_ENUMS = [
    r'class TrajectoryType',
    r'class StabilityClass',
    r'class HealthStatus',
    r'class AnomalyType',
    r'class RegimeType',
]

# Classification labels being returned/assigned (not in typology)
CLASSIFICATION_LABELS_STRICT = [
    r"return.*['\"]chaotic['\"]",
    r"return.*['\"]stable['\"]",
    r"return.*['\"]unstable['\"]",
    r"return.*['\"]converging['\"]",
    r"return.*['\"]diverging['\"]",
    r"return.*['\"]oscillating['\"]",
    r"return.*['\"]collapsing['\"]",
    r"return.*['\"]expanding['\"]",
    r"return.*['\"]healthy['\"]",
    r"return.*['\"]degraded['\"]",
    r"return.*['\"]critical['\"]",
    r"return.*['\"]anomaly['\"]",
    r"return.*['\"]fault['\"]",
    r"return.*['\"]transient['\"]",
    r"return.*['\"]marginally_stable['\"]",
]

# Files to skip entirely (typology is allowed to classify signals)
SKIP_FILES = [
    'typology_engine.py',  # Typology classifies signals - this is OK
    'engine_manifest.yaml',  # Configuration
    'audit_classification.py',  # This file
]


# ============================================================
# AUDIT LOGIC
# ============================================================

def should_skip_file(filepath: Path) -> bool:
    """Check if file should be skipped entirely."""
    return filepath.name in SKIP_FILES


def should_skip_line(line: str) -> bool:
    """Check if line should be skipped (comments, docstrings, etc.)."""
    stripped = line.strip()
    if stripped.startswith('#'):
        return True
    if stripped.startswith('"""') or stripped.startswith("'''"):
        return True
    if 'logging.' in line or 'print(' in line:
        return True
    if 'raise ' in line or 'assert ' in line:
        return True
    return False


def audit_file(filepath: Path) -> List[Violation]:
    """Audit a single file for classification violations."""
    violations = []

    if should_skip_file(filepath):
        return []

    try:
        content = filepath.read_text()
        lines = content.split('\n')
    except Exception as e:
        return [Violation(str(filepath), 0, '', f'Could not read: {e}', 'error')]

    for i, line in enumerate(lines, 1):
        if should_skip_line(line):
            continue

        # Check classification functions
        for pattern in CLASSIFICATION_FUNCTIONS:
            if re.search(pattern, line):
                violations.append(Violation(
                    str(filepath), i, line.strip(),
                    f'Classification function: {pattern}',
                    'error'
                ))

        # Check classification enums
        for pattern in CLASSIFICATION_ENUMS:
            if re.search(pattern, line):
                violations.append(Violation(
                    str(filepath), i, line.strip(),
                    f'Classification enum: {pattern}',
                    'error'
                ))

        # Check classification columns
        for pattern in CLASSIFICATION_COLUMNS:
            if re.search(pattern, line):
                violations.append(Violation(
                    str(filepath), i, line.strip(),
                    f'Classification column: {pattern}',
                    'error'
                ))

        # Check classification labels being returned
        for pattern in CLASSIFICATION_LABELS_STRICT:
            if re.search(pattern, line, re.IGNORECASE):
                violations.append(Violation(
                    str(filepath), i, line.strip(),
                    f'Classification label return: {pattern}',
                    'error'
                ))

    return violations


def audit_directory(directory: Path) -> List[Violation]:
    """Audit all Python files in directory."""
    all_violations = []

    # Skip these directories
    skip_dirs = {'venv', '.venv', '__pycache__', '.git', 'node_modules', '_legacy'}

    for filepath in directory.rglob('*.py'):
        # Skip excluded directories
        if any(skip in filepath.parts for skip in skip_dirs):
            continue

        # Skip test files
        if 'test' in filepath.name.lower():
            continue

        violations = audit_file(filepath)
        all_violations.extend(violations)

    return all_violations


def print_report(violations: List[Violation]) -> None:
    """Print audit report."""
    print("=" * 70)
    print("PRISM CLASSIFICATION AUDIT")
    print("=" * 70)

    if not violations:
        print("\n[OK] No classification violations found!")
        return

    # Group by file
    by_file = {}
    for v in violations:
        if v.file not in by_file:
            by_file[v.file] = []
        by_file[v.file].append(v)

    errors = [v for v in violations if v.severity == 'error']
    warnings = [v for v in violations if v.severity == 'warning']

    print(f"\nFound {len(violations)} violations:")
    print(f"  Errors:   {len(errors)}")
    print(f"  Warnings: {len(warnings)}")
    print()

    for filepath, file_violations in sorted(by_file.items()):
        print(f"\n{filepath}")
        print("-" * min(len(filepath), 70))
        for v in file_violations:
            icon = "X" if v.severity == 'error' else "!"
            print(f"  {icon} Line {v.line_num}: {v.violation_type}")
            print(f"    {v.line[:70]}{'...' if len(v.line) > 70 else ''}")


def main():
    """Run audit."""
    # Find prism directory
    prism_dir = Path(__file__).parent.parent
    if not (prism_dir / '__init__.py').exists():
        prism_dir = Path.cwd() / 'prism'

    if not prism_dir.exists():
        print(f"Error: Could not find prism directory at {prism_dir}")
        sys.exit(1)

    print(f"Auditing: {prism_dir}")

    violations = audit_directory(prism_dir)
    print_report(violations)

    # Exit with error code if violations found
    errors = [v for v in violations if v.severity == 'error']
    if errors:
        print(f"\n[FAIL] {len(errors)} errors found. PRISM should not classify.")
        sys.exit(1)
    elif violations:
        print(f"\n[WARN] {len(violations)} warnings found. Review manually.")
        sys.exit(0)
    else:
        print("\n[OK] Clean!")
        sys.exit(0)


if __name__ == "__main__":
    main()
