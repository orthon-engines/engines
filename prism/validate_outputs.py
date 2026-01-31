"""
PRISM Output Validation

Validates output parquet files for correctness and completeness.
"""

import polars as pl
from pathlib import Path
from typing import List, Tuple


def validate_directory(output_dir: str) -> bool:
    """Validate all output files in a directory."""
    output_dir = Path(output_dir)

    if not output_dir.exists():
        print(f"ERROR: Directory does not exist: {output_dir}")
        return False

    print("=" * 60)
    print("PRISM OUTPUT VALIDATION")
    print("=" * 60)
    print(f"Directory: {output_dir}")

    checks_passed = 0
    checks_failed = 0

    # Find all parquet files
    parquet_files = list(output_dir.glob('*.parquet'))

    if not parquet_files:
        print("\nWARNING: No parquet files found")
        return False

    print(f"\nFound {len(parquet_files)} parquet files")

    # Validate each file
    for pq_file in sorted(parquet_files):
        file_checks = validate_parquet(pq_file)
        for check_name, passed, message in file_checks:
            if passed:
                print(f"  ✓ {check_name}")
                checks_passed += 1
            else:
                print(f"  ✗ {check_name}: {message}")
                checks_failed += 1

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Checks passed: {checks_passed}")
    print(f"Checks failed: {checks_failed}")

    if checks_failed == 0:
        print("\n✓ ALL VALIDATION CHECKS PASSED")
        return True
    else:
        print(f"\n✗ {checks_failed} VALIDATION CHECKS FAILED")
        return False


def validate_parquet(file_path: Path) -> List[Tuple[str, bool, str]]:
    """Validate a single parquet file."""
    results = []
    file_name = file_path.name

    print(f"\n[{file_name}]")

    try:
        df = pl.read_parquet(file_path)
    except Exception as e:
        results.append((f"{file_name} readable", False, str(e)))
        return results

    # Check 1: File is readable and has rows
    if len(df) > 0:
        results.append((f"{file_name} has data ({len(df):,} rows)", True, ""))
    else:
        results.append((f"{file_name} has data", False, "0 rows"))

    # Check 2: No completely null columns
    null_cols = []
    for col in df.columns:
        if df[col].null_count() == len(df):
            null_cols.append(col)

    if null_cols:
        results.append((f"{file_name} no all-null columns", False, f"All null: {null_cols}"))
    else:
        results.append((f"{file_name} no all-null columns", True, ""))

    # Check 3: No domain prefixes in column names
    domain_prefixes = ['bearing_', 'motor_', 'heat_', 'flow_', 'vib_', 'pressure_', 'rotor_']
    bad_cols = [col for col in df.columns if any(col.startswith(p) for p in domain_prefixes)]

    if bad_cols:
        results.append((f"{file_name} no domain prefixes", False, f"Found: {bad_cols}"))
    else:
        results.append((f"{file_name} no domain prefixes", True, ""))

    # File-specific checks
    if file_name == 'primitives.parquet':
        results.extend(validate_primitives(df, file_name))
    elif file_name == 'observations_enriched.parquet':
        results.extend(validate_observations_enriched(df, file_name))
    elif file_name == 'geometry.parquet':
        results.extend(validate_geometry(df, file_name))
    elif file_name == 'zscore.parquet':
        results.extend(validate_zscore(df, file_name))
    elif file_name == 'statistics.parquet':
        results.extend(validate_statistics(df, file_name))

    return results


def validate_primitives(df: pl.DataFrame, name: str) -> List[Tuple[str, bool, str]]:
    """Validate primitives.parquet structure."""
    results = []

    # Required columns
    required = ['unit_id', 'signal_id']
    missing = [c for c in required if c not in df.columns]

    if missing:
        results.append((f"{name} has required columns", False, f"Missing: {missing}"))
    else:
        results.append((f"{name} has required columns", True, ""))

    # Should have numeric metric columns
    numeric_cols = [c for c in df.columns if df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
    if len(numeric_cols) >= 3:
        results.append((f"{name} has numeric metrics ({len(numeric_cols)} cols)", True, ""))
    else:
        results.append((f"{name} has numeric metrics", False, f"Only {len(numeric_cols)} numeric columns"))

    return results


def validate_observations_enriched(df: pl.DataFrame, name: str) -> List[Tuple[str, bool, str]]:
    """Validate observations_enriched.parquet structure."""
    results = []

    # Required columns
    required = ['unit_id', 'signal_id', 'I', 'value']
    missing = [c for c in required if c not in df.columns]

    if missing:
        results.append((f"{name} has required columns", False, f"Missing: {missing}"))
    else:
        results.append((f"{name} has required columns", True, ""))

    # Should have enrichment columns beyond the base
    enrichment_cols = [c for c in df.columns if c not in ['unit_id', 'signal_id', 'I', 'value', 'unit', 'timestamp']]
    if len(enrichment_cols) >= 1:
        results.append((f"{name} has enrichment columns ({len(enrichment_cols)})", True, ""))
    else:
        results.append((f"{name} has enrichment columns", False, "No enrichment columns found"))

    return results


def validate_geometry(df: pl.DataFrame, name: str) -> List[Tuple[str, bool, str]]:
    """Validate geometry.parquet structure."""
    results = []

    # Required columns for pair data
    required = ['unit_id']
    missing = [c for c in required if c not in df.columns]

    if missing:
        results.append((f"{name} has required columns", False, f"Missing: {missing}"))
    else:
        results.append((f"{name} has required columns", True, ""))

    # Should have signal pair columns
    has_pair = ('signal_a' in df.columns and 'signal_b' in df.columns) or \
               ('source_signal' in df.columns and 'target_signal' in df.columns)

    if has_pair:
        results.append((f"{name} has signal pair columns", True, ""))
    else:
        results.append((f"{name} has signal pair columns", False, "Missing signal_a/signal_b or source_signal/target_signal"))

    return results


def validate_zscore(df: pl.DataFrame, name: str) -> List[Tuple[str, bool, str]]:
    """Validate zscore.parquet structure."""
    results = []

    # Required columns
    required = ['unit_id', 'signal_id', 'I', 'value', 'z_score']
    missing = [c for c in required if c not in df.columns]

    if missing:
        results.append((f"{name} has required columns", False, f"Missing: {missing}"))
    else:
        results.append((f"{name} has required columns", True, ""))

    # Z-scores should be reasonable (mostly within -10 to 10)
    if 'z_score' in df.columns:
        extreme = df.filter((pl.col('z_score').abs() > 10) & pl.col('z_score').is_not_null())
        extreme_pct = len(extreme) / len(df) * 100 if len(df) > 0 else 0

        if extreme_pct < 1:
            results.append((f"{name} z_scores reasonable (<1% extreme)", True, ""))
        else:
            results.append((f"{name} z_scores reasonable", False, f"{extreme_pct:.2f}% are extreme (|z| > 10)"))

    return results


def validate_statistics(df: pl.DataFrame, name: str) -> List[Tuple[str, bool, str]]:
    """Validate statistics.parquet structure."""
    results = []

    # Required columns
    required = ['unit_id', 'signal_id', 'mean', 'std']
    missing = [c for c in required if c not in df.columns]

    if missing:
        results.append((f"{name} has required columns", False, f"Missing: {missing}"))
    else:
        results.append((f"{name} has required columns", True, ""))

    # Std should be non-negative
    if 'std' in df.columns:
        negative_std = df.filter(pl.col('std') < 0)
        if len(negative_std) == 0:
            results.append((f"{name} std values non-negative", True, ""))
        else:
            results.append((f"{name} std values non-negative", False, f"{len(negative_std)} negative values"))

    return results


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m prism.validate_outputs <output_dir>")
        sys.exit(1)

    success = validate_directory(sys.argv[1])
    sys.exit(0 if success else 1)
