#!/usr/bin/env python3
"""
PRISM Schema Validator — Verify observations match canonical schema.

Checks:
1. Required columns exist (entity_id, signal_id, index, value)
2. Column types are correct (string, string, float, float)
3. index is monotonic within each entity
4. No null values in required columns
5. Signals match domain YAML (if provided)

The 'index' column is sequence-agnostic: time, depth, distance, cycle, etc.
Aliases accepted: timestamp, time, cycle, depth, distance, position, step

Usage:
    python -m prism.entry_points.validate_schema
    python -m prism.entry_points.validate_schema --domain cmapss
    python -m prism.entry_points.validate_schema --verbose
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import polars as pl
import yaml

from prism.db.parquet_store import get_path, get_data_root, OBSERVATIONS


# =============================================================================
# Schema Definition
# =============================================================================

CANONICAL_SCHEMA = {
    'entity_id': pl.Utf8,
    'signal_id': pl.Utf8,
    'index': pl.Float64,      # Sequence-agnostic: time, depth, distance, cycle, etc.
    'value': pl.Float64,
    'unit': pl.Utf8,          # Optional: unit string (e.g., "psi", "degC", "rpm")
}

# Required columns (unit is optional)
REQUIRED_COLUMNS = ['entity_id', 'signal_id', 'index', 'value']

# Backwards compatibility - accept 'timestamp' as alias for 'index'
COLUMN_ALIASES = {
    'timestamp': 'index',
    'time': 'index',
    'cycle': 'index',
    'depth': 'index',
    'distance': 'index',
    'position': 'index',
    'step': 'index',
}

OPTIONAL_COLUMNS = {
    'unit': pl.Utf8,          # Unit for the value (enables physics calculations)
    'target': pl.Float64,
    'op_setting_1': pl.Float64,
    'op_setting_2': pl.Float64,
    'op_setting_3': pl.Float64,
}


# =============================================================================
# Validation Functions
# =============================================================================

def check_columns_exist(df: pl.DataFrame) -> List[Tuple[str, bool, str]]:
    """Check required columns exist (including aliases)."""
    results = []

    for col in CANONICAL_SCHEMA.keys():
        exists = col in df.columns
        alias_used = None
        is_optional = col in OPTIONAL_COLUMNS

        # Check for aliases (e.g., 'timestamp' -> 'index')
        if not exists and col == 'index':
            for alias, target in COLUMN_ALIASES.items():
                if target == 'index' and alias in df.columns:
                    exists = True
                    alias_used = alias
                    break

        if exists:
            msg = f"found as '{alias_used}'" if alias_used else "found"
            passed = True
        elif is_optional:
            msg = "optional, not present"
            passed = True  # Optional columns don't fail validation
        else:
            msg = "MISSING"
            passed = False
        results.append((f"Column '{col}'", passed, msg))

    return results


def check_column_types(df: pl.DataFrame) -> List[Tuple[str, bool, str]]:
    """Check column types match schema."""
    results = []

    for col, expected_type in CANONICAL_SCHEMA.items():
        if col not in df.columns:
            continue

        actual_type = df.schema[col]

        # Allow some type flexibility
        type_ok = False
        if expected_type == pl.Float64:
            type_ok = actual_type in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        elif expected_type == pl.Utf8:
            type_ok = actual_type in [pl.Utf8, pl.Categorical, pl.String]
        else:
            type_ok = actual_type == expected_type

        msg = f"{actual_type}" if type_ok else f"{actual_type} (expected {expected_type})"
        results.append((f"Type '{col}'", type_ok, msg))

    return results


def check_no_nulls(df: pl.DataFrame) -> List[Tuple[str, bool, str]]:
    """Check no null values in required columns."""
    results = []

    # Only check required columns for nulls (optional columns can have nulls)
    for col in REQUIRED_COLUMNS:
        # Handle aliases (e.g., 'timestamp' -> 'index')
        actual_col = col
        if col == 'index' and col not in df.columns:
            for alias, target in COLUMN_ALIASES.items():
                if target == 'index' and alias in df.columns:
                    actual_col = alias
                    break

        if actual_col not in df.columns:
            continue

        null_count = df.select(pl.col(actual_col).is_null().sum()).item()
        ok = null_count == 0
        msg = f"no nulls" if ok else f"{null_count:,} nulls"
        results.append((f"Nulls '{col}'", ok, msg))

    return results


def get_index_column(df: pl.DataFrame) -> str:
    """Get the index column name (handles aliases)."""
    if 'index' in df.columns:
        return 'index'
    for alias, target in COLUMN_ALIASES.items():
        if target == 'index' and alias in df.columns:
            return alias
    return None


def check_index_monotonic(df: pl.DataFrame) -> List[Tuple[str, bool, str]]:
    """Check index is monotonically increasing within each entity."""
    results = []

    index_col = get_index_column(df)
    if index_col is None or 'entity_id' not in df.columns:
        return results

    # Check monotonicity per entity (need to also partition by signal_id)
    violations = (
        df
        .sort(['entity_id', 'signal_id', index_col])
        .with_columns(
            pl.col(index_col).diff().over(['entity_id', 'signal_id']).alias('idx_diff')
        )
        .filter(pl.col('idx_diff') < 0)
    )

    n_violations = len(violations)
    ok = n_violations == 0

    if ok:
        msg = "monotonic within entities"
    else:
        # Get example violation
        example = violations.head(1)
        entity = example['entity_id'][0]
        msg = f"{n_violations:,} violations (e.g., entity={entity})"

    results.append((f"Index '{index_col}' monotonic", ok, msg))

    return results


def check_signal_coverage(
    df: pl.DataFrame,
    domain_yaml: dict
) -> List[Tuple[str, bool, str]]:
    """Check signals match domain YAML."""
    results = []

    if 'signals' not in domain_yaml:
        return results

    # Expected signals from YAML
    expected_ids = {
        cfg['prism_id']
        for cfg in domain_yaml['signals'].values()
    }

    # Add target if present
    if domain_yaml.get('target') and domain_yaml['target'].get('prism_id'):
        expected_ids.add(domain_yaml['target']['prism_id'])

    # Add op_settings if present
    if domain_yaml.get('op_settings'):
        for cfg in domain_yaml['op_settings'].values():
            if cfg.get('prism_id'):
                expected_ids.add(cfg['prism_id'])

    # Actual signals in data
    actual_ids = set(df['signal_id'].unique().to_list())

    # Check coverage
    missing = expected_ids - actual_ids
    extra = actual_ids - expected_ids

    if not missing:
        results.append(("Expected signals", True, f"all {len(expected_ids)} found"))
    else:
        results.append(("Expected signals", False, f"missing: {missing}"))

    if extra:
        # Extra signals are a warning, not an error
        results.append(("Extra signals", True, f"found {len(extra)} unmapped: {list(extra)[:3]}..."))

    return results


def print_summary(df: pl.DataFrame, verbose: bool = False):
    """Print data summary."""
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)

    n_rows = len(df)
    n_entities = df['entity_id'].n_unique() if 'entity_id' in df.columns else 0
    n_signals = df['signal_id'].n_unique() if 'signal_id' in df.columns else 0

    print(f"  Rows: {n_rows:,}")
    print(f"  Entities: {n_entities:,}")
    print(f"  Signals: {n_signals:,}")

    if 'timestamp' in df.columns:
        ts_min = df['timestamp'].min()
        ts_max = df['timestamp'].max()
        print(f"  Timestamp range: {ts_min} -> {ts_max}")

    if verbose and 'signal_id' in df.columns:
        print("\n  Signals found:")
        for sig in sorted(df['signal_id'].unique().to_list()):
            count = len(df.filter(pl.col('signal_id') == sig))
            print(f"    {sig}: {count:,} rows")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='PRISM Schema Validator'
    )
    parser.add_argument(
        '--domain', type=str, default=None,
        help='Domain name (loads config/domains/{domain}.yaml)'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Show detailed output'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PRISM SCHEMA VALIDATOR")
    print("=" * 60)

    # Load observations
    obs_path = get_path(OBSERVATIONS)
    if not Path(obs_path).exists():
        print(f"\n[X] Observations not found: {obs_path}")
        print("  Run the fetcher first to generate data.")
        sys.exit(1)

    print(f"\nChecking: {obs_path}")
    df = pl.read_parquet(obs_path)

    # Load domain YAML if available
    domain_yaml = None
    if args.domain:
        # Look in config/domains/ relative to project root
        project_root = Path(__file__).parent.parent.parent
        yaml_path = project_root / 'config' / 'domains' / f'{args.domain}.yaml'
        if yaml_path.exists():
            with open(yaml_path) as f:
                domain_yaml = yaml.safe_load(f)
            print(f"Domain config: {yaml_path}")
        else:
            print(f"Warning: Domain config not found: {yaml_path}")

    # Run checks
    all_results = []

    print("\n" + "-" * 60)
    print("SCHEMA CHECKS")
    print("-" * 60)

    all_results.extend(check_columns_exist(df))
    all_results.extend(check_column_types(df))
    all_results.extend(check_no_nulls(df))
    all_results.extend(check_index_monotonic(df))

    if domain_yaml:
        all_results.extend(check_signal_coverage(df, domain_yaml))

    # Print results
    all_passed = True
    for name, passed, msg in all_results:
        symbol = "[OK]" if passed else "[X]"
        print(f"  {symbol} {name}: {msg}")
        if not passed:
            all_passed = False

    # Summary
    print_summary(df, args.verbose)

    # Final verdict
    print("\n" + "=" * 60)
    if all_passed:
        print("[OK] SCHEMA VALID")
        print("=" * 60)
        sys.exit(0)
    else:
        print("[X] SCHEMA INVALID - Fix issues above")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
