"""
PRISM Data Check

Validates observations.parquet BEFORE any compute.
Reads validation rules from canonical PRISM_SCHEMA.yaml.

This is PRISM's gate - if it fails, PRISM aborts.

Schema v2.0.0:
- REQUIRED: signal_id, I, value
- OPTIONAL: unit_id (just a label, blank is fine)

Usage:
    # In runner.py (automatic)
    from prism.data_check import check_data, abort_if_invalid
    abort_if_invalid(observations_path)

    # Standalone
    python -m prism.data_check observations.parquet
"""

import polars as pl
from pathlib import Path
import sys
import yaml
from dataclasses import dataclass, field
from typing import Optional, Tuple


# =============================================================================
# DEFAULT SCHEMA LOCATIONS
# =============================================================================

SCHEMA_LOCATIONS = [
    Path(__file__).parent.parent / "shared" / "schema" / "PRISM_SCHEMA.yaml",
    Path(__file__).parent / "PRISM_SCHEMA.yaml",
    Path.home() / "prism" / "PRISM_SCHEMA.yaml",
    Path("/etc/prism/PRISM_SCHEMA.yaml"),
]


def find_schema() -> Optional[Path]:
    """Find the canonical schema file."""
    for loc in SCHEMA_LOCATIONS:
        if loc.exists():
            return loc
    return None


def load_schema(path: Optional[Path] = None) -> dict:
    """Load the canonical schema."""
    if path is None:
        path = find_schema()

    if path is None or not path.exists():
        return get_default_schema()

    with open(path) as f:
        return yaml.safe_load(f)


def get_default_schema() -> dict:
    """Embedded schema defaults if file not found."""
    return {
        "columns": {
            "unit_id": {"type": "string", "nullable": True, "required": False},
            "signal_id": {"type": "string", "nullable": False, "required": True},
            "I": {"type": "uint32", "nullable": False, "required": True},
            "value": {"type": "float64", "nullable": True, "required": True},
        },
        "required_columns": ["signal_id", "I", "value"],
        "optional_columns": ["unit_id"],
        "requirements": {
            "min_signals": 2,
            "min_observations": 50,
        },
    }


# =============================================================================
# RESULT CLASS
# =============================================================================

@dataclass
class CheckResult:
    """Result of data validation."""
    passed: bool = True
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    df: pl.DataFrame = None  # Store potentially modified df

    def error(self, rule_id: str, message: str):
        self.passed = False
        self.errors.append({"rule": rule_id, "message": message})

    def warn(self, rule_id: str, message: str):
        self.warnings.append({"rule": rule_id, "message": message})

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
        }


# =============================================================================
# TYPE MAPPING
# =============================================================================

TYPE_MAP = {
    "string": [pl.String, pl.Utf8],
    "uint32": [pl.UInt32, pl.UInt64, pl.Int32, pl.Int64],
    "float64": [pl.Float64, pl.Float32],
    "integer": [pl.UInt32, pl.UInt64, pl.Int32, pl.Int64],
    "float": [pl.Float64, pl.Float32],
}


# =============================================================================
# MAIN CHECK FUNCTION
# =============================================================================

def check_data(
    path: Path,
    schema_path: Optional[Path] = None,
    verbose: bool = True
) -> CheckResult:
    """
    Validate observations.parquet against canonical schema.

    Args:
        path: Path to observations.parquet
        schema_path: Optional path to PRISM_SCHEMA.yaml
        verbose: Print detailed output

    Returns:
        CheckResult with passed/failed status and potentially modified df
    """

    result = CheckResult()
    schema = load_schema(schema_path)

    # =========================================================================
    # HEADER
    # =========================================================================

    if verbose:
        print(f"\n{'='*60}")
        print(f"PRISM DATA CHECK (Schema v2.0)")
        print(f"{'='*60}")
        print(f"File: {path}")
        print()

    # =========================================================================
    # FILE EXISTS
    # =========================================================================

    if not path.exists():
        result.error("FILE_EXISTS", f"File not found: {path}")
        if verbose:
            print(f"  File not found: {path}")
        return result

    # =========================================================================
    # LOAD DATA
    # =========================================================================

    try:
        df = pl.read_parquet(path)
    except Exception as e:
        result.error("FILE_READABLE", f"Failed to read parquet: {e}")
        if verbose:
            print(f"  Failed to read: {e}")
        return result

    result.stats["rows"] = df.shape[0]
    result.stats["columns"] = list(df.columns)

    if verbose:
        print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
        print(f"Columns: {df.columns}\n")

    # =========================================================================
    # CHECK: REQUIRED COLUMNS (signal_id, I, value)
    # =========================================================================

    if verbose:
        print("[1] REQUIRED COLUMNS")

    required_cols = ["signal_id", "I", "value"]

    for col in required_cols:
        if col not in df.columns:
            result.error("REQUIRED_COLUMNS", f"Missing REQUIRED column: {col}")
            if verbose:
                print(f"    X {col}: MISSING")
        else:
            if verbose:
                print(f"    + {col}: present")

    if not result.passed:
        if verbose:
            print("\n[!] Cannot continue - missing required columns")
        return result

    # =========================================================================
    # CHECK: OPTIONAL unit_id COLUMN
    # =========================================================================

    if verbose:
        print(f"\n[2] OPTIONAL COLUMNS")

    if "unit_id" not in df.columns:
        if verbose:
            print(f"    ~ unit_id: not present (adding blank)")
        # Add blank unit_id - this is fine
        df = df.with_columns(pl.lit("").alias("unit_id"))
    else:
        n_units = df["unit_id"].n_unique()
        if verbose:
            print(f"    + unit_id: present ({n_units} unique values)")

    # Store the potentially modified df
    result.df = df

    # =========================================================================
    # CHECK: COLUMN TYPES
    # =========================================================================

    if verbose:
        print(f"\n[3] COLUMN TYPES")

    type_checks = {
        "signal_id": ["string"],
        "I": ["uint32", "integer"],
        "value": ["float64", "float"],
    }

    for col, expected_types in type_checks.items():
        actual_type = df[col].dtype
        valid = False
        for et in expected_types:
            if actual_type in TYPE_MAP.get(et, []):
                valid = True
                break

        if not valid:
            result.error("COLUMN_TYPES", f"{col}: expected {expected_types[0]}, got {actual_type}")
            if verbose:
                print(f"    X {col}: {actual_type} (expected {expected_types[0]})")
        else:
            if verbose:
                print(f"    + {col}: {actual_type}")

    # =========================================================================
    # CHECK: NULL VALUES IN REQUIRED COLUMNS
    # =========================================================================

    if verbose:
        print(f"\n[4] NULL VALUES")

    # signal_id and I must not have nulls
    for col in ["signal_id", "I"]:
        null_count = df[col].null_count()
        if null_count > 0:
            result.error("NO_NULLS_IN_KEYS", f"{col} has {null_count} null values")
            if verbose:
                print(f"    X {col}: {null_count} nulls (not allowed)")
        else:
            if verbose:
                print(f"    + {col}: no nulls")

    # value can have nulls (NaN for missing data)
    value_nulls = df["value"].null_count()
    if verbose:
        status = "+" if value_nulls == 0 else "~"
        print(f"    {status} value: {value_nulls} nulls (allowed)")

    # =========================================================================
    # CHECK: I IS SEQUENTIAL
    # =========================================================================

    if verbose:
        print(f"\n[5] INDEX SEQUENTIALITY")

    # Check if I is sequential per unit/signal
    seq_check = (
        df.group_by(["unit_id", "signal_id"])
        .agg([
            pl.col("I").min().alias("min_i"),
            pl.col("I").max().alias("max_i"),
            pl.len().alias("count")
        ])
        .with_columns(
            ((pl.col("max_i") - pl.col("min_i") + 1) == pl.col("count")).alias("is_sequential")
        )
    )

    non_sequential = seq_check.filter(~pl.col("is_sequential"))

    if len(non_sequential) > 0:
        result.error("I_SEQUENTIAL", f"{len(non_sequential)} unit/signal pairs have non-sequential I")
        if verbose:
            print(f"    X {len(non_sequential)} groups have non-sequential I")
            sample = non_sequential.head(3)
            for row in sample.iter_rows(named=True):
                unit = row['unit_id'] or '(blank)'
                print(f"       - {unit}/{row['signal_id']}: I={row['min_i']}->{row['max_i']} but {row['count']} values")
    else:
        if verbose:
            print(f"    + I is sequential for all unit/signal pairs")

    # Check I starts at 0 per unit
    min_i_check = df.group_by("unit_id").agg(pl.col("I").min().alias("min_i"))
    non_zero = min_i_check.filter(pl.col("min_i") != 0)

    if len(non_zero) > 0:
        result.error("I_STARTS_ZERO", f"{len(non_zero)} units don't start at I=0")
        if verbose:
            print(f"    X {len(non_zero)} units don't start at I=0")
    else:
        if verbose:
            print(f"    + I starts at 0 for all units")

    # =========================================================================
    # CHECK: MINIMUM SIGNALS
    # =========================================================================

    if verbose:
        print(f"\n[6] SIGNAL COUNT")

    n_signals = df["signal_id"].n_unique()
    result.stats["n_signals"] = n_signals
    result.stats["signals"] = df["signal_id"].unique().sort().to_list()

    if n_signals < 2:
        result.error("MIN_SIGNALS", f"Need >= 2 signals, found {n_signals}")
        if verbose:
            print(f"    X Only {n_signals} signal(s) - pair engines need >= 2")
    else:
        if verbose:
            print(f"    + {n_signals} signals: {result.stats['signals']}")

    # =========================================================================
    # CHECK: MINIMUM OBSERVATIONS
    # =========================================================================

    if verbose:
        print(f"\n[7] OBSERVATIONS PER UNIT/SIGNAL")

    obs_per_group = (
        df.group_by(["unit_id", "signal_id"])
        .agg(pl.len().alias("n_obs"))
    )

    insufficient_obs = obs_per_group.filter(pl.col("n_obs") < 50)

    result.stats["min_obs"] = obs_per_group["n_obs"].min()
    result.stats["max_obs"] = obs_per_group["n_obs"].max()
    result.stats["mean_obs"] = obs_per_group["n_obs"].mean()
    result.stats["n_units"] = df["unit_id"].n_unique()

    if len(insufficient_obs) > 0:
        result.warn("MIN_OBSERVATIONS", f"{len(insufficient_obs)} groups have < 50 observations")
        if verbose:
            print(f"    ~ {len(insufficient_obs)} groups have < 50 observations")
            print(f"       Some engines may return NaN")
    else:
        if verbose:
            print(f"    + All groups have >= 50 observations")

    if verbose:
        print(f"    Range: {result.stats['min_obs']} - {result.stats['max_obs']} (mean: {result.stats['mean_obs']:.0f})")
        print(f"    Units: {result.stats['n_units']}")

    # =========================================================================
    # SUMMARY
    # =========================================================================

    if verbose:
        print(f"\n{'='*60}")
        if result.passed:
            print("+ DATA CHECK PASSED")
            print("   PRISM will proceed with compute")
        else:
            print("X DATA CHECK FAILED")
            print(f"   {len(result.errors)} error(s):")
            for err in result.errors:
                print(f"   - [{err['rule']}] {err['message']}")
            print("\n[!] PRISM will NOT run on this data")
            print("   Fix the errors and try again")

        if result.warnings:
            print(f"\n   {len(result.warnings)} warning(s):")
            for warn in result.warnings:
                print(f"   - [{warn['rule']}] {warn['message']}")

        print(f"{'='*60}\n")

    return result


def abort_if_invalid(path: Path, schema_path: Optional[Path] = None) -> Tuple[CheckResult, pl.DataFrame]:
    """
    Check data and abort if invalid.
    Use at the start of PRISM runner.

    Returns:
        Tuple of (CheckResult, DataFrame) - df may have unit_id added if missing
    """
    result = check_data(path, schema_path, verbose=True)

    if not result.passed:
        print("\n[!] ABORTING: Data validation failed")
        print("   Run ORTHON data_confirmation.py to fix the data")
        sys.exit(1)

    return result, result.df


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PRISM Data Check")
    parser.add_argument("path", type=Path, help="Path to observations.parquet")
    parser.add_argument("--schema", type=Path, default=None, help="Path to PRISM_SCHEMA.yaml")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--json", action="store_true", help="Output JSON result")

    args = parser.parse_args()

    result = check_data(args.path, args.schema, verbose=not args.quiet and not args.json)

    if args.json:
        import json
        print(json.dumps(result.to_dict(), indent=2))

    sys.exit(0 if result.passed else 1)
