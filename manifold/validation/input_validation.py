"""
Input Data Validation

Validates observations.parquet and typology.parquet before compute stages.
Filters out CONSTANT signals that carry no information.

PRINCIPLE: "CONSTANT signals have zero variance = zero information"

Usage:
    from manifold.validation import validate_input, filter_constant_signals

    # Full validation
    report = validate_input(data_dir='/path/to/data')

    # Filter CONSTANT signals from manifest
    filtered_manifest = filter_constant_signals(manifest, typology_df)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import polars as pl
import yaml


class ValidationError(Exception):
    """Raised when input validation fails."""

    def __init__(self, errors: List[str], warnings: List[str] = None):
        self.errors = errors
        self.warnings = warnings or []

        message = "Input validation failed:\n" + "\n".join(
            f"  ERROR: {e}" for e in errors
        )
        if warnings:
            message += "\n" + "\n".join(f"  WARNING: {w}" for w in warnings)

        super().__init__(message)


@dataclass
class InputValidationReport:
    """Report from input validation."""

    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Counts
    total_signals: int = 0
    constant_signals: int = 0
    active_signals: int = 0
    total_observations: int = 0

    # Signal lists
    constant_signal_ids: List[str] = field(default_factory=list)
    active_signal_ids: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            "INPUT VALIDATION REPORT",
            "=" * 60,
            "",
            f"Total signals: {self.total_signals}",
            f"  Active: {self.active_signals}",
            f"  CONSTANT (filtered): {self.constant_signals}",
            f"Total observations: {self.total_observations:,}",
            "",
        ]

        if self.constant_signal_ids:
            lines.append(f"CONSTANT signals filtered ({len(self.constant_signal_ids)}):")
            for sig in self.constant_signal_ids[:10]:
                lines.append(f"  - {sig}")
            if len(self.constant_signal_ids) > 10:
                lines.append(f"  ... and {len(self.constant_signal_ids) - 10} more")
            lines.append("")

        if self.errors:
            lines.append("ERRORS:")
            for e in self.errors:
                lines.append(f"  - {e}")
            lines.append("")

        if self.warnings:
            lines.append("WARNINGS:")
            for w in self.warnings:
                lines.append(f"  - {w}")
            lines.append("")

        status = "PASSED" if self.valid else "FAILED"
        lines.append(f"Status: {status}")
        lines.append("=" * 60)

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            'valid': self.valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'total_signals': self.total_signals,
            'constant_signals': self.constant_signals,
            'active_signals': self.active_signals,
            'total_observations': self.total_observations,
            'constant_signal_ids': self.constant_signal_ids,
            'active_signal_ids': self.active_signal_ids,
        }


def get_constant_signals(typology_df: pl.DataFrame) -> Set[str]:
    """
    Get set of CONSTANT signal IDs from typology.

    CONSTANT signals are identified by temporal_primary='CONSTANT' (new dual-classification
    schema) or temporal_pattern='CONSTANT' (legacy) or continuity='CONSTANT' in the typology.

    Args:
        typology_df: Typology DataFrame with signal_id and classification columns

    Returns:
        Set of signal_id values that are CONSTANT
    """
    constant_ids: Set[str] = set()

    # Check temporal_primary (new dual-classification schema)
    if 'temporal_primary' in typology_df.columns:
        constant_temporal = (
            typology_df
            .filter(pl.col('temporal_primary') == 'CONSTANT')
            .select('signal_id')
            .to_series()
            .to_list()
        )
        constant_ids.update(constant_temporal)
    # Fallback: old schema where temporal_pattern is a plain string
    elif 'temporal_pattern' in typology_df.columns:
        constant_temporal = (
            typology_df
            .filter(pl.col('temporal_pattern') == 'CONSTANT')
            .select('signal_id')
            .to_series()
            .to_list()
        )
        constant_ids.update(constant_temporal)

    # Check continuity column
    if 'continuity' in typology_df.columns:
        constant_continuity = (
            typology_df
            .filter(pl.col('continuity') == 'CONSTANT')
            .select('signal_id')
            .to_series()
            .to_list()
        )
        constant_ids.update(constant_continuity)

    return constant_ids


def filter_constant_signals(
    manifest: Dict[str, Any],
    typology_df: pl.DataFrame,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Remove CONSTANT signals from manifest.

    CONSTANT signals have zero variance and carry no information for analysis.
    They should be excluded from all compute stages.

    Args:
        manifest: Manifest dict with cohorts section
        typology_df: Typology DataFrame
        verbose: Print filtered signals

    Returns:
        New manifest with CONSTANT signals removed from cohorts
    """
    constant_ids = get_constant_signals(typology_df)

    if not constant_ids:
        return manifest

    # Deep copy manifest
    import copy
    filtered = copy.deepcopy(manifest)

    removed_count = 0

    # Filter each cohort
    for cohort_name, cohort_signals in list(filtered.get('cohorts', {}).items()):
        if not isinstance(cohort_signals, dict):
            continue

        for signal_id in list(cohort_signals.keys()):
            if signal_id in constant_ids:
                del cohort_signals[signal_id]
                removed_count += 1
                if verbose:
                    print(f"  Filtered CONSTANT: {signal_id} from {cohort_name}")

        # Remove empty cohorts
        if not cohort_signals:
            del filtered['cohorts'][cohort_name]
            if verbose:
                print(f"  Removed empty cohort: {cohort_name}")

    if verbose and removed_count > 0:
        print(f"Filtered {removed_count} CONSTANT signal(s) from manifest")

    return filtered


def validate_observations(
    observations_path: Path,
    report: InputValidationReport,
) -> pl.DataFrame:
    """
    Validate observations.parquet schema and basic data quality.

    Args:
        observations_path: Path to observations.parquet
        report: Report to update with findings

    Returns:
        Loaded observations DataFrame
    """
    # Check file exists
    if not observations_path.exists():
        report.errors.append(f"observations.parquet not found: {observations_path}")
        report.valid = False
        return None

    # Load and validate schema
    try:
        df = pl.read_parquet(observations_path)
    except Exception as e:
        report.errors.append(f"Failed to read observations.parquet: {e}")
        report.valid = False
        return None

    # Required columns
    required_cols = {'signal_id', 'I', 'value'}
    actual_cols = set(df.columns)

    missing_cols = required_cols - actual_cols
    if missing_cols:
        report.errors.append(f"Missing required columns: {missing_cols}")
        report.valid = False

    # Update report
    report.total_observations = len(df)
    if 'signal_id' in df.columns:
        report.total_signals = df['signal_id'].n_unique()

    # Data quality checks
    if 'value' in df.columns:
        null_count = df['value'].null_count()
        if null_count > 0:
            pct = 100.0 * null_count / len(df)
            report.warnings.append(f"{null_count:,} null values ({pct:.1f}%)")

        inf_count = df.filter(~pl.col('value').is_finite()).height
        if inf_count > 0:
            pct = 100.0 * inf_count / len(df)
            report.warnings.append(f"{inf_count:,} infinite values ({pct:.1f}%)")

    return df


def validate_typology(
    typology_path: Path,
    report: InputValidationReport,
) -> pl.DataFrame:
    """
    Validate typology.parquet schema.

    Args:
        typology_path: Path to typology.parquet
        report: Report to update with findings

    Returns:
        Loaded typology DataFrame
    """
    if not typology_path.exists():
        report.errors.append(f"typology.parquet not found: {typology_path}")
        report.valid = False
        return None

    try:
        df = pl.read_parquet(typology_path)
    except Exception as e:
        report.errors.append(f"Failed to read typology.parquet: {e}")
        report.valid = False
        return None

    # Required columns
    if 'signal_id' not in df.columns:
        report.errors.append("typology.parquet missing 'signal_id' column")
        report.valid = False

    return df


def validate_manifest(
    manifest_path: Path,
    report: InputValidationReport,
) -> Dict[str, Any]:
    """
    Validate manifest.yaml structure.

    Args:
        manifest_path: Path to manifest.yaml
        report: Report to update with findings

    Returns:
        Loaded manifest dict
    """
    if not manifest_path.exists():
        report.errors.append(f"manifest.yaml not found: {manifest_path}")
        report.valid = False
        return None

    try:
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
    except Exception as e:
        report.errors.append(f"Failed to read manifest.yaml: {e}")
        report.valid = False
        return None

    # Validate structure
    if 'system' not in manifest:
        report.errors.append("manifest.yaml missing 'system' section")
        report.valid = False

    if 'cohorts' not in manifest:
        report.errors.append("manifest.yaml missing 'cohorts' section")
        report.valid = False

    system = manifest.get('system', {})
    if 'window' not in system:
        report.errors.append("manifest.yaml system section missing 'window'")
        report.valid = False

    if 'stride' not in system:
        report.errors.append("manifest.yaml system section missing 'stride'")
        report.valid = False

    return manifest


def validate_input(
    data_dir: str,
    raise_on_error: bool = True,
    verbose: bool = False,
) -> InputValidationReport:
    """
    Validate all input files for signal_vector stage.

    Checks:
        1. observations.parquet exists and has correct schema
        2. typology.parquet exists and has signal classifications
        3. manifest.yaml exists and has required structure
        4. Identifies CONSTANT signals that should be filtered

    Args:
        data_dir: Directory containing pipeline files
        raise_on_error: If True, raise ValidationError on failure
        verbose: If True, print validation progress

    Returns:
        InputValidationReport with validation results

    Raises:
        ValidationError: If validation fails and raise_on_error=True
    """
    data_path = Path(data_dir)
    report = InputValidationReport()

    if verbose:
        print(f"Validating inputs in: {data_path}")

    # Validate each file
    obs_df = validate_observations(data_path / 'observations.parquet', report)
    typology_df = validate_typology(data_path / 'typology.parquet', report)
    manifest = validate_manifest(data_path / 'manifest.yaml', report)

    # If typology loaded, check for CONSTANT signals
    if typology_df is not None:
        constant_ids = get_constant_signals(typology_df)
        report.constant_signal_ids = sorted(constant_ids)
        report.constant_signals = len(constant_ids)

        # Get active signals
        if 'signal_id' in typology_df.columns:
            all_ids = set(typology_df['signal_id'].to_list())
            active_ids = all_ids - constant_ids
            report.active_signal_ids = sorted(active_ids)
            report.active_signals = len(active_ids)

    if verbose:
        print(report.summary())

    if not report.valid and raise_on_error:
        raise ValidationError(report.errors, report.warnings)

    return report


def validate_and_filter(
    data_dir: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Validate inputs and return filtered manifest (CONSTANT signals removed).

    Convenience function that combines validation and filtering.

    Args:
        data_dir: Directory containing pipeline files
        verbose: Print progress

    Returns:
        Dict with:
            - manifest: Filtered manifest (CONSTANT signals removed)
            - typology: Typology DataFrame
            - observations_path: Path to observations.parquet
            - report: InputValidationReport

    Raises:
        ValidationError: If validation fails
    """
    report = validate_input(data_dir, raise_on_error=True, verbose=verbose)

    data_path = Path(data_dir)
    typology_df = pl.read_parquet(data_path / 'typology.parquet')

    with open(data_path / 'manifest.yaml') as f:
        manifest = yaml.safe_load(f)

    # Filter CONSTANT signals
    filtered_manifest = filter_constant_signals(manifest, typology_df, verbose=verbose)

    return {
        'manifest': filtered_manifest,
        'typology': typology_df,
        'observations_path': str(data_path / 'observations.parquet'),
        'report': report,
    }
