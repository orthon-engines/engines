"""
Input Loader — Auto-detect and convert input data to ENGINES observations format.

Supports:
  - CSV/TSV in long format (I, signal_id, value)
  - CSV/TSV in wide format (columns = signals, rows = timepoints)
  - Parquet files (observations.parquet format)
  - Directory containing observations.parquet

Column name aliases are auto-detected and mapped to canonical names.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Optional, Dict, List, Tuple


# Canonical column names → common aliases (case-insensitive matching)
COLUMN_ALIASES = {
    'I': ['i', 'index', 'time', 'timestamp', 't', 'step', 'sample', 'obs',
          'observation', 'row', 'timepoint', 'time_step', 'timestep'],
    'signal_id': ['signal_id', 'signal', 'sensor', 'channel', 'variable',
                  'feature', 'tag', 'name', 'metric', 'measure', 'sensor_id',
                  'var', 'column', 'col'],
    'value': ['value', 'val', 'measurement', 'reading', 'data', 'y',
              'response', 'output', 'observation'],
    'cohort': ['cohort', 'group', 'unit', 'run', 'trial', 'experiment',
               'fault', 'unit_id', 'entity', 'entity_id', 'batch',
               'session', 'subject'],
}


def load_input(path: str | Path) -> pl.DataFrame:
    """
    Auto-detect input format and load as observations DataFrame.

    Args:
        path: Path to CSV, TSV, parquet file, or directory

    Returns:
        DataFrame with columns: I (UInt32), signal_id (Utf8), value (Float64),
        and optionally cohort (Utf8)

    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If format cannot be detected or data is invalid
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    if path.is_dir():
        return _load_directory(path)
    elif path.suffix == '.parquet':
        return _load_parquet(path)
    elif path.suffix in ('.csv', '.tsv', '.txt'):
        return _load_csv(path)
    else:
        # Try CSV as fallback
        try:
            return _load_csv(path)
        except Exception:
            raise ValueError(f"Unsupported file type: {path.suffix}")


def _load_directory(path: Path) -> pl.DataFrame:
    """Load observations.parquet from a directory."""
    obs_path = path / 'observations.parquet'
    if obs_path.exists():
        return _load_parquet(obs_path)
    raise FileNotFoundError(
        f"No observations.parquet in {path}. "
        f"Files found: {[f.name for f in path.iterdir() if f.is_file()][:10]}"
    )


def _load_parquet(path: Path) -> pl.DataFrame:
    """Load and validate a parquet file."""
    df = pl.read_parquet(path)
    return _normalize_columns(df)


def _load_csv(path: Path) -> pl.DataFrame:
    """Load CSV/TSV and detect format (long vs wide)."""
    sep = '\t' if path.suffix == '.tsv' else ','

    # Try to infer separator for .csv files
    if path.suffix == '.csv':
        with open(path) as f:
            first_line = f.readline()
            if '\t' in first_line and ',' not in first_line:
                sep = '\t'

    df = pl.read_csv(path, separator=sep, infer_schema_length=5000)

    # Try long format first
    mapped = _detect_column_mapping(df)
    if mapped is not None:
        return _normalize_long(df, mapped)

    # Fall back to wide format
    return _wide_to_long(df)


def _detect_column_mapping(df: pl.DataFrame) -> Optional[Dict[str, str]]:
    """
    Detect if DataFrame is in long format by looking for canonical column aliases.

    Returns:
        Dict mapping actual column names → canonical names, or None if not long format
    """
    cols_lower = {c.lower().strip(): c for c in df.columns}
    mapping = {}

    # Must find at least signal_id and value to be long format
    found_signal = False
    found_value = False

    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in cols_lower:
                mapping[cols_lower[alias]] = canonical
                if canonical == 'signal_id':
                    found_signal = True
                elif canonical == 'value':
                    found_value = True
                break

    if found_signal and found_value:
        return mapping

    return None


def _normalize_long(df: pl.DataFrame, mapping: Dict[str, str]) -> pl.DataFrame:
    """Normalize a long-format DataFrame with detected column mapping."""
    df = df.rename(mapping)
    return _normalize_columns(df)


def _normalize_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure DataFrame has required columns with correct types."""
    # Required: signal_id, I, value
    if 'signal_id' not in df.columns:
        raise ValueError(f"Cannot find signal_id column. Columns: {df.columns}")
    if 'value' not in df.columns:
        raise ValueError(f"Cannot find value column. Columns: {df.columns}")

    # Ensure signal_id is string
    df = df.with_columns(pl.col('signal_id').cast(pl.Utf8))

    # Ensure value is float
    df = df.with_columns(pl.col('value').cast(pl.Float64))

    # Create or fix I column
    if 'I' not in df.columns:
        # Generate sequential I per signal (and cohort if present)
        group_cols = ['cohort', 'signal_id'] if 'cohort' in df.columns else ['signal_id']
        df = df.with_columns(
            pl.arange(0, pl.len()).over(group_cols).cast(pl.UInt32).alias('I')
        )
    else:
        # Check if I looks like timestamps (values too large for sequential index)
        i_max = df['I'].max()
        n_per_signal = df.group_by('signal_id').len()['len'].min()
        if i_max is not None and n_per_signal is not None and i_max > n_per_signal * 10:
            # Regenerate as sequential
            group_cols = ['cohort', 'signal_id'] if 'cohort' in df.columns else ['signal_id']
            df = df.sort(group_cols + ['I'])
            df = df.with_columns(
                pl.arange(0, pl.len()).over(group_cols).cast(pl.UInt32).alias('I')
            )
        else:
            df = df.with_columns(pl.col('I').cast(pl.UInt32))

    # Remove null signal_ids
    df = df.filter(pl.col('signal_id').is_not_null())

    # Remove null values
    df = df.filter(pl.col('value').is_not_null())

    # Select canonical columns
    cols = ['I', 'signal_id', 'value']
    if 'cohort' in df.columns:
        cols.append('cohort')
        df = df.with_columns(pl.col('cohort').cast(pl.Utf8))

    return df.select(cols).sort(['signal_id', 'I'])


def _wide_to_long(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert wide-format data (columns = signals) to long-format observations.

    Assumes:
      - First column = index (becomes I), OR a detected index column
      - Remaining numeric columns = signals
      - Non-numeric columns checked for cohort alias
    """
    cols_lower = {c.lower().strip(): c for c in df.columns}

    # Find index column
    index_col = None
    for alias in COLUMN_ALIASES['I']:
        if alias in cols_lower:
            index_col = cols_lower[alias]
            break
    if index_col is None:
        index_col = df.columns[0]

    # Find cohort column if present
    cohort_col = None
    for alias in COLUMN_ALIASES['cohort']:
        if alias in cols_lower and cols_lower[alias] != index_col:
            cohort_col = cols_lower[alias]
            break

    # Signal columns = all numeric columns except index and cohort
    exclude = {index_col}
    if cohort_col:
        exclude.add(cohort_col)

    signal_cols = [
        c for c in df.columns
        if c not in exclude
        and df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt64, pl.UInt32)
    ]

    if not signal_cols:
        raise ValueError(
            f"No numeric signal columns found. "
            f"Index column: '{index_col}', columns: {df.columns}"
        )

    # Melt to long format
    id_vars = [index_col]
    if cohort_col:
        id_vars.append(cohort_col)

    long = df.unpivot(
        on=signal_cols,
        index=id_vars,
        variable_name='signal_id',
        value_name='value',
    )

    # Rename index column to I
    if index_col != 'I':
        long = long.rename({index_col: 'I'})

    if cohort_col and cohort_col != 'cohort':
        long = long.rename({cohort_col: 'cohort'})
    elif cohort_col is None:
        long = long.with_columns(pl.lit('default').alias('cohort'))

    return _normalize_columns(long)


def detect_data_characteristics(df: pl.DataFrame) -> Dict:
    """
    Analyze input data and return characteristics for auto-manifest.

    Args:
        df: Observations DataFrame (long format)

    Returns:
        Dict with data characteristics
    """
    signal_list = df['signal_id'].unique().sort().to_list()
    n_signals = len(signal_list)

    samples_per_signal = df.group_by('signal_id').len()
    min_samples = int(samples_per_signal['len'].min())
    max_samples = int(samples_per_signal['len'].max())
    median_samples = int(samples_per_signal['len'].median())

    cohorts = df['cohort'].unique().to_list() if 'cohort' in df.columns else ['default']
    n_cohorts = len(cohorts)

    # Check for constant signals
    signal_stats = df.group_by('signal_id').agg([
        pl.col('value').std().alias('std'),
        pl.col('value').n_unique().alias('n_unique'),
    ])
    constant_signals = signal_stats.filter(
        (pl.col('std') < 1e-10) | (pl.col('n_unique') <= 1)
    )['signal_id'].to_list()

    # NaN count
    nan_count = int(df['value'].is_null().sum())

    return {
        'n_signals': n_signals,
        'n_cohorts': n_cohorts,
        'signal_list': signal_list,
        'cohorts': cohorts,
        'min_samples': min_samples,
        'max_samples': max_samples,
        'median_samples': median_samples,
        'constant_signals': constant_signals,
        'nan_count': nan_count,
        'total_rows': len(df),
        'ftle_viable': min_samples >= 200,
        'rolling_ftle_viable': min_samples >= 400,
        'granger_viable': n_signals <= 100 and min_samples >= 50,
    }


def generate_auto_manifest(
    characteristics: Dict,
    atlas: bool = False,
    segments: Optional[List[Dict]] = None,
) -> Dict:
    """
    Generate manifest configuration from data characteristics.

    Args:
        characteristics: Output from detect_data_characteristics()
        atlas: Enable all atlas engines
        segments: Segment definitions [{'name': str, 'range': [start, end]}]

    Returns:
        Manifest dict ready for yaml.dump or pipeline.run()
    """
    n_signals = characteristics['n_signals']
    min_samples = characteristics['min_samples']
    signal_list = characteristics['signal_list']
    cohorts = characteristics['cohorts']
    constant_signals = set(characteristics['constant_signals'])

    # Window sizing: capture ~5-10% of samples, clamped to [16, 128]
    raw_window = max(16, min(128, min_samples // 15))
    window = int(2 ** round(np.log2(max(raw_window, 1))))
    stride = max(1, window // 2)

    # Base engines — inclusive philosophy (run everything that could be useful)
    # NOTE: Do not expand this list without first adding min-sample guards to each engine.
    # See PR #1, #3, #6, #7 for the history of broken engines producing null columns.
    base_engines = [
        'crest_factor', 'kurtosis', 'skewness',
        'spectral',
        'sample_entropy', 'perm_entropy',
        'hurst', 'acf_decay',
        'lyapunov',
    ]

    manifest = {
        'version': '2.5',
        'generator': 'engines auto-manifest',
        'paths': {
            'observations': 'observations.parquet',
            'output_dir': 'output/',
        },
        'system': {
            'window': window,
            'stride': stride,
        },
        'engine_windows': {
            'spectral': 64,
            'harmonics': 64,
            'sample_entropy': 64,
            'hurst': 128,
        },
        'params': {
            'default_window': window,
            'default_stride': stride,
            'min_samples': max(16, window),
        },
        'summary': {
            'total_signals': n_signals,
            'active_signals': n_signals - len(constant_signals),
            'constant_signals': len(constant_signals),
            'signal_engines': base_engines,
            'n_signal_engines': len(base_engines),
        },
    }

    # Build cohorts section — per-signal engine configuration
    cohorts_config = {}
    for cohort_name in cohorts:
        cohort_signals = {}
        for sig_id in signal_list:
            if sig_id in constant_signals:
                continue
            cohort_signals[sig_id] = {
                'engines': list(base_engines),
                'rolling_engines': [],
                'window_size': window,
                'stride': stride,
                'derivative_depth': 1,
                'eigenvalue_budget': min(5, n_signals),
            }
            # Add engine_window_overrides if signal window < engine minimums
            overrides = {}
            if window < 64:
                for eng in ['spectral', 'harmonics', 'sample_entropy']:
                    overrides[eng] = 64
            if window < 128:
                overrides['hurst'] = 128
            if overrides:
                cohort_signals[sig_id]['engine_window_overrides'] = overrides

        cohorts_config[str(cohort_name)] = cohort_signals

    manifest['cohorts'] = cohorts_config

    # Skip constant signals
    if constant_signals:
        manifest['skip_signals'] = list(constant_signals)

    # Pair engines (Granger, correlation)
    if n_signals <= 30 and min_samples >= 50:
        manifest['pair_engines'] = ['granger', 'transfer_entropy']
        manifest['symmetric_pair_engines'] = ['correlation', 'mutual_info']
    elif n_signals <= 100 and min_samples >= 50:
        manifest['pair_engines'] = ['granger']
        manifest['symmetric_pair_engines'] = ['correlation']

    # FTLE configuration
    if min_samples >= 200:
        manifest['ftle'] = {
            'directions': ['forward'],
        }
        if min_samples >= 400 and atlas:
            manifest['ftle']['directions'] = ['forward', 'backward']
            manifest['ftle']['rolling'] = True
            manifest['ftle']['rolling_window'] = min(200, min_samples // 3)
            manifest['ftle']['rolling_stride'] = max(10, manifest['ftle']['rolling_window'] // 4)

    # Atlas mode
    if atlas:
        manifest['geometry'] = {
            'full_span': True,
            'full_span_min_window': window,
        }
        manifest['velocity_field'] = {'enabled': True, 'smooth': 'savgol'}
        manifest['break_sequence'] = {'enabled': True, 'reference_index': 0}

        if manifest.get('ftle', {}).get('rolling', False):
            manifest['ridge_proximity'] = {'enabled': True}

    # Segments
    if segments:
        manifest['segments'] = segments

    manifest['_auto_generated'] = True

    return manifest
