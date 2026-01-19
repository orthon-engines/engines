#!/usr/bin/env python3
"""
NASA C-MAPSS Turbofan Engine Degradation Fetcher

Downloads and processes the C-MAPSS (Commercial Modular Aero-Propulsion System
Simulation) dataset from NASA. Contains run-to-failure sensor data for turbofan
engines under various operating conditions.

Dataset Info:
- 4 sub-datasets (FD001-FD004)
- 21 sensors per engine
- Known RUL (Remaining Useful Life) for validation
- Operating conditions and fault modes vary by dataset

Output Format (PRISM Canonical Schema):
    entity_id: string  - Engine identifier (e.g., "FD001_U001")
    signal_id: string  - PRISM signal name (from domain yaml)
    timestamp: float   - Cycle number
    value: float       - Measurement value

Usage:
    from fetchers.cmapss_fetcher import fetch

    config = {
        "datasets": ["FD001", "FD002"],  # Which datasets to fetch
        "data_type": "train",             # "train" or "test"
    }
    observations = fetch(config)
"""

import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

# NASA C-MAPSS download URL
CMAPSS_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
SOURCE = "cmapss"

# Sensor column names (21 sensors)
SENSOR_NAMES = [
    'T2',      # Total temperature at fan inlet (R)
    'T24',     # Total temperature at LPC outlet (R)
    'T30',     # Total temperature at HPC outlet (R)
    'T50',     # Total temperature at LPT outlet (R)
    'P2',      # Pressure at fan inlet (psia)
    'P15',     # Total pressure in bypass-duct (psia)
    'P30',     # Total pressure at HPC outlet (psia)
    'Nf',      # Physical fan speed (rpm)
    'Nc',      # Physical core speed (rpm)
    'epr',     # Engine pressure ratio (P50/P2)
    'Ps30',    # Static pressure at HPC outlet (psia)
    'phi',     # Ratio of fuel flow to Ps30
    'NRf',     # Corrected fan speed (rpm)
    'NRc',     # Corrected core speed (rpm)
    'BPR',     # Bypass ratio
    'farB',    # Burner fuel-air ratio
    'htBleed', # Bleed enthalpy
    'Nf_dmd',  # Demanded fan speed
    'PCNfR_dmd', # Demanded corrected fan speed
    'W31',     # HPT coolant bleed (lbm/s)
    'W32',     # LPT coolant bleed (lbm/s)
]

# Operating condition names
OP_NAMES = ['op1', 'op2', 'op3']

# All column names
ALL_COLUMNS = ['unit', 'cycle'] + OP_NAMES + SENSOR_NAMES


def download_cmapss(cache_dir: Optional[Path] = None) -> Path:
    """
    Download and extract C-MAPSS data.

    Args:
        cache_dir: Directory to cache downloaded data

    Returns:
        Path to extracted data directory
    """
    if cache_dir is None:
        cache_dir = Path(tempfile.gettempdir()) / "cmapss_data"

    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "CMAPSSData.zip"

    # Check if already downloaded (files are extracted to cache_dir directly)
    if (cache_dir / "train_FD001.txt").exists():
        print(f"  Using cached data at {cache_dir}")
        return cache_dir

    # Download
    print(f"  Downloading C-MAPSS data from NASA...")
    urllib.request.urlretrieve(CMAPSS_URL, zip_path)

    # Extract (files go directly into cache_dir)
    print(f"  Extracting to {cache_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(cache_dir)

    return cache_dir


def load_dataset(
    data_dir: Path,
    dataset: str,
    data_type: str = "train"
) -> pd.DataFrame:
    """
    Load a single C-MAPSS dataset.

    Args:
        data_dir: Path to extracted CMAPSSData directory
        dataset: Dataset name (FD001, FD002, FD003, FD004)
        data_type: "train" or "test"

    Returns:
        DataFrame with columns: unit, cycle, op1-3, sensor_1-21
    """
    filename = f"{data_type}_{dataset}.txt"
    filepath = data_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    # Read space-separated file
    df = pd.read_csv(
        filepath,
        sep=r'\s+',
        header=None,
        names=ALL_COLUMNS,
        index_col=False
    )

    # Drop any trailing NaN columns (some files have trailing spaces)
    df = df.dropna(axis=1, how='all')

    return df


def load_rul(data_dir: Path, dataset: str) -> pd.Series:
    """
    Load RUL (Remaining Useful Life) ground truth for test set.

    Args:
        data_dir: Path to extracted CMAPSSData directory
        dataset: Dataset name (FD001, FD002, FD003, FD004)

    Returns:
        Series of RUL values indexed by unit
    """
    filepath = data_dir / f"RUL_{dataset}.txt"

    if not filepath.exists():
        return pd.Series(dtype=float)

    rul = pd.read_csv(filepath, header=None, names=['RUL'])
    rul.index = rul.index + 1  # Units are 1-indexed
    return rul['RUL']


def add_rul_to_train(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate RUL for training data.

    For training data, RUL = max_cycle_for_unit - current_cycle

    Args:
        df: Training DataFrame

    Returns:
        DataFrame with RUL column added
    """
    max_cycles = df.groupby('unit')['cycle'].max()
    df = df.merge(max_cycles.rename('max_cycle'), left_on='unit', right_index=True)
    df['RUL'] = df['max_cycle'] - df['cycle']
    df = df.drop('max_cycle', axis=1)
    return df


def load_domain_yaml() -> Dict[str, Any]:
    """Load the cmapss domain configuration YAML."""
    # Look for domain yaml relative to this file's location
    fetcher_dir = Path(__file__).parent
    project_root = fetcher_dir.parent
    yaml_path = project_root / 'config' / 'domains' / 'cmapss.yaml'

    if not yaml_path.exists():
        print(f"  Warning: Domain config not found at {yaml_path}")
        return {}

    with open(yaml_path) as f:
        return yaml.safe_load(f)


def fetch(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Fetch C-MAPSS observations in PRISM canonical schema.

    Args:
        config: Dict with keys:
            - datasets: List of datasets to fetch (default: ["FD001"])
            - data_type: "train" or "test" (default: "train")
            - include_rul: Include RUL as separate signal (default: True)
            - include_ops: Include operating conditions (default: True)
            - cache_dir: Directory for cached downloads

    Returns:
        List of observation dicts with PRISM canonical schema:
            entity_id: Engine identifier
            signal_id: PRISM internal signal name (from domain yaml)
            timestamp: Cycle number (float)
            value: Measurement value (float)
    """
    datasets = config.get("datasets", ["FD001"])
    data_type = config.get("data_type", "train")
    include_rul = config.get("include_rul", True)
    include_ops = config.get("include_ops", True)
    cache_dir = config.get("cache_dir")

    if cache_dir:
        cache_dir = Path(cache_dir)

    # Load domain YAML for signal mapping
    domain_yaml = load_domain_yaml()

    # Build signal mapping: source_name -> prism_id
    signal_map = {}
    if domain_yaml.get('signals'):
        for source_name, cfg in domain_yaml['signals'].items():
            signal_map[source_name] = cfg['prism_id']

    # Build op_settings mapping
    op_map = {}
    if domain_yaml.get('op_settings'):
        for source_name, cfg in domain_yaml['op_settings'].items():
            op_map[source_name] = cfg.get('prism_id', source_name)

    # Target mapping
    target_prism_id = 'target_rul'  # Default
    if domain_yaml.get('target') and domain_yaml['target'].get('prism_id'):
        target_prism_id = domain_yaml['target']['prism_id']

    print(f"  Signal mappings loaded: {len(signal_map)} sensors, {len(op_map)} op_settings")

    # Download data
    data_dir = download_cmapss(cache_dir)

    all_observations = []

    for dataset in datasets:
        print(f"  Loading {dataset} ({data_type})...")

        try:
            df = load_dataset(data_dir, dataset, data_type)
        except FileNotFoundError as e:
            print(f"    SKIP: {e}")
            continue

        # Add RUL for training data
        if data_type == "train":
            df = add_rul_to_train(df)
        else:
            # For test data, we'd need to look up final RUL
            rul_series = load_rul(data_dir, dataset)
            if not rul_series.empty:
                # Last cycle RUL is from file, compute backwards
                df['RUL'] = None
                for unit in df['unit'].unique():
                    unit_mask = df['unit'] == unit
                    final_rul = rul_series.get(unit, 0)
                    unit_df = df.loc[unit_mask].copy()
                    max_cycle = unit_df['cycle'].max()
                    df.loc[unit_mask, 'RUL'] = final_rul + (max_cycle - df.loc[unit_mask, 'cycle'])

        print(f"    {len(df)} records, {df['unit'].nunique()} units")

        # Convert to PRISM canonical schema
        # entity_id = dataset_unit (engine), signal_id = prism_id from yaml
        # timestamp = cycle number (float)

        for _, row in df.iterrows():
            unit_id = int(row['unit'])
            cycle = int(row['cycle'])
            entity_id = f"{dataset}_U{unit_id:03d}"

            # Sensor observations - map to PRISM signal IDs
            for sensor in SENSOR_NAMES:
                if sensor in df.columns:
                    value = row[sensor]
                    if pd.notna(value):
                        # Use mapped prism_id if available, else original name
                        prism_signal_id = signal_map.get(sensor, sensor)
                        all_observations.append({
                            "entity_id": entity_id,
                            "signal_id": prism_signal_id,
                            "timestamp": float(cycle),
                            "value": float(value),
                        })

            # Operating conditions (optional) - map to PRISM IDs
            if include_ops:
                for op in OP_NAMES:
                    if op in df.columns:
                        value = row[op]
                        if pd.notna(value):
                            prism_op_id = op_map.get(op, op)
                            all_observations.append({
                                "entity_id": entity_id,
                                "signal_id": prism_op_id,
                                "timestamp": float(cycle),
                                "value": float(value),
                            })

            # RUL (optional - useful for ML validation)
            if include_rul and 'RUL' in df.columns and pd.notna(row.get('RUL')):
                all_observations.append({
                    "entity_id": entity_id,
                    "signal_id": target_prism_id,
                    "timestamp": float(cycle),
                    "value": float(row['RUL']),
                })

    print(f"  Total: {len(all_observations):,} observations")
    return all_observations


if __name__ == "__main__":
    # Example usage
    config = {
        "datasets": ["FD001"],
        "data_type": "train",
        "include_rul": True,
    }

    results = fetch(config)
    print(f"\nFetched {len(results):,} observations")

    # Show sample - PRISM canonical schema
    if results:
        df = pd.DataFrame(results[:10])
        print("\nSample observations (PRISM canonical schema):")
        print("Columns: entity_id, signal_id, timestamp, value")
        print(df)
