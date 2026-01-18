#!/usr/bin/env python3
"""
Ingest missing C-MAPSS test units into PRISM format.

Creates observations in the format: FD001_1XXX_sensor
where XXX is the test unit number + 1000.
"""

import polars as pl
from pathlib import Path
from datetime import date, timedelta

DATA_DIR = Path('/Users/jasonrudder/prism-mac/data')
CMAPSS_DIR = DATA_DIR / 'CMAPSSData'
PRISM_DIR = DATA_DIR / 'cmapss_fd001'

# Sensor column mapping (s1-s21 -> named sensors)
SENSOR_MAP = {
    's1': 'T2',       # Total temperature at fan inlet (R)
    's2': 'T24',      # Total temperature at LPC outlet (R)
    's3': 'T30',      # Total temperature at HPC outlet (R)
    's4': 'T50',      # Total temperature at LPT outlet (R)
    's5': 'P2',       # Pressure at fan inlet (psia)
    's6': 'P15',      # Total pressure in bypass-duct (psia)
    's7': 'P30',      # Total pressure at HPC outlet (psia)
    's8': 'Nf',       # Physical fan speed (rpm)
    's9': 'Nc',       # Physical core speed (rpm)
    's10': 'epr',     # Engine pressure ratio (P50/P2)
    's11': 'Ps30',    # Static pressure at HPC outlet (psia)
    's12': 'phi',     # Ratio of fuel flow to Ps30
    's13': 'NRf',     # Corrected fan speed (rpm)
    's14': 'NRc',     # Corrected core speed (rpm)
    's15': 'BPR',     # Bypass ratio
    's16': 'farB',    # Burner fuel-air ratio
    's17': 'htBleed', # Bleed enthalpy
    's18': 'Nf_dmd',  # Demanded fan speed
    's19': 'PCNfR_dmd', # Demanded corrected fan speed
    's20': 'W31',     # HPT coolant bleed (lbm/s)
    's21': 'W32',     # LPT coolant bleed (lbm/s)
}


def get_existing_test_units():
    """Get test units that already have PRISM features."""
    vec = pl.read_parquet(PRISM_DIR / 'vector' / 'indicator.parquet')
    vec = vec.with_columns(
        pl.col('indicator_id').str.extract(r'FD001_1(\d{3})_', 1).cast(pl.Int64).alias('test_unit')
    ).filter(pl.col('test_unit').is_not_null())

    return set(vec['test_unit'].unique().to_list())


def get_missing_test_units():
    """Get test units that need PRISM features."""
    existing = get_existing_test_units()
    all_test = set(range(1, 101))
    return sorted(all_test - existing)


def ingest_test_data():
    """Ingest missing test units into PRISM observation format."""

    # Load test data
    test_df = pl.read_parquet(CMAPSS_DIR / 'test_FD001.parquet')

    # Get missing units
    missing = get_missing_test_units()
    print(f"Missing test units: {len(missing)}")
    print(f"Units: {missing}")

    # Filter to missing units
    test_missing = test_df.filter(pl.col('unit').is_in(missing))
    print(f"\nRows for missing units: {len(test_missing)}")

    # Base date for synthetic time series
    base_date = date(2000, 1, 1)

    # Convert to PRISM observation format
    observations = []

    for row in test_missing.iter_rows(named=True):
        unit = row['unit']
        cycle = row['cycle']

        # Test units use 1000 + unit offset
        unit_offset = 1000 + unit

        # Create synthetic date (same as fetcher logic)
        obs_date = base_date + timedelta(days=(unit_offset - 1) * 500 + cycle)

        # Create observations for each sensor
        for s_col, sensor_name in SENSOR_MAP.items():
            if s_col in row and row[s_col] is not None:
                observations.append({
                    'indicator_id': f'FD001_{unit_offset:04d}_{sensor_name}',
                    'obs_date': obs_date,
                    'value': float(row[s_col]),
                })

    obs_df = pl.DataFrame(observations)

    # Convert date to datetime to match existing schema
    obs_df = obs_df.with_columns(
        pl.col('obs_date').cast(pl.Datetime('us')).alias('obs_date')
    )

    print(f"\nCreated {len(obs_df)} observations")
    print(f"Unique indicators: {obs_df['indicator_id'].n_unique()}")

    # Load existing observations
    obs_path = PRISM_DIR / 'raw' / 'observations.parquet'
    if obs_path.exists():
        existing_obs = pl.read_parquet(obs_path)
        print(f"\nExisting observations: {len(existing_obs)}")

        # Check if already has test data in this format
        test_in_existing = existing_obs.filter(
            pl.col('indicator_id').str.contains(r'FD001_1\d{3}_')
        )
        print(f"Existing test observations: {len(test_in_existing)}")

        # Combine with new
        combined = pl.concat([existing_obs, obs_df])

        # Deduplicate
        combined = combined.unique(['indicator_id', 'obs_date'])
        print(f"\nCombined (deduped): {len(combined)}")
    else:
        combined = obs_df

    # Save
    combined.write_parquet(obs_path)
    print(f"\nSaved to {obs_path}")

    return obs_df


if __name__ == '__main__':
    ingest_test_data()
