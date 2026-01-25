"""
PRISM Config Loader — Load config.json from data directory

Usage:
    from prism.config.loader import load_config

    config = load_config("data/cmapss/")
    # Looks for: data/cmapss/config.json
    # Also loads: data/cmapss/observations.parquet
"""

from pathlib import Path
from typing import Union, Tuple, Optional, List
import polars as pl

from .schema import PrismConfig, SignalInfo, WindowConfig


def load_config(data_dir: Union[str, Path]) -> PrismConfig:
    """
    Load config.json from data directory.

    Args:
        data_dir: Path to directory containing config.json

    Returns:
        PrismConfig object

    Raises:
        FileNotFoundError: If config.json not found
        ValidationError: If config.json is invalid
    """
    data_dir = Path(data_dir)
    config_path = data_dir / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {data_dir}")

    return PrismConfig.from_json(config_path)


def load_observations(data_dir: Union[str, Path]) -> pl.DataFrame:
    """
    Load observations.parquet from data directory.

    Args:
        data_dir: Path to directory containing observations.parquet

    Returns:
        Polars DataFrame with schema:
        - entity_id: Utf8
        - signal_id: Utf8
        - index: Float64
        - value: Float64
        - unit: Utf8 (nullable)

    Raises:
        FileNotFoundError: If observations.parquet not found
    """
    data_dir = Path(data_dir)
    obs_path = data_dir / "observations.parquet"

    if not obs_path.exists():
        raise FileNotFoundError(f"observations.parquet not found in {data_dir}")

    return pl.read_parquet(obs_path)


def load_dataset(data_dir: Union[str, Path]) -> Tuple[pl.DataFrame, PrismConfig]:
    """
    Load both observations and config from data directory.

    Args:
        data_dir: Path to directory containing:
            - observations.parquet
            - config.json

    Returns:
        Tuple of (observations DataFrame, config)

    Example:
        observations, config = load_dataset("data/cmapss/")

        for entity in config.entities:
            entity_data = observations.filter(pl.col("entity_id") == entity)
            # Process...
    """
    config = load_config(data_dir)
    observations = load_observations(data_dir)

    return observations, config


def validate_observations(
    observations: pl.DataFrame,
    config: PrismConfig,
) -> List[str]:
    """
    Validate observations against config.

    Returns list of warnings (empty if valid).
    """
    warnings = []

    # Check required columns
    required = ["entity_id", "signal_id", "index", "value"]
    for col in required:
        if col not in observations.columns:
            warnings.append(f"Missing required column: {col}")

    if warnings:
        return warnings  # Can't continue validation

    # Check entities match
    obs_entities = set(observations["entity_id"].unique().to_list())
    config_entities = set(config.entities)

    if obs_entities != config_entities:
        missing = config_entities - obs_entities
        extra = obs_entities - config_entities
        if missing:
            warnings.append(f"Entities in config but not in observations: {missing}")
        if extra:
            warnings.append(f"Entities in observations but not in config: {extra}")

    # Check signals match
    obs_signals = set(observations["signal_id"].unique().to_list())
    config_signals = set(config.signal_ids())

    if obs_signals != config_signals:
        missing = config_signals - obs_signals
        extra = obs_signals - config_signals
        if missing:
            warnings.append(f"Signals in config but not in observations: {missing}")
        if extra:
            warnings.append(f"Signals in observations but not in config: {extra}")

    # Check for nulls in value column
    null_count = observations.filter(pl.col("value").is_null()).height
    if null_count > 0:
        pct = null_count / len(observations) * 100
        warnings.append(f"Null values in observations: {null_count} ({pct:.1f}%)")

    return warnings


# =============================================================================
# CONVENIENCE: Get data for specific entity/signal
# =============================================================================

def get_signal_data(
    observations: pl.DataFrame,
    entity_id: str,
    signal_id: str,
) -> pl.DataFrame:
    """
    Get data for a specific entity and signal.

    Returns DataFrame with columns: index, value, unit
    Sorted by index.
    """
    return (
        observations
        .filter(
            (pl.col("entity_id") == entity_id) &
            (pl.col("signal_id") == signal_id)
        )
        .select(["index", "value", "unit"])
        .sort("index")
    )


def get_entity_data(
    observations: pl.DataFrame,
    entity_id: str,
) -> pl.DataFrame:
    """
    Get all data for a specific entity.

    Returns DataFrame with all signals for this entity.
    """
    return (
        observations
        .filter(pl.col("entity_id") == entity_id)
        .sort(["signal_id", "index"])
    )


def pivot_entity_wide(
    observations: pl.DataFrame,
    entity_id: str,
) -> pl.DataFrame:
    """
    Get entity data in wide format (signals as columns).

    Returns DataFrame with:
    - index as rows
    - signal_id values as columns
    """
    entity_data = get_entity_data(observations, entity_id)

    return (
        entity_data
        .pivot(
            values="value",
            index="index",
            on="signal_id",
        )
        .sort("index")
    )
