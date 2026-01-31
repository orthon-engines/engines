"""
Energy Engine

Computes energy balance: input power, output power, stored energy,
dissipated energy, and efficiency. Detects energy anomalies.

Conservation Law: Ein = Eout + Estored + Edissipated
Violation means: Leak, friction change, sensor fault
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from ..primitives.pairwise import product
from ..primitives.individual import integral, derivative, rms


@dataclass
class PowerDefinition:
    """Definition of a power term."""
    name: str
    type: str = 'product'  # 'product' | 'direct'
    signal_a: Optional[str] = None  # For product type
    signal_b: Optional[str] = None  # For product type
    signal: Optional[str] = None  # For direct type
    direction: str = 'out'  # 'in' | 'out' | 'stored' | 'dissipated'


@dataclass
class EnergyConfig:
    """Configuration for energy engine."""
    dt: float = 1.0  # Time step in seconds
    power_definitions: List[PowerDefinition] = field(default_factory=list)
    efficiency_numerator: List[str] = field(default_factory=list)  # Power names for output
    efficiency_denominator: List[str] = field(default_factory=list)  # Power names for input
    balance_threshold_pct: float = 5.0  # % residual to flag as anomaly
    window_size: Optional[int] = None
    window_step: Optional[int] = None


class EnergyEngine:
    """
    Energy Balance Engine.

    Enforces conservation of energy: Ein = Eout + Estored + Edissipated

    Outputs:
    - power_in_total: Total input power
    - power_out_total: Total output power
    - power_stored: Rate of energy storage
    - power_dissipated: Dissipation rate
    - energy_in: Integrated input energy
    - energy_out: Integrated output energy
    - efficiency: Output/Input power ratio
    - balance_residual: Should be ~0 if balanced
    - balance_residual_pct: Percentage residual
    - energy_status: NORMAL, BALANCE_ERROR, HIGH_DISSIPATION, LOW_EFFICIENCY
    """

    ENGINE_TYPE = "physics"

    def __init__(self, config: Optional[EnergyConfig] = None):
        self.config = config or EnergyConfig()

    def compute(
        self,
        signals: Dict[str, np.ndarray],
        unit_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Compute energy balance for an entity.

        Parameters
        ----------
        signals : dict
            Dictionary mapping signal_id to numpy array of values
        unit_id : str
            Entity identifier

        Returns
        -------
        dict
            Energy balance metrics
        """
        if not signals:
            return self._empty_result(unit_id)

        # Get minimum length
        min_len = min(len(v) for v in signals.values())
        if min_len < 10:
            return self._empty_result(unit_id)

        # Compute each power term
        powers = {}
        power_in_list = []
        power_out_list = []
        power_stored_list = []
        power_dissipated_list = []

        for power_def in self.config.power_definitions:
            try:
                if power_def.type == 'product':
                    if power_def.signal_a not in signals or power_def.signal_b not in signals:
                        continue
                    sig_a = np.asarray(signals[power_def.signal_a])[:min_len]
                    sig_b = np.asarray(signals[power_def.signal_b])[:min_len]
                    power_values = product(sig_a, sig_b)
                elif power_def.type == 'direct':
                    if power_def.signal not in signals:
                        continue
                    power_values = np.asarray(signals[power_def.signal])[:min_len]
                else:
                    continue

                mean_power = float(np.nanmean(power_values))
                powers[power_def.name] = mean_power

                if power_def.direction == 'in':
                    power_in_list.append(mean_power)
                elif power_def.direction == 'out':
                    power_out_list.append(mean_power)
                elif power_def.direction == 'stored':
                    power_stored_list.append(mean_power)
                elif power_def.direction == 'dissipated':
                    power_dissipated_list.append(mean_power)

            except Exception:
                powers[power_def.name] = np.nan

        # Totals
        total_in = sum(power_in_list) if power_in_list else 0
        total_out = sum(power_out_list) if power_out_list else 0
        total_stored = sum(power_stored_list) if power_stored_list else 0
        total_dissipated = sum(power_dissipated_list) if power_dissipated_list else 0

        # Energy (integrated power)
        window_duration = min_len * self.config.dt
        energy_in = total_in * window_duration
        energy_out = total_out * window_duration

        # Balance residual: Ein - Eout - Estored - Edissipated should be ~0
        balance_residual = total_in - total_out - total_stored - total_dissipated

        if abs(total_in) > 1e-10:
            balance_residual_pct = balance_residual / total_in * 100
        else:
            balance_residual_pct = 0.0

        # Efficiency
        if self.config.efficiency_numerator and self.config.efficiency_denominator:
            num_powers = sum(powers.get(n, 0) for n in self.config.efficiency_numerator)
            den_powers = sum(powers.get(n, 0) for n in self.config.efficiency_denominator)
            efficiency = num_powers / den_powers if den_powers > 0 else 0
        elif total_in > 0:
            efficiency = total_out / total_in
        else:
            efficiency = 0

        # Dissipation percentage
        if total_in > 0:
            dissipation_pct = total_dissipated / total_in * 100
        else:
            dissipation_pct = 0

        # Status determination
        if abs(balance_residual_pct) > 10:
            status = 'BALANCE_ERROR'
        elif dissipation_pct > 20:
            status = 'HIGH_DISSIPATION'
        elif efficiency < 0.7 and total_in > 0:
            status = 'LOW_EFFICIENCY'
        else:
            status = 'NORMAL'

        return {
            'unit_id': unit_id,
            'n_samples': min_len,
            'power_in_total': float(total_in),
            'power_out_total': float(total_out),
            'power_stored': float(total_stored),
            'power_dissipated': float(total_dissipated),
            'energy_in': float(energy_in),
            'energy_out': float(energy_out),
            'efficiency': float(efficiency),
            'dissipation_pct': float(dissipation_pct),
            'balance_residual': float(balance_residual),
            'balance_residual_pct': float(balance_residual_pct),
            'energy_status': status,
            'individual_powers': powers,
        }

    def _empty_result(self, unit_id: str) -> Dict[str, Any]:
        """Return empty result for insufficient data."""
        return {
            'unit_id': unit_id,
            'n_samples': 0,
            'power_in_total': np.nan,
            'power_out_total': np.nan,
            'power_stored': np.nan,
            'power_dissipated': np.nan,
            'energy_in': np.nan,
            'energy_out': np.nan,
            'efficiency': np.nan,
            'dissipation_pct': np.nan,
            'balance_residual': np.nan,
            'balance_residual_pct': np.nan,
            'energy_status': 'INSUFFICIENT_DATA',
            'individual_powers': {},
        }

    def to_parquet_row(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert result to flat row for parquet output."""
        row = {
            'unit_id': result['unit_id'],
            'n_samples': result['n_samples'],
            'power_in_total': result['power_in_total'],
            'power_out_total': result['power_out_total'],
            'power_stored': result['power_stored'],
            'power_dissipated': result['power_dissipated'],
            'energy_in': result['energy_in'],
            'energy_out': result['energy_out'],
            'efficiency': result['efficiency'],
            'dissipation_pct': result['dissipation_pct'],
            'balance_residual': result['balance_residual'],
            'balance_residual_pct': result['balance_residual_pct'],
            'energy_status': result['energy_status'],
        }
        # Add individual powers
        for name, value in result.get('individual_powers', {}).items():
            row[f'power_{name}'] = value
        return row


def run_energy_engine(
    observations: pl.DataFrame,
    config: EnergyConfig,
    output_path: Optional[Path] = None
) -> pl.DataFrame:
    """
    Run energy engine on observations DataFrame.

    Parameters
    ----------
    observations : pl.DataFrame
        Observations with unit_id, signal_id, index, value
    config : EnergyConfig
        Engine configuration
    output_path : Path, optional
        Path to write output parquet

    Returns
    -------
    pl.DataFrame
        Energy balance results
    """
    engine = EnergyEngine(config)

    # Pivot observations to wide format
    entities = observations.select('unit_id').unique().to_series().to_list()
    results = []

    for unit_id in entities:
        entity_obs = observations.filter(pl.col('unit_id') == unit_id)

        # Extract signals
        signals = {}
        for sig_id in entity_obs.select('signal_id').unique().to_series().to_list():
            sig_data = (
                entity_obs
                .filter(pl.col('signal_id') == sig_id)
                .sort('index')
                .select('value')
                .to_series()
                .to_numpy()
            )
            signals[sig_id] = sig_data

        result = engine.compute(signals, unit_id)
        results.append(engine.to_parquet_row(result))

    df = pl.DataFrame(results)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_path)

    return df
