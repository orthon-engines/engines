"""
Mass Engine

Computes mass balance: flow in, flow out, accumulation.
Detects leaks and blockages.

Conservation Law: m_dot_in = m_dot_out + dm/dt
Violation means: Leak, blockage, accumulation
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from ..primitives.individual import derivative


@dataclass
class FlowDefinition:
    """Definition of a flow measurement."""
    signal: str
    direction: str = 'out'  # 'in' | 'out'
    units: str = ''  # For documentation


@dataclass
class MassConfig:
    """Configuration for mass engine."""
    dt: float = 1.0  # Time step in seconds
    flow_definitions: List[FlowDefinition] = field(default_factory=list)
    accumulation_signal: Optional[str] = None  # Signal for stored mass
    density: float = 1.0  # Fluid density for conversions
    leak_threshold_pct: float = 5.0  # % residual to flag leak
    window_size: Optional[int] = None
    window_step: Optional[int] = None


class MassEngine:
    """
    Mass Balance Engine.

    Enforces conservation of mass: m_dot_in = m_dot_out + dm/dt

    Outputs:
    - flow_in_total: Total inflow rate
    - flow_out_total: Total outflow rate
    - accumulation_rate: Rate of mass accumulation (dm/dt)
    - mass_balance_residual: Should be ~0 if balanced
    - mass_balance_residual_pct: Percentage residual
    - leak_indicator: Positive residual suggests leak
    - blockage_indicator: Negative residual suggests blockage
    - mass_status: NORMAL, POTENTIAL_LEAK, POTENTIAL_BLOCKAGE
    """

    ENGINE_TYPE = "physics"

    def __init__(self, config: Optional[MassConfig] = None):
        self.config = config or MassConfig()

    def compute(
        self,
        signals: Dict[str, np.ndarray],
        unit_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Compute mass balance for an entity.

        Parameters
        ----------
        signals : dict
            Dictionary mapping signal_id to numpy array of values
        unit_id : str
            Entity identifier

        Returns
        -------
        dict
            Mass balance metrics
        """
        if not signals:
            return self._empty_result(unit_id)

        # Get minimum length
        min_len = min(len(v) for v in signals.values())
        if min_len < 10:
            return self._empty_result(unit_id)

        # Compute flows
        flows = {}
        flow_in_list = []
        flow_out_list = []

        for flow_def in self.config.flow_definitions:
            if flow_def.signal not in signals:
                continue

            try:
                flow_values = np.asarray(signals[flow_def.signal])[:min_len]
                mean_flow = float(np.nanmean(flow_values))
                flows[flow_def.signal] = mean_flow

                if flow_def.direction == 'in':
                    flow_in_list.append(mean_flow)
                else:
                    flow_out_list.append(mean_flow)

            except Exception:
                flows[flow_def.signal] = np.nan

        # Totals
        flow_in_total = sum(flow_in_list) if flow_in_list else 0
        flow_out_total = sum(flow_out_list) if flow_out_list else 0

        # Accumulation rate (dm/dt)
        if self.config.accumulation_signal and self.config.accumulation_signal in signals:
            try:
                acc_values = np.asarray(signals[self.config.accumulation_signal])[:min_len]
                # Rate of change of stored mass
                accumulation_rate = (acc_values[-1] - acc_values[0]) / (min_len * self.config.dt)
            except Exception:
                accumulation_rate = 0
        else:
            accumulation_rate = 0

        # Mass balance: m_dot_in = m_dot_out + dm/dt
        # Residual = m_dot_in - m_dot_out - dm/dt (should be ~0)
        residual = flow_in_total - flow_out_total - accumulation_rate

        if abs(flow_in_total) > 1e-10:
            residual_pct = residual / flow_in_total * 100
        else:
            residual_pct = 0.0

        # Leak/blockage indicators
        leak_indicator = max(0, residual)  # Positive residual = mass leaving
        blockage_indicator = max(0, -residual)  # Negative residual = mass not flowing

        # Status determination
        if flow_in_total > 0:
            leak_pct = leak_indicator / flow_in_total * 100
            blockage_pct = blockage_indicator / flow_in_total * 100
        else:
            leak_pct = 0
            blockage_pct = 0

        if leak_pct > 10:
            status = 'POTENTIAL_LEAK'
        elif blockage_pct > 10:
            status = 'POTENTIAL_BLOCKAGE'
        elif abs(residual_pct) > self.config.leak_threshold_pct:
            status = 'BALANCE_WARNING'
        else:
            status = 'NORMAL'

        return {
            'unit_id': unit_id,
            'n_samples': min_len,
            'flow_in_total': float(flow_in_total),
            'flow_out_total': float(flow_out_total),
            'accumulation_rate': float(accumulation_rate),
            'mass_balance_residual': float(residual),
            'mass_balance_residual_pct': float(residual_pct),
            'leak_indicator': float(leak_indicator),
            'blockage_indicator': float(blockage_indicator),
            'mass_status': status,
            'individual_flows': flows,
        }

    def _empty_result(self, unit_id: str) -> Dict[str, Any]:
        """Return empty result for insufficient data."""
        return {
            'unit_id': unit_id,
            'n_samples': 0,
            'flow_in_total': np.nan,
            'flow_out_total': np.nan,
            'accumulation_rate': np.nan,
            'mass_balance_residual': np.nan,
            'mass_balance_residual_pct': np.nan,
            'leak_indicator': np.nan,
            'blockage_indicator': np.nan,
            'mass_status': 'INSUFFICIENT_DATA',
            'individual_flows': {},
        }

    def to_parquet_row(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert result to flat row for parquet output."""
        row = {
            'unit_id': result['unit_id'],
            'n_samples': result['n_samples'],
            'flow_in_total': result['flow_in_total'],
            'flow_out_total': result['flow_out_total'],
            'accumulation_rate': result['accumulation_rate'],
            'mass_balance_residual': result['mass_balance_residual'],
            'mass_balance_residual_pct': result['mass_balance_residual_pct'],
            'leak_indicator': result['leak_indicator'],
            'blockage_indicator': result['blockage_indicator'],
            'mass_status': result['mass_status'],
        }
        # Add individual flows
        for name, value in result.get('individual_flows', {}).items():
            row[f'flow_{name}'] = value
        return row


def run_mass_engine(
    observations: pl.DataFrame,
    config: MassConfig,
    output_path: Optional[Path] = None
) -> pl.DataFrame:
    """
    Run mass engine on observations DataFrame.

    Parameters
    ----------
    observations : pl.DataFrame
        Observations with unit_id, signal_id, index, value
    config : MassConfig
        Engine configuration
    output_path : Path, optional
        Path to write output parquet

    Returns
    -------
    pl.DataFrame
        Mass balance results
    """
    engine = MassEngine(config)

    entities = observations.select('unit_id').unique().to_series().to_list()
    results = []

    for unit_id in entities:
        entity_obs = observations.filter(pl.col('unit_id') == unit_id)

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
