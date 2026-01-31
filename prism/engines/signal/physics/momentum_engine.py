"""
Momentum Engine

Computes momentum/force/torque balance for rotating and linear systems.
Detects imbalance, bearing issues, structural problems.

Conservation Laws:
- Rotational: Sum(tau) = I * alpha
- Linear: Sum(F) = m * a
Violation means: Imbalance, bearing degradation, structural issues
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from ..primitives.pairwise import product
from ..primitives.individual import derivative, rms


@dataclass
class TorqueDefinition:
    """Definition of a torque measurement."""
    signal: str
    direction: str = 'applied'  # 'applied' | 'load' | 'friction'


@dataclass
class ForceDefinition:
    """Definition of a force measurement."""
    signal: str
    direction: str = 'applied'  # 'applied' | 'load' | 'friction'


@dataclass
class MomentumConfig:
    """Configuration for momentum engine."""
    dt: float = 1.0  # Time step in seconds
    system_type: str = 'rotational'  # 'rotational' | 'linear' | 'both'

    # Rotational parameters
    torque_definitions: List[TorqueDefinition] = field(default_factory=list)
    angular_velocity_signal: Optional[str] = None
    inertia: float = 1.0  # kg*m^2

    # Linear parameters
    force_definitions: List[ForceDefinition] = field(default_factory=list)
    velocity_signal: Optional[str] = None
    mass: float = 1.0  # kg

    # Vibration
    vibration_signals: List[str] = field(default_factory=list)

    # Thresholds
    torque_residual_threshold: float = 10.0  # N*m
    vibration_threshold: float = 1.0  # m/s^2 RMS


class MomentumEngine:
    """
    Momentum Balance Engine.

    Enforces conservation of momentum:
    - Rotational: Sum(tau) = I * alpha
    - Linear: Sum(F) = m * a

    Outputs:
    - torque_applied: Total applied torque
    - torque_load: Total load torque
    - torque_friction: Total friction torque
    - net_torque: Applied - Load - Friction
    - expected_torque: I * alpha
    - torque_residual: Net - Expected (should be ~0)
    - angular_acceleration: d(omega)/dt
    - vibration_energy: Sum of vibration RMS^2
    - momentum_status: NORMAL, IMBALANCE, HIGH_VIBRATION
    """

    ENGINE_TYPE = "physics"

    def __init__(self, config: Optional[MomentumConfig] = None):
        self.config = config or MomentumConfig()

    def compute(
        self,
        signals: Dict[str, np.ndarray],
        unit_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Compute momentum balance for an entity.

        Parameters
        ----------
        signals : dict
            Dictionary mapping signal_id to numpy array of values
        unit_id : str
            Entity identifier

        Returns
        -------
        dict
            Momentum balance metrics
        """
        if not signals:
            return self._empty_result(unit_id)

        min_len = min(len(v) for v in signals.values())
        if min_len < 10:
            return self._empty_result(unit_id)

        result = {
            'unit_id': unit_id,
            'n_samples': min_len,
        }

        # Rotational momentum balance: Sum(tau) = I * alpha
        if self.config.system_type in ['rotational', 'both']:
            result.update(self._compute_rotational(signals, min_len))

        # Linear momentum balance: Sum(F) = m * a
        if self.config.system_type in ['linear', 'both']:
            result.update(self._compute_linear(signals, min_len))

        # Vibration energy
        if self.config.vibration_signals:
            vib_energy = 0
            for vib_sig in self.config.vibration_signals:
                if vib_sig in signals:
                    try:
                        vib = np.asarray(signals[vib_sig])[:min_len]
                        vib_energy += rms(vib) ** 2
                    except Exception:
                        pass
            result['vibration_energy'] = float(vib_energy)
        else:
            result['vibration_energy'] = 0.0

        # Determine overall status
        torque_res = result.get('torque_residual', 0)
        force_res = result.get('force_residual', 0)
        vib_energy = result.get('vibration_energy', 0)

        if abs(torque_res) > self.config.torque_residual_threshold:
            status = 'IMBALANCE'
        elif abs(force_res) > self.config.torque_residual_threshold:
            status = 'IMBALANCE'
        elif vib_energy > self.config.vibration_threshold:
            status = 'HIGH_VIBRATION'
        else:
            status = 'NORMAL'

        result['momentum_status'] = status

        return result

    def _compute_rotational(
        self,
        signals: Dict[str, np.ndarray],
        min_len: int
    ) -> Dict[str, Any]:
        """Compute rotational momentum balance."""
        result = {}

        # Angular velocity and acceleration
        if self.config.angular_velocity_signal and self.config.angular_velocity_signal in signals:
            omega = np.asarray(signals[self.config.angular_velocity_signal])[:min_len]
            alpha = derivative(omega, self.config.dt)
            mean_alpha = float(np.nanmean(alpha))
            expected_torque = self.config.inertia * mean_alpha

            result['angular_velocity_mean'] = float(np.nanmean(omega))
            result['angular_acceleration'] = mean_alpha
            result['expected_torque'] = expected_torque
        else:
            expected_torque = 0
            result['angular_velocity_mean'] = np.nan
            result['angular_acceleration'] = np.nan
            result['expected_torque'] = np.nan

        # Torque components
        torque_applied = 0
        torque_load = 0
        torque_friction = 0

        for torque_def in self.config.torque_definitions:
            if torque_def.signal not in signals:
                continue

            try:
                torque_values = np.asarray(signals[torque_def.signal])[:min_len]
                mean_torque = float(np.nanmean(torque_values))

                if torque_def.direction == 'applied':
                    torque_applied += mean_torque
                elif torque_def.direction == 'load':
                    torque_load += mean_torque
                elif torque_def.direction == 'friction':
                    torque_friction += mean_torque
            except Exception:
                pass

        # Net torque and residual
        net_torque = torque_applied - torque_load - torque_friction
        torque_residual = net_torque - expected_torque if not np.isnan(expected_torque) else net_torque

        result['torque_applied'] = float(torque_applied)
        result['torque_load'] = float(torque_load)
        result['torque_friction'] = float(torque_friction)
        result['net_torque'] = float(net_torque)
        result['torque_residual'] = float(torque_residual)

        return result

    def _compute_linear(
        self,
        signals: Dict[str, np.ndarray],
        min_len: int
    ) -> Dict[str, Any]:
        """Compute linear momentum balance."""
        result = {}

        # Velocity and acceleration
        if self.config.velocity_signal and self.config.velocity_signal in signals:
            velocity = np.asarray(signals[self.config.velocity_signal])[:min_len]
            acceleration = derivative(velocity, self.config.dt)
            mean_accel = float(np.nanmean(acceleration))
            expected_force = self.config.mass * mean_accel

            result['velocity_mean'] = float(np.nanmean(velocity))
            result['acceleration'] = mean_accel
            result['expected_force'] = expected_force
        else:
            expected_force = 0
            result['velocity_mean'] = np.nan
            result['acceleration'] = np.nan
            result['expected_force'] = np.nan

        # Force components
        force_applied = 0
        force_load = 0
        force_friction = 0

        for force_def in self.config.force_definitions:
            if force_def.signal not in signals:
                continue

            try:
                force_values = np.asarray(signals[force_def.signal])[:min_len]
                mean_force = float(np.nanmean(force_values))

                if force_def.direction == 'applied':
                    force_applied += mean_force
                elif force_def.direction == 'load':
                    force_load += mean_force
                elif force_def.direction == 'friction':
                    force_friction += mean_force
            except Exception:
                pass

        # Net force and residual
        net_force = force_applied - force_load - force_friction
        force_residual = net_force - expected_force if not np.isnan(expected_force) else net_force

        result['force_applied'] = float(force_applied)
        result['force_load'] = float(force_load)
        result['force_friction'] = float(force_friction)
        result['net_force'] = float(net_force)
        result['force_residual'] = float(force_residual)

        return result

    def _empty_result(self, unit_id: str) -> Dict[str, Any]:
        """Return empty result for insufficient data."""
        return {
            'unit_id': unit_id,
            'n_samples': 0,
            'torque_applied': np.nan,
            'torque_load': np.nan,
            'torque_friction': np.nan,
            'net_torque': np.nan,
            'expected_torque': np.nan,
            'torque_residual': np.nan,
            'angular_velocity_mean': np.nan,
            'angular_acceleration': np.nan,
            'vibration_energy': np.nan,
            'momentum_status': 'INSUFFICIENT_DATA',
        }

    def to_parquet_row(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert result to flat row for parquet output."""
        return {k: v for k, v in result.items() if not isinstance(v, dict)}


def run_momentum_engine(
    observations: pl.DataFrame,
    config: MomentumConfig,
    output_path: Optional[Path] = None
) -> pl.DataFrame:
    """
    Run momentum engine on observations DataFrame.

    Parameters
    ----------
    observations : pl.DataFrame
        Observations with unit_id, signal_id, index, value
    config : MomentumConfig
        Engine configuration
    output_path : Path, optional
        Path to write output parquet

    Returns
    -------
    pl.DataFrame
        Momentum balance results
    """
    engine = MomentumEngine(config)

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
