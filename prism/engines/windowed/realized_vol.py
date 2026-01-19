"""
PRISM Realized Volatility Engine

Comprehensive short-window metrics for micro tier (21d) and all tiers.
Computes volatility, decline-from-peak, and distribution characteristics.

Metrics (13 total):
  Volatility:
    - realized_vol: Scaled volatility (std dev of changes * sqrt(scale_factor))
    - realized_vol_daily: Per-observation volatility
    - up_vol: Upside volatility (scaled)
    - down_vol: Downside volatility (scaled)
    - vol_skew: Asymmetry ratio (down_vol / up_vol - 1)

  Decline from Peak:
    - max_drawdown: Maximum peak-to-trough decline
    - max_drawdown_duration: Observations from peak to trough
    - avg_drawdown: Average decline depth
    - time_underwater: Fraction of window in decline

  Distribution:
    - skewness: Change distribution skewness
    - kurtosis: Excess kurtosis (fat tails)
    - mean_change: Average change (scaled)
    - signal_to_noise: Mean/volatility ratio (signal-to-noise)

Min observations: 15
Phase: Unbound
Normalization: None (works on levels, computes changes internally)

Note: Default scale factor is 252 (daily observations). For other domains:
  - Sensor data (per-cycle): use scale_factor=1
  - Climate (daily): use scale_factor=365
  - Configure via domain settings.
"""

import logging
from typing import Dict

import numpy as np

from prism.engines.engine_base import BaseEngine
from prism.engines.metadata import EngineMetadata


logger = logging.getLogger(__name__)


METADATA = EngineMetadata(
    name="realized_vol",
    engine_type="vector",
    description="Realized volatility, decline-from-peak, and distribution metrics",
    domains={"volatility", "variability", "decline", "signal_topology"},
    requires_window=True,
    deterministic=True,
)


# =============================================================================
# Configuration
# =============================================================================

def _get_scale_factor() -> int:
    """Load scale factor from domain_info or domain.yaml. Fails if not configured."""
    import os
    import json

    # Try domain_info.json first
    try:
        from prism.db.parquet_store import get_data_root
        domain_info_path = get_data_root() / "domain_info.json"
        if domain_info_path.exists():
            with open(domain_info_path) as f:
                info = json.load(f)
            # scale_factor = observations per year/period for volatility scaling
            # For cycle-based data, this is 1 (no annualization)
            sf = info.get('scale_factor')
            if sf:
                return sf
            # Infer from sampling_rate_hz if available
            sr = info.get('sampling_rate_hz')
            if sr:
                # Convert to yearly equivalent
                return int(sr * 3600 * 24 * 365)
    except Exception:
        pass

    # Try domain.yaml
    try:
        from prism.config.loader import load_clock_config
        config = load_clock_config()
        sf = config.get('scale_factor')
        if sf:
            return sf
    except Exception:
        pass

    raise RuntimeError(
        "No scale_factor configured for volatility calculation. "
        "Configure 'scale_factor' in domain_info.json or config/domain.yaml. "
        "Use 1 for per-cycle data, 365 for daily climate, etc."
    )


# =============================================================================
# Vector Engine Contract: Simple function interface
# =============================================================================

def compute_realized_vol(values: np.ndarray, min_obs: int = 15, scale_factor: int = None) -> dict:
    """
    Compute comprehensive short-window metrics from a value series.

    Args:
        values: Array of observed values (levels, measurements, readings)
        min_obs: Minimum observations required
        scale_factor: Scaling factor for volatility. If None, loads from domain config.
                      Use 1 for per-cycle sensor data, 365 for daily climate, etc.

    Returns:
        Dict of metric_name -> metric_value (13 metrics)
    """
    if scale_factor is None:
        scale_factor = _get_scale_factor()
    if len(values) < min_obs:
        return {}

    try:
        values = np.asarray(values, dtype=np.float64)

        # Remove nan/inf
        valid_mask = np.isfinite(values)
        if not np.all(valid_mask):
            values = values[valid_mask]

        if len(values) < min_obs:
            return {}

        n = len(values)

        # =====================================================================
        # Period-over-period changes (log or simple)
        # =====================================================================
        if np.any(values <= 0):
            # Use simple fractional changes for non-positive values
            changes = np.diff(values) / np.where(values[:-1] != 0, values[:-1], 1)
        else:
            # Log changes for strictly positive values
            changes = np.diff(np.log(values))

        changes = changes[np.isfinite(changes)]
        if len(changes) < 5:
            return {}

        # =====================================================================
        # Volatility metrics
        # =====================================================================
        obs_vol = float(np.std(changes, ddof=1))
        scaled_vol = obs_vol * np.sqrt(scale_factor)

        # Upside/downside volatility
        up_changes = changes[changes > 0]
        down_changes = changes[changes < 0]

        up_vol = float(np.std(up_changes, ddof=1)) * np.sqrt(scale_factor) if len(up_changes) > 2 else 0.0
        down_vol = float(np.std(down_changes, ddof=1)) * np.sqrt(scale_factor) if len(down_changes) > 2 else 0.0

        # Vol skew (positive means downside vol > upside vol)
        vol_skew = (down_vol / up_vol - 1.0) if up_vol > 0 else 0.0

        # =====================================================================
        # Decline-from-peak metrics
        # =====================================================================
        running_max = np.maximum.accumulate(values)
        declines = np.where(running_max > 0, (running_max - values) / running_max, 0)

        max_dd = float(np.max(declines))
        max_dd_idx = int(np.argmax(declines))

        # Duration: find peak before trough
        peak_idx = int(np.argmax(values[:max_dd_idx + 1])) if max_dd_idx > 0 else 0
        max_dd_duration = max_dd_idx - peak_idx

        # Average decline (excluding peaks)
        nonzero_dd = declines[declines > 0.001]
        avg_dd = float(np.mean(nonzero_dd)) if len(nonzero_dd) > 0 else 0.0

        # Time underwater (fraction of observations below previous peak)
        time_underwater = float(np.sum(declines > 0.001) / n)

        # =====================================================================
        # Distribution metrics
        # =====================================================================
        mean_change = float(np.mean(changes))
        mean_change_scaled = mean_change * scale_factor

        # Skewness (Fisher-Pearson)
        m2 = np.mean((changes - mean_change) ** 2)
        m3 = np.mean((changes - mean_change) ** 3)
        skewness = float(m3 / (m2 ** 1.5)) if m2 > 0 else 0.0

        # Excess kurtosis
        m4 = np.mean((changes - mean_change) ** 4)
        kurtosis = float(m4 / (m2 ** 2) - 3.0) if m2 > 0 else 0.0

        # Signal-to-noise ratio (mean / volatility)
        signal_to_noise = float(mean_change_scaled / scaled_vol) if scaled_vol > 0 else 0.0

        return {
            # Volatility (5)
            'realized_vol': scaled_vol,
            'realized_vol_daily': obs_vol,
            'up_vol': up_vol,
            'down_vol': down_vol,
            'vol_skew': vol_skew,
            # Decline from peak (4)
            'max_drawdown': max_dd,
            'max_drawdown_duration': float(max_dd_duration),
            'avg_drawdown': avg_dd,
            'time_underwater': time_underwater,
            # Distribution (4)
            'skewness': skewness,
            'kurtosis': kurtosis,
            'mean_change': mean_change_scaled,
            'signal_to_noise': signal_to_noise,
        }

    except Exception as e:
        logger.debug(f"Realized vol computation failed: {e}")
        return {}


# =============================================================================
# Legacy Class Interface (for backwards compatibility)
# =============================================================================

class RealizedVolEngine(BaseEngine):
    """Realized volatility engine class interface."""

    name = "realized_vol"
    phase = "derived"
    default_normalization = None

    @property
    def metadata(self) -> EngineMetadata:
        return METADATA

    def run(self, df, run_id: str, **params) -> Dict:
        """Run realized vol on dataframe columns."""
        results = {}
        for col in df.columns:
            values = df[col].dropna().values
            metrics = compute_realized_vol(values)
            if metrics:
                results[col] = metrics
        return results


# =============================================================================
# Standalone function with derivation
# =============================================================================

def compute_realized_vol_with_derivation(
    values: np.ndarray,
    signal_id: str = "unknown",
    window_id: str = "0",
    window_start: str = None,
    window_end: str = None,
    min_obs: int = 15,
    scale_factor: int = None,
) -> tuple:
    """
    Compute realized volatility with full mathematical derivation.

    Args:
        values: Array of observed values (levels, measurements)
        signal_id: Signal identifier
        window_id: Window identifier
        window_start, window_end: Date range
        min_obs: Minimum observations required
        scale_factor: Scaling factor for volatility. If None, loads from domain config.

    Returns:
        tuple: (result_dict, Derivation object)
    """
    if scale_factor is None:
        scale_factor = _get_scale_factor()

    from prism.entry_points.derivations.base import Derivation

    n = len(values)
    values = np.asarray(values, dtype=np.float64)

    # Remove nan/inf
    valid_mask = np.isfinite(values)
    if not np.all(valid_mask):
        values = values[valid_mask]
        n = len(values)

    deriv = Derivation(
        engine_name="realized_vol",
        method_name="Realized Volatility & Distribution Metrics",
        signal_id=signal_id,
        window_id=window_id,
        window_start=window_start,
        window_end=window_end,
        sample_size=n,
        parameters={'min_obs': min_obs, 'scale_factor': scale_factor}
    )

    # Step 1: Input data
    deriv.add_step(
        title="Input Value Series",
        equation="V = {V₁, V₂, ..., Vₙ}",
        calculation=f"Series: {signal_id}\n"
                    f"n = {n} observations\n\n"
                    f"Value statistics:\n"
                    f"  First: {values[0]:.4f}\n"
                    f"  Last: {values[-1]:.4f}\n"
                    f"  Min: {np.min(values):.4f}\n"
                    f"  Max: {np.max(values):.4f}\n"
                    f"  Range: {np.max(values) - np.min(values):.4f}",
        result=n,
        result_name="n",
        notes="Level data (not changes) as input"
    )

    # Step 2: Compute period-over-period changes
    if np.any(values <= 0):
        changes = np.diff(values) / np.where(values[:-1] != 0, values[:-1], 1)
        change_type = "simple"
    else:
        changes = np.diff(np.log(values))
        change_type = "log"

    changes = changes[np.isfinite(changes)]
    n_changes = len(changes)

    deriv.add_step(
        title="Compute Period Changes",
        equation="Δₜ = ln(Vₜ/Vₜ₋₁)" if change_type == "log" else "Δₜ = (Vₜ - Vₜ₋₁)/Vₜ₋₁",
        calculation=f"Change type: {change_type}\n"
                    f"n_changes = {n_changes}\n\n"
                    f"Change statistics:\n"
                    f"  Mean: {np.mean(changes):.6f}\n"
                    f"  Std: {np.std(changes):.6f}\n"
                    f"  Min: {np.min(changes):.6f}\n"
                    f"  Max: {np.max(changes):.6f}",
        result=np.mean(changes),
        result_name="μ_Δ",
        notes="Log changes preferred for strictly positive values"
    )

    # Step 3: Realized volatility
    obs_vol = float(np.std(changes, ddof=1))
    scaled_vol = obs_vol * np.sqrt(scale_factor)

    deriv.add_step(
        title="Realized Volatility",
        equation=f"σ_obs = √[Σ(Δₜ - μ)² / (n-1)],  σ_scaled = σ_obs × √{scale_factor}",
        calculation=f"Per-observation volatility:\n"
                    f"  σ_obs = {obs_vol:.6f}\n\n"
                    f"Scaled volatility (factor={scale_factor}):\n"
                    f"  σ_scaled = {obs_vol:.6f} × √{scale_factor}\n"
                    f"  σ_scaled = {obs_vol:.6f} × {np.sqrt(scale_factor):.4f}\n"
                    f"  σ_scaled = {scaled_vol:.4f} ({scaled_vol*100:.2f}%)",
        result=scaled_vol,
        result_name="σ",
        notes=f"Scale factor {scale_factor} converts per-observation to scaled volatility"
    )

    # Step 4: Upside/downside volatility
    up_changes = changes[changes > 0]
    down_changes = changes[changes < 0]

    up_vol = float(np.std(up_changes, ddof=1)) * np.sqrt(scale_factor) if len(up_changes) > 2 else 0.0
    down_vol = float(np.std(down_changes, ddof=1)) * np.sqrt(scale_factor) if len(down_changes) > 2 else 0.0
    vol_skew = (down_vol / up_vol - 1.0) if up_vol > 0 else 0.0

    deriv.add_step(
        title="Asymmetric Volatility",
        equation="σ_up = std(Δₜ | Δₜ > 0),  σ_down = std(Δₜ | Δₜ < 0)",
        calculation=f"Upside changes: {len(up_changes)} observations\n"
                    f"  σ_up (scaled) = {up_vol:.4f} ({up_vol*100:.2f}%)\n\n"
                    f"Downside changes: {len(down_changes)} observations\n"
                    f"  σ_down (scaled) = {down_vol:.4f} ({down_vol*100:.2f}%)\n\n"
                    f"Volatility skew:\n"
                    f"  skew = σ_down/σ_up - 1 = {vol_skew:.4f}\n"
                    f"  {'Downside > Upside' if vol_skew > 0 else 'Upside > Downside'}",
        result=vol_skew,
        result_name="skew_vol",
        notes="Positive skew indicates more downside volatility"
    )

    # Step 5: Decline-from-peak analysis
    running_max = np.maximum.accumulate(values)
    declines = np.where(running_max > 0, (running_max - values) / running_max, 0)

    max_dd = float(np.max(declines))
    max_dd_idx = int(np.argmax(declines))
    peak_idx = int(np.argmax(values[:max_dd_idx + 1])) if max_dd_idx > 0 else 0
    max_dd_duration = max_dd_idx - peak_idx

    nonzero_dd = declines[declines > 0.001]
    avg_dd = float(np.mean(nonzero_dd)) if len(nonzero_dd) > 0 else 0.0
    time_underwater = float(np.sum(declines > 0.001) / n)

    deriv.add_step(
        title="Decline-from-Peak Analysis",
        equation="Dₜ = (Peak_t - Vₜ) / Peak_t where Peak_t = max(V₁...Vₜ)",
        calculation=f"Maximum decline:\n"
                    f"  Max decline = {max_dd:.4f} ({max_dd*100:.2f}%)\n"
                    f"  Peak at index {peak_idx}\n"
                    f"  Trough at index {max_dd_idx}\n"
                    f"  Duration: {max_dd_duration} observations\n\n"
                    f"Decline statistics:\n"
                    f"  Average decline (when below peak): {avg_dd:.4f} ({avg_dd*100:.2f}%)\n"
                    f"  Time below peak: {time_underwater:.2%}",
        result=max_dd,
        result_name="D_max",
        notes="Decline measures peak-to-trough drop as fraction of peak"
    )

    # Step 6: Distribution moments
    mean_change = float(np.mean(changes))
    mean_change_scaled = mean_change * scale_factor

    # Skewness (Fisher-Pearson)
    m2 = np.mean((changes - mean_change) ** 2)
    m3 = np.mean((changes - mean_change) ** 3)
    skewness = float(m3 / (m2 ** 1.5)) if m2 > 0 else 0.0

    # Excess kurtosis
    m4 = np.mean((changes - mean_change) ** 4)
    kurtosis = float(m4 / (m2 ** 2) - 3.0) if m2 > 0 else 0.0

    deriv.add_step(
        title="Change Distribution Moments",
        equation="Skew = E[(Δ-μ)³]/σ³,  Kurt = E[(Δ-μ)⁴]/σ⁴ - 3",
        calculation=f"Mean change:\n"
                    f"  μ_obs = {mean_change:.6f}\n"
                    f"  μ_scaled = {mean_change_scaled:.4f} ({mean_change_scaled*100:.2f}%)\n\n"
                    f"Skewness:\n"
                    f"  γ₁ = {skewness:.4f}\n"
                    f"  {'Negative skew (left tail)' if skewness < -0.5 else 'Positive skew (right tail)' if skewness > 0.5 else 'Approximately symmetric'}\n\n"
                    f"Excess Kurtosis:\n"
                    f"  γ₂ = {kurtosis:.4f}\n"
                    f"  {'Fat tails (leptokurtic)' if kurtosis > 1 else 'Thin tails (platykurtic)' if kurtosis < -1 else 'Near-normal tails'}",
        result=skewness,
        result_name="γ₁",
        notes="Normal distribution: skew=0, kurtosis=0"
    )

    # Step 7: Signal-to-noise ratio
    signal_to_noise = float(mean_change_scaled / scaled_vol) if scaled_vol > 0 else 0.0

    deriv.add_step(
        title="Signal-to-Noise Ratio",
        equation="SNR = μ_scaled / σ_scaled",
        calculation=f"Signal-to-noise ratio:\n"
                    f"  SNR = {mean_change_scaled:.4f} / {scaled_vol:.4f}\n"
                    f"  SNR = {signal_to_noise:.4f}\n\n"
                    f"Interpretation:\n"
                    f"  SNR > 1.0: Strong signal relative to noise\n"
                    f"  SNR > 0.5: Moderate signal\n"
                    f"  SNR > 0.0: Positive but weak signal\n"
                    f"  SNR < 0.0: Negative signal (downward trend)",
        result=signal_to_noise,
        result_name="SNR",
        notes="Higher values indicate stronger signal relative to variability"
    )

    # Final result
    result = {
        'realized_vol': scaled_vol,
        'realized_vol_daily': obs_vol,
        'up_vol': up_vol,
        'down_vol': down_vol,
        'vol_skew': vol_skew,
        'max_drawdown': max_dd,
        'max_drawdown_duration': float(max_dd_duration),
        'avg_drawdown': avg_dd,
        'time_underwater': time_underwater,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'mean_change': mean_change_scaled,
        'signal_to_noise': signal_to_noise,
    }

    deriv.final_result = scaled_vol
    deriv.prism_output = scaled_vol

    # Interpretation
    if scaled_vol > 0.5:
        interp = f"**High volatility** ({scaled_vol*100:.1f}% scaled)."
    elif scaled_vol > 0.2:
        interp = f"**Moderate volatility** ({scaled_vol*100:.1f}% scaled)."
    else:
        interp = f"**Low volatility** ({scaled_vol*100:.1f}% scaled)."

    if vol_skew > 0.2:
        interp += f" Asymmetric with {vol_skew*100:.0f}% higher downside vol."

    if kurtosis > 3:
        interp += f" Fat tails present (excess kurtosis = {kurtosis:.2f})."

    if max_dd > 0.2:
        interp += f" Max decline {max_dd*100:.1f}% over {max_dd_duration} observations."

    deriv.interpretation = interp

    return result, deriv
