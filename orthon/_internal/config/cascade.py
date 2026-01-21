"""
PRISM Cascade Geometry Configuration
=====================================
Window sizes and flag thresholds for multi-pass geometry computation.

The cascade system progressively drills down on flagged pairs:
- Pass 1: All pairs at largest window (coarse scan)
- Pass 2: Flagged pairs at medium window (structural shift detection)
- Pass 3: Double-flagged pairs at finest window (event classification)

This reduces compute by ~60% while capturing structural events.
"""

import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


# =============================================================================
# Window Cascade Configuration
# =============================================================================

def _load_window_cascade() -> dict:
    """Load window cascade from config/stride.yaml. Fails if not configured."""
    try:
        from orthon._internal.utils.stride import load_stride_config
        config = load_stride_config()
        if hasattr(config, 'cascades') and config.cascades:
            return config.cascades
        # Build from windows if cascades not defined
        if hasattr(config, 'windows') and config.windows:
            windows = sorted(config.windows.keys(), key=lambda x: config.windows[x].window_days, reverse=True)
            if len(windows) >= 3:
                cascade = {}
                for freq in ['daily', 'annual', 'monthly', 'per_cycle']:
                    cascade[freq] = {
                        'pass_1': config.windows[windows[0]].window_days,
                        'pass_2': config.windows[windows[1]].window_days if len(windows) > 1 else config.windows[windows[0]].window_days,
                        'pass_3': config.windows[windows[2]].window_days if len(windows) > 2 else config.windows[windows[0]].window_days,
                    }
                return cascade
    except Exception as e:
        raise RuntimeError(f"Failed to load window cascade from config: {e}")

    raise RuntimeError(
        "No window cascade configured in config/stride.yaml. "
        "Configure domain-specific window sizes before running."
    )


# Load at module level - will fail if not configured
WINDOW_CASCADE = _load_window_cascade()


# =============================================================================
# Flag Thresholds
# =============================================================================

FLAG_THRESHOLDS = {
    # Structural shift: geometry changed > 30% between windows
    'structural_shift': 0.3,

    # Entropy coupling thresholds
    'entropy_coupling_high': 0.7,   # unusually high coupling
    'entropy_coupling_low': 0.2,    # unusually low (decoupling)

    # Phase coherence breakdown threshold
    'phase_coherence_break': 0.4,

    # Correlation threshold for mismatch detection
    'correlation_threshold': 0.7,

    # Phase coherence thresholds for breakdown detection
    'phase_coherence_high': 0.6,    # was coherent
    'phase_coherence_low': 0.3,     # now broken

    # Structural variance multiplier for volatility cluster detection
    'variance_spike_multiplier': 2.0,

    # Multi-metric shift count for regime change classification
    'regime_change_shift_count': 2,
    'metric_shift_threshold': 0.3,
}


# =============================================================================
# Event Classification Labels
# =============================================================================

EVENT_TYPES = {
    'decoupling': 'Signals that were coupled became independent',
    'coupling_emergence': 'Signals that were independent became coupled',
    'phase_shift': 'Phase coherence changed sign or broke',
    'volatility_cluster': 'Structural variance spiked significantly',
    'regime_change': 'Multiple metrics shifted simultaneously',
    'relationship_inversion': 'Correlation sign flipped',
    'unclassified': 'Structural change not matching known patterns',
}


# =============================================================================
# Dataclass Configuration
# =============================================================================

@dataclass
class CascadeConfig:
    """
    Configuration for cascade geometry computation.
    """
    # Window sizes for each pass
    pass_1_window: int
    pass_2_window: int
    pass_3_window: int

    # Data frequency
    frequency: str = 'daily'

    # Flag thresholds (use defaults if not specified)
    thresholds: Dict[str, float] = field(default_factory=lambda: FLAG_THRESHOLDS.copy())

    @classmethod
    def for_daily(cls) -> 'CascadeConfig':
        """Configuration for daily data."""
        cascade = WINDOW_CASCADE.get('daily')
        if not cascade:
            raise RuntimeError("No 'daily' cascade configured in config/stride.yaml")
        return cls(
            pass_1_window=cascade['pass_1'],
            pass_2_window=cascade['pass_2'],
            pass_3_window=cascade['pass_3'],
            frequency='daily',
        )

    @classmethod
    def for_annual(cls) -> 'CascadeConfig':
        """Configuration for annual data."""
        cascade = WINDOW_CASCADE.get('annual')
        if not cascade:
            raise RuntimeError("No 'annual' cascade configured in config/stride.yaml")
        return cls(
            pass_1_window=cascade['pass_1'],
            pass_2_window=cascade['pass_2'],
            pass_3_window=cascade['pass_3'],
            frequency='annual',
        )

    @classmethod
    def for_monthly(cls) -> 'CascadeConfig':
        """Configuration for monthly data (climate)."""
        cascade = WINDOW_CASCADE.get('monthly')
        if not cascade:
            raise RuntimeError("No 'monthly' cascade configured in config/stride.yaml")
        return cls(
            pass_1_window=cascade['pass_1'],
            pass_2_window=cascade['pass_2'],
            pass_3_window=cascade['pass_3'],
            frequency='monthly',
        )

    @property
    def windows(self) -> Dict[str, int]:
        """Get windows as dict."""
        return {
            'pass_1': self.pass_1_window,
            'pass_2': self.pass_2_window,
            'pass_3': self.pass_3_window,
        }


# =============================================================================
# Domain Configuration Mapping (lazy-loaded)
# =============================================================================

_DOMAIN_CASCADE_CONFIGS = None


def get_domain_cascade_configs() -> dict:
    """Get domain cascade configs. Loads from config on first access."""
    global _DOMAIN_CASCADE_CONFIGS
    if _DOMAIN_CASCADE_CONFIGS is None:
        _DOMAIN_CASCADE_CONFIGS = {
            # Regional domains (annual data)
            'world': CascadeConfig.for_annual(),
            'G7': CascadeConfig.for_annual(),
            'NORDIC': CascadeConfig.for_annual(),
            'BRICS': CascadeConfig.for_annual(),
            'EU_CORE': CascadeConfig.for_annual(),
            'ASEAN': CascadeConfig.for_annual(),
            'ASIA_TIGERS': CascadeConfig.for_annual(),
            'LATAM': CascadeConfig.for_annual(),
            'MENA': CascadeConfig.for_annual(),
            'SSA': CascadeConfig.for_annual(),
            'OCEANIA': CascadeConfig.for_annual(),

            # Climate domain (monthly data)
            'climate': CascadeConfig.for_monthly(),
        }
    return _DOMAIN_CASCADE_CONFIGS




# =============================================================================
# API Functions
# =============================================================================

def get_cascade_config(conn=None, frequency: str = 'daily', domain: str = None) -> Dict[str, int]:
    """
    Load cascade config from domain_config table or use defaults.

    Priority:
    1. Database raw.domain_config if conn provided
    2. Domain-specific config if domain provided
    3. Frequency-based defaults

    Args:
        conn: Optional database connection for database lookup
        frequency: Data frequency ('daily', 'annual', 'monthly')
        domain: Optional domain name for domain-specific config

    Returns:
        Dict with pass_1, pass_2, pass_3 window sizes
    """
    # Try database config first
    if conn is not None:
        try:
            result = conn.execute("""
                SELECT value FROM raw.domain_config
                WHERE key = 'cascade_windows'
            """).fetchone()
            if result:
                return json.loads(result[0])
        except Exception:
            pass

    # Try domain-specific config
    domain_configs = get_domain_cascade_configs()
    if domain and domain in domain_configs:
        return domain_configs[domain].windows

    # Fall back to frequency defaults
    if frequency in WINDOW_CASCADE:
        return WINDOW_CASCADE[frequency]

    # No fallback - must be configured
    raise RuntimeError(
        f"No cascade config found for frequency '{frequency}'. "
        "Configure domain-specific windows in config/stride.yaml."
    )


def get_cascade_config_for_domain(domain: str) -> CascadeConfig:
    """
    Get full cascade configuration for a domain.

    Args:
        domain: Domain name

    Returns:
        CascadeConfig instance

    Raises:
        KeyError: If domain not configured
    """
    domain_configs = get_domain_cascade_configs()
    if domain not in domain_configs:
        available = ', '.join(sorted(domain_configs.keys()))
        raise KeyError(
            f"No cascade config for domain '{domain}'. "
            f"Available domains: {available}. "
            f"Configure domain in config/stride.yaml"
        )
    return domain_configs[domain]


def get_thresholds(conn=None) -> Dict[str, float]:
    """
    Get flag thresholds from database or defaults.

    Args:
        conn: Optional database connection for database lookup

    Returns:
        Dict of threshold name -> value
    """
    if conn is not None:
        try:
            result = conn.execute("""
                SELECT value FROM raw.domain_config
                WHERE key = 'cascade_thresholds'
            """).fetchone()
            if result:
                return json.loads(result[0])
        except Exception:
            pass

    return FLAG_THRESHOLDS.copy()
