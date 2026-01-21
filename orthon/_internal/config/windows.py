"""
PRISM Window Configuration
==========================
Shared window parameters for all geometry computations.

Ensures Method A (cohort_geometry), Method B (system_window), and Residual
all use identical windows - eliminating alignment gaps.

FAST LOOKUP DICTS:
- WINDOW_WEIGHTS: Weights for aggregation across window tiers
- MIN_OBSERVATIONS: Minimum observations per window tier
- ENGINE_MIN_OBS: Minimum observations per vector engine
"""

from dataclasses import dataclass, field
from typing import Literal, List, Dict
import hashlib
import json


# =============================================================================
# FAST LOOKUP DICTS (loaded from config/stride.yaml)
# =============================================================================

def _load_window_weights() -> Dict[int, float]:
    """Load window weights from stride.yaml. Fails if not configured."""
    try:
        from orthon._internal.utils.stride import get_barycenter_weights
        weights = get_barycenter_weights()
        if weights:
            return weights
    except Exception as e:
        raise RuntimeError(f"Failed to load window weights: {e}")

    raise RuntimeError(
        "No window weights configured in config/stride.yaml. "
        "Configure domain-specific window weights before running."
    )


def _load_min_observations() -> Dict[int, int]:
    """Load minimum observations from stride.yaml. Fails if not configured."""
    try:
        from orthon._internal.utils.stride import load_stride_config
        config = load_stride_config()
        if hasattr(config, 'windows') and config.windows:
            return {
                w.window_days: getattr(w, 'min_observations', max(15, w.window_days // 5))
                for w in config.windows.values()
            }
    except Exception as e:
        raise RuntimeError(f"Failed to load min observations: {e}")

    raise RuntimeError(
        "No window tiers configured in config/stride.yaml. "
        "Configure domain-specific windows before running."
    )


# Window weights for geometry/state aggregation
# Loaded from config/stride.yaml - fails if not configured
WINDOW_WEIGHTS: Dict[int, float] = _load_window_weights()

# Minimum observations required per window tier
# Loaded from config/stride.yaml - fails if not configured
MIN_OBSERVATIONS: Dict[int, int] = _load_min_observations()

# Engine minimum observations (vector layer)
ENGINE_MIN_OBS: Dict[str, int] = {
    'hurst': 20,
    'entropy': 30,
    'lyapunov': 30,
    'garch': 50,
    'spectral': 40,
    'wavelet': 40,
    'rqa': 30,
    'realized_vol': 15,
}


def get_window_weight(window_days: int) -> float:
    """Get weight for a window size. Returns 0.5 if unknown."""
    return WINDOW_WEIGHTS.get(window_days, 0.5)


def get_min_obs(window_days: int) -> int:
    """Get minimum observations for a window size."""
    return MIN_OBSERVATIONS.get(window_days, 15)


# =============================================================================
# DOMAIN WINDOW CONFIGS
# =============================================================================


@dataclass
class DomainWindowConfig:
    """
    Domain-specific window configuration for signal_geometry.

    Defines the window sizes to compute for each domain based on
    data frequency (daily vs monthly vs annual).
    """
    # Window sizes in observations (loaded from config)
    window_sizes: List[int] = field(default_factory=lambda: list(WINDOW_WEIGHTS.keys()))

    # Step size for rolling windows (loaded from config)
    step_size: int = 5

    # Data frequency for labeling
    frequency: Literal['daily', 'monthly', 'annual'] = 'daily'

    # Minimum observations per signal per window
    min_observations_ratio: float = 0.5  # min_obs = window_size * ratio

    @classmethod
    def from_config(cls) -> 'DomainWindowConfig':
        """Load from config/stride.yaml."""
        try:
            from orthon._internal.utils.stride import load_stride_config
            config = load_stride_config()
            if hasattr(config, 'windows') and config.windows:
                window_sizes = sorted([w.window_days for w in config.windows.values()])
                # Get step from smallest window stride
                smallest = min(config.windows.values(), key=lambda w: w.window_days)
                step = getattr(smallest, 'stride_days', 5)
                return cls(
                    window_sizes=window_sizes,
                    step_size=step,
                    frequency='daily',  # Default, can be overridden
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load window config: {e}")

        raise RuntimeError(
            "No windows configured in config/stride.yaml. "
            "Configure domain-specific windows before running."
        )

    @classmethod
    def for_daily(cls) -> 'DomainWindowConfig':
        """Load daily config from stride.yaml."""
        return cls.from_config()

    @classmethod
    def for_annual(cls) -> 'DomainWindowConfig':
        """Load annual config from stride.yaml."""
        config = cls.from_config()
        config.frequency = 'annual'
        config.min_observations_ratio = 0.6
        return config

    @classmethod
    def for_monthly(cls) -> 'DomainWindowConfig':
        """Load monthly config from stride.yaml."""
        config = cls.from_config()
        config.frequency = 'monthly'
        return config


# Domain -> Window Configuration mapping
# Runner MUST find domain here or fail
DOMAIN_WINDOW_CONFIGS = {
    # Regional domains (annual data)
    'world': DomainWindowConfig.for_annual(),
    'G7': DomainWindowConfig.for_annual(),
    'NORDIC': DomainWindowConfig.for_annual(),
    'BRICS': DomainWindowConfig.for_annual(),
    'EU_CORE': DomainWindowConfig.for_annual(),
    'ASEAN': DomainWindowConfig.for_annual(),
    'ASIA_TIGERS': DomainWindowConfig.for_annual(),
    'LATAM': DomainWindowConfig.for_annual(),
    'MENA': DomainWindowConfig.for_annual(),
    'SSA': DomainWindowConfig.for_annual(),
    'OCEANIA': DomainWindowConfig.for_annual(),

    # Climate domain (monthly data)
    'climate': DomainWindowConfig.for_monthly(),
}


def get_window_config_for_domain(domain: str) -> DomainWindowConfig:
    """
    Get window configuration for a domain.

    Raises KeyError if domain not configured - this is intentional
    to prevent running with wrong window sizes.
    """
    if domain not in DOMAIN_WINDOW_CONFIGS:
        available = ', '.join(sorted(DOMAIN_WINDOW_CONFIGS.keys()))
        raise KeyError(
            f"No window config for domain '{domain}'. "
            f"Available domains: {available}. "
            f"Add config to DOMAIN_WINDOW_CONFIGS in prism/config/windows.py"
        )
    return DOMAIN_WINDOW_CONFIGS[domain]


def _get_default_window_size() -> int:
    """Get default window size from config."""
    try:
        from orthon._internal.utils.stride import load_stride_config
        config = load_stride_config()
        if hasattr(config, 'windows') and config.windows:
            # Return largest window as default
            return max(w.window_days for w in config.windows.values())
    except Exception:
        pass
    raise RuntimeError("No window size configured in config/stride.yaml")


def _get_default_step_size() -> int:
    """Get default step size from config."""
    try:
        from orthon._internal.utils.stride import load_stride_config
        config = load_stride_config()
        if hasattr(config, 'windows') and config.windows:
            # Return stride from largest window
            largest = max(config.windows.values(), key=lambda w: w.window_days)
            return largest.stride_days
    except Exception:
        pass
    raise RuntimeError("No stride configured in config/stride.yaml")


@dataclass
class WindowConfig:
    """
    Shared window configuration for all geometry computations.

    Both Method A and Method B use these parameters, ensuring
    identical window_end dates for residual analysis.
    """

    # Window size in observations (must be configured)
    window_size: int = field(default_factory=_get_default_window_size)

    # Step size (how often to compute a new window)
    step_size: int = field(default_factory=_get_default_step_size)

    # Window end convention
    end_convention: Literal['end_of_month', 'mid_month', 'exact'] = 'end_of_month'

    # Minimum observations required per signal
    min_observations: int = 100

    # Minimum signals required per window
    min_signals: int = 3

    @classmethod
    def from_config(cls, tier: str = 'anchor') -> 'WindowConfig':
        """Load from config/stride.yaml for a specific tier."""
        try:
            from orthon._internal.utils.stride import load_stride_config
            config = load_stride_config()
            if hasattr(config, 'windows') and tier in config.windows:
                w = config.windows[tier]
                return cls(
                    window_size=w.window_days,
                    step_size=w.stride_days,
                    min_observations=getattr(w, 'min_observations', 100),
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load window config for tier '{tier}': {e}")

        raise RuntimeError(f"No window config found for tier '{tier}' in config/stride.yaml")

    @classmethod
    def monthly(cls) -> 'WindowConfig':
        """Load anchor tier config with monthly stepping."""
        config = cls.from_config('anchor')
        config.end_convention = 'end_of_month'
        return config

    @classmethod
    def weekly(cls) -> 'WindowConfig':
        """Load anchor tier config with weekly stepping."""
        config = cls.from_config('anchor')
        config.step_size = max(1, config.window_size // 50)  # ~weekly
        config.end_convention = 'exact'
        return config

    @classmethod
    def quarterly(cls) -> 'WindowConfig':
        """Load anchor tier config with quarterly stepping."""
        config = cls.from_config('anchor')
        config.step_size = max(1, config.window_size // 4)  # quarterly
        config.end_convention = 'end_of_month'
        return config

    @classmethod
    def for_climate(cls) -> 'WindowConfig':
        """
        Climate data uses monthly observations, not daily.
        Loads from config/stride.yaml or fails if not configured.
        """
        try:
            config = cls.from_config('anchor')
            config.end_convention = 'mid_month'  # Climate data uses 15th
            return config
        except Exception:
            raise RuntimeError(
                "No climate window configuration found. "
                "Configure climate domain windows in config/stride.yaml."
            )

    @property
    def config_hash(self) -> str:
        """Generate hash for this config (for tracking in DB)."""
        config_dict = {
            'window_size': self.window_size,
            'step_size': self.step_size,
            'end_convention': self.end_convention,
            'min_observations': self.min_observations,
            'min_signals': self.min_signals,
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def __repr__(self) -> str:
        return (
            f"WindowConfig(size={self.window_size}, step={self.step_size}, "
            f"end={self.end_convention}, hash={self.config_hash})"
        )


# Lazy-loaded configs (to avoid import-time failures if stride.yaml not configured)
_DEFAULT_WINDOW_CONFIG = None
_DOMAIN_CONFIGS = None


def get_default_window_config() -> WindowConfig:
    """Get default window config. Loads from config on first access."""
    global _DEFAULT_WINDOW_CONFIG
    if _DEFAULT_WINDOW_CONFIG is None:
        _DEFAULT_WINDOW_CONFIG = WindowConfig.monthly()
    return _DEFAULT_WINDOW_CONFIG


def get_domain_configs() -> dict:
    """Get domain configs dict. Loads from config on first access."""
    global _DOMAIN_CONFIGS
    if _DOMAIN_CONFIGS is None:
        _DOMAIN_CONFIGS = {
            'macro': WindowConfig.quarterly(),
            'climate': WindowConfig.for_climate(),
        }
    return _DOMAIN_CONFIGS


def get_config_for_domain(domain: str) -> WindowConfig:
    """Get appropriate window config for a domain."""
    configs = get_domain_configs()
    if domain in configs:
        return configs[domain]
    return get_default_window_config()


# Backward-compatible exports
# These call the getter functions to provide lazy-loaded values
# Note: These are evaluated at import time, but the WindowConfig methods handle
# lazy loading internally so stride.yaml doesn't need to exist at import time
try:
    DEFAULT_WINDOW_CONFIG = get_default_window_config()
    DOMAIN_CONFIGS = get_domain_configs()
except Exception:
    # If config loading fails at import time, set to None
    # Callers should use the getter functions for safer access
    DEFAULT_WINDOW_CONFIG = None
    DOMAIN_CONFIGS = None
