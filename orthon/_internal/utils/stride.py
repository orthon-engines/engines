"""
PRISM Stride Configuration Loader
=================================

Loads window/stride configuration from config/stride.yaml.
Applies system-wide across Vector, Geometry, and State layers.

"When five year olds run around - normal noise. When adults start running - regime change."

Usage:
    from orthon._internal.utils.stride import load_stride_config, get_window_dates, WINDOWS

    # Get all window tiers
    config = load_stride_config()

    # Generate dates for a specific window tier
    dates = get_window_dates('anchor', start_date, end_date)

    # Access window config directly
    anchor = WINDOWS['anchor']
    print(anchor.window_days)  # 252
    print(anchor.stride_days)  # 21
"""

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class WindowConfig:
    """Configuration for a single window tier."""
    name: str
    window_days: int
    stride_days: int
    min_observations: int
    weight: float
    description: str
    always_run: bool = True  # If False, only run on drill-down

    def __repr__(self) -> str:
        run_mode = "always" if self.always_run else "drill-down"
        return f"WindowConfig({self.name}: {self.window_days}d/{self.stride_days}d stride, weight={self.weight}, {run_mode})"


@dataclass
class BisectionConfig:
    """Configuration for bisection drill-down."""
    enabled: bool
    min_window_days: int
    thresholds: Dict[str, float]


@dataclass
class StrideConfig:
    """Complete stride configuration."""
    windows: Dict[str, WindowConfig]
    bisection: BisectionConfig
    profiles: Dict[str, Dict[str, Any]]
    default_tiers: List[str]  # Tiers that always run (e.g., ['anchor', 'bridge'])
    drilldown_tiers: List[str]  # Tiers that only run when bisection flags (e.g., ['scout', 'micro'])

    def get_window(self, name: str) -> WindowConfig:
        """Get a specific window tier."""
        if name not in self.windows:
            available = ', '.join(self.windows.keys())
            raise KeyError(f"Unknown window tier: {name}. Available: {available}")
        return self.windows[name]

    def list_windows(self) -> List[str]:
        """List all window tier names."""
        return list(self.windows.keys())

    def get_weights(self) -> Dict[int, float]:
        """Get weights keyed by window_days (for barycenter)."""
        return {w.window_days: w.weight for w in self.windows.values()}

    def get_default_tiers(self) -> List[str]:
        """Get tiers that should always run."""
        return self.default_tiers

    def get_drilldown_tiers(self) -> List[str]:
        """Get tiers that only run on drill-down."""
        return self.drilldown_tiers

    def get_all_tiers(self) -> List[str]:
        """Get all tiers (default + drilldown)."""
        return self.default_tiers + self.drilldown_tiers

    def get_always_run_windows(self) -> Dict[str, WindowConfig]:
        """Get only windows that always run (not drill-down only)."""
        return {k: v for k, v in self.windows.items() if v.always_run}

    def get_drilldown_windows(self) -> Dict[str, WindowConfig]:
        """Get only windows that are drill-down only."""
        return {k: v for k, v in self.windows.items() if not v.always_run}


# =============================================================================
# CONFIG LOADING
# =============================================================================

_config_cache: Optional[StrideConfig] = None


def _find_config_path() -> Path:
    """Find the stride.yaml config file."""
    # Try relative to this file
    this_dir = Path(__file__).parent
    config_path = this_dir.parent.parent / "config" / "stride.yaml"

    if config_path.exists():
        return config_path

    # Try current working directory
    cwd_path = Path.cwd() / "config" / "stride.yaml"
    if cwd_path.exists():
        return cwd_path

    raise FileNotFoundError(
        f"Could not find stride.yaml config. Tried:\n"
        f"  {config_path}\n"
        f"  {cwd_path}"
    )


def load_stride_config(profile: Optional[str] = None, force_reload: bool = False) -> StrideConfig:
    """
    Load stride configuration from YAML.

    Args:
        profile: Optional profile name to apply (e.g., 'seismology')
        force_reload: If True, reload from disk even if cached

    Returns:
        StrideConfig with all window tiers
    """
    global _config_cache

    if _config_cache is not None and not force_reload and profile is None:
        return _config_cache

    config_path = _find_config_path()
    logger.info(f"Loading stride config from {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # Parse windows
    windows = {}
    for name, cfg in raw['windows'].items():
        windows[name] = WindowConfig(
            name=name,
            window_days=cfg['window_days'],
            stride_days=cfg['stride_days'],
            min_observations=cfg['min_observations'],
            weight=cfg['weight'],
            description=cfg.get('description', ''),
            always_run=cfg.get('always_run', True),
        )

    # Parse bisection
    bisection = BisectionConfig(
        enabled=raw['bisection']['enabled'],
        min_window_days=raw['bisection']['min_window_days'],
        thresholds=raw['bisection']['thresholds'],
    )

    # Parse profiles
    profiles = raw.get('profiles', {})

    # Parse tier lists (default to all windows if not specified)
    default_tiers = raw.get('default_tiers', list(windows.keys()))
    drilldown_tiers = raw.get('drilldown_tiers', [])

    config = StrideConfig(
        windows=windows,
        bisection=bisection,
        profiles=profiles,
        default_tiers=default_tiers,
        drilldown_tiers=drilldown_tiers,
    )

    # Apply profile overrides if specified
    if profile and profile in profiles:
        config = _apply_profile(config, profiles[profile])
        logger.info(f"Applied profile: {profile}")

    # Cache if no profile (base config)
    if profile is None:
        _config_cache = config

    return config


def _apply_profile(config: StrideConfig, profile_overrides: Dict[str, Any]) -> StrideConfig:
    """Apply profile overrides to base config."""
    windows = dict(config.windows)

    for window_name, overrides in profile_overrides.items():
        if window_name in windows and overrides:
            old = windows[window_name]
            windows[window_name] = WindowConfig(
                name=old.name,
                window_days=overrides.get('window_days', old.window_days),
                stride_days=overrides.get('stride_days', old.stride_days),
                min_observations=overrides.get('min_observations', old.min_observations),
                weight=overrides.get('weight', old.weight),
                description=old.description,
                always_run=overrides.get('always_run', old.always_run),
            )

    return StrideConfig(
        windows=windows,
        bisection=config.bisection,
        profiles=config.profiles,
        default_tiers=config.default_tiers,
        drilldown_tiers=config.drilldown_tiers,
    )


# =============================================================================
# DATE GENERATION
# =============================================================================

def get_window_dates(
    window_name: str,
    start_date: date,
    end_date: date,
    config: Optional[StrideConfig] = None,
) -> List[date]:
    """
    Generate window_end dates for a specific window tier.

    Args:
        window_name: Window tier name ('anchor', 'bridge', 'scout', 'micro')
        start_date: First possible window_end date
        end_date: Last possible window_end date
        config: Optional config (loads default if not provided)

    Returns:
        List of dates at the configured stride
    """
    if config is None:
        config = load_stride_config()

    window = config.get_window(window_name)
    stride = timedelta(days=window.stride_days)

    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += stride

    return dates


def get_all_window_dates(
    start_date: date,
    end_date: date,
    config: Optional[StrideConfig] = None,
) -> Dict[str, List[date]]:
    """
    Generate window_end dates for all window tiers.

    Args:
        start_date: First possible window_end date
        end_date: Last possible window_end date
        config: Optional config (loads default if not provided)

    Returns:
        Dict mapping window_name -> list of dates
    """
    if config is None:
        config = load_stride_config()

    return {
        name: get_window_dates(name, start_date, end_date, config)
        for name in config.list_windows()
    }


# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

def get_windows() -> Dict[str, WindowConfig]:
    """Get all window configurations."""
    return load_stride_config().windows


def get_bisection_config() -> BisectionConfig:
    """Get bisection configuration."""
    return load_stride_config().bisection


def get_barycenter_weights() -> Dict[int, float]:
    """Get weights for barycenter computation (keyed by window_days)."""
    return load_stride_config().get_weights()


def get_default_tiers() -> List[str]:
    """Get tier names that should always run (e.g., ['anchor', 'bridge'])."""
    return load_stride_config().get_default_tiers()


def get_drilldown_tiers() -> List[str]:
    """Get tier names that only run on bisection drill-down (e.g., ['scout', 'micro'])."""
    return load_stride_config().get_drilldown_tiers()


def get_default_window_dates(
    start_date: date,
    end_date: date,
    config: Optional[StrideConfig] = None,
) -> Dict[str, List[date]]:
    """
    Generate window_end dates for default tiers only.

    This is the standard execution mode - only runs anchor + bridge tiers.
    Scout and micro tiers are only run when bisection flags displacement.

    Args:
        start_date: First possible window_end date
        end_date: Last possible window_end date
        config: Optional config (loads default if not provided)

    Returns:
        Dict mapping window_name -> list of dates (only for default tiers)
    """
    if config is None:
        config = load_stride_config()

    return {
        name: get_window_dates(name, start_date, end_date, config)
        for name in config.get_default_tiers()
    }


# =============================================================================
# DOMAIN-SPECIFIC WINDOWS
# =============================================================================

@dataclass
class DomainWindowConfig:
    """Domain-specific window configuration."""
    domain: str
    window: int       # Window size in samples/observations
    stride: int       # Stride in samples/observations
    min_obs: int      # Minimum observations required
    description: str


def get_domain_window_config(domain: str) -> Optional[DomainWindowConfig]:
    """
    Get domain-specific window configuration.

    Args:
        domain: Domain name (e.g., 'cheme', 'cmapss', 'climate')

    Returns:
        DomainWindowConfig if found, None otherwise
    """
    config_path = _find_config_path()
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    domain_windows = raw.get('domain_windows', {})
    if domain not in domain_windows:
        return None

    cfg = domain_windows[domain]
    return DomainWindowConfig(
        domain=domain,
        window=cfg['window'],
        stride=cfg['stride'],
        min_obs=cfg.get('min_obs', 100),
        description=cfg.get('description', ''),
    )


def get_domain_window(domain: str) -> tuple:
    """
    Get window and stride for a domain. Fails if not configured.

    Args:
        domain: Domain name

    Returns:
        (window, stride, min_obs) tuple

    Raises:
        RuntimeError: If domain not configured in config/stride.yaml
    """
    cfg = get_domain_window_config(domain)
    if cfg:
        return (cfg.window, cfg.stride, cfg.min_obs)
    raise RuntimeError(
        f"No window configuration found for domain '{domain}' in config/stride.yaml. "
        "Configure domain-specific windows before running."
    )


def list_domain_windows() -> Dict[str, DomainWindowConfig]:
    """List all configured domain window settings."""
    config_path = _find_config_path()
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    domain_windows = raw.get('domain_windows', {})
    return {
        name: DomainWindowConfig(
            domain=name,
            window=cfg['window'],
            stride=cfg['stride'],
            min_obs=cfg.get('min_obs', 100),
            description=cfg.get('description', ''),
        )
        for name, cfg in domain_windows.items()
    }


# =============================================================================
# LAZY-LOADED MODULE-LEVEL ACCESS
# =============================================================================

class _WindowsProxy:
    """Lazy proxy for WINDOWS dict."""
    def __getitem__(self, key: str) -> WindowConfig:
        return load_stride_config().get_window(key)

    def __iter__(self):
        return iter(load_stride_config().windows)

    def keys(self):
        return load_stride_config().windows.keys()

    def values(self):
        return load_stride_config().windows.values()

    def items(self):
        return load_stride_config().windows.items()


WINDOWS = _WindowsProxy()
