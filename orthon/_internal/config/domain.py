"""
Domain configuration loader.

Reads YAML files to configure temporal characteristics per domain.
PRISM adapts windowing, resampling, and validation accordingly.

Usage:
    from orthon._internal.config.domain import get_domain_config

    config = get_domain_config('cmapss')
    window_days = config.default_window.to_base_units('days')
    resample_freq = config.get_resample_freq('weekly')
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import timedelta
import yaml


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TemporalUnit:
    """A unit of time measurement."""
    name: str
    seconds: float

    def to_timedelta(self) -> timedelta:
        return timedelta(seconds=self.seconds)

    def __repr__(self):
        return f"TemporalUnit({self.name}, {self.seconds}s)"


@dataclass
class FrequencyConfig:
    """Configuration for a data frequency."""
    name: str
    resolution: str           # e.g., "1d", "1w", "1ps"
    resample_to: Optional[str] = None  # Pandas freq string
    default: bool = False
    rare: bool = False

    def __repr__(self):
        default_str = " [default]" if self.default else ""
        return f"FrequencyConfig({self.name}, {self.resolution}{default_str})"


@dataclass
class WindowConfig:
    """Configuration for analysis windows."""
    name: str
    label: str

    # Duration in various units (only one should be set)
    days: Optional[int] = None
    years: Optional[int] = None
    nanoseconds: Optional[int] = None
    microseconds: Optional[int] = None
    milliseconds: Optional[int] = None
    seconds: Optional[int] = None
    minutes: Optional[int] = None
    hours: Optional[int] = None

    def to_base_units(self, base_unit: str = 'days') -> float:
        """
        Convert window to base units.

        Args:
            base_unit: Target unit ('days', 'seconds', 'milliseconds')

        Returns:
            Duration in base units
        """
        # Convert everything to seconds first
        total_seconds = 0

        if self.nanoseconds is not None:
            total_seconds += self.nanoseconds * 1e-9
        if self.microseconds is not None:
            total_seconds += self.microseconds * 1e-6
        if self.milliseconds is not None:
            total_seconds += self.milliseconds * 1e-3
        if self.seconds is not None:
            total_seconds += self.seconds
        if self.minutes is not None:
            total_seconds += self.minutes * 60
        if self.hours is not None:
            total_seconds += self.hours * 3600
        if self.days is not None:
            total_seconds += self.days * 86400
        if self.years is not None:
            total_seconds += self.years * 365.25 * 86400

        # Convert to target unit
        if base_unit == 'seconds':
            return total_seconds
        elif base_unit == 'milliseconds':
            return total_seconds * 1000
        elif base_unit == 'microseconds':
            return total_seconds * 1e6
        elif base_unit == 'nanoseconds':
            return total_seconds * 1e9
        elif base_unit == 'days':
            return total_seconds / 86400
        elif base_unit == 'years':
            return total_seconds / (365.25 * 86400)
        else:
            return total_seconds  # Default to seconds

    def to_timedelta(self) -> timedelta:
        """Convert window to timedelta."""
        return timedelta(seconds=self.to_base_units('seconds'))

    def __repr__(self):
        return f"WindowConfig({self.name}, {self.label})"


@dataclass
class ValidationConfig:
    """Validation thresholds for the domain."""
    min_observations_per_window: int = 20
    coverage_threshold: float = 0.50

    # Max gap in various units (only one should be set)
    max_gap_days: Optional[int] = None
    max_gap_years: Optional[int] = None
    max_gap_hours: Optional[int] = None
    max_gap_minutes: Optional[int] = None
    max_gap_seconds: Optional[int] = None
    max_gap_milliseconds: Optional[int] = None
    max_gap_microseconds: Optional[int] = None
    max_gap_nanoseconds: Optional[int] = None

    def get_max_gap_seconds(self) -> float:
        """Get maximum allowed gap in seconds."""
        if self.max_gap_nanoseconds is not None:
            return self.max_gap_nanoseconds * 1e-9
        if self.max_gap_microseconds is not None:
            return self.max_gap_microseconds * 1e-6
        if self.max_gap_milliseconds is not None:
            return self.max_gap_milliseconds * 1e-3
        if self.max_gap_seconds is not None:
            return self.max_gap_seconds
        if self.max_gap_minutes is not None:
            return self.max_gap_minutes * 60
        if self.max_gap_hours is not None:
            return self.max_gap_hours * 3600
        if self.max_gap_days is not None:
            return self.max_gap_days * 86400
        if self.max_gap_years is not None:
            return self.max_gap_years * 365.25 * 86400
        return 30 * 86400  # Default: 30 days

    def get_max_gap_timedelta(self) -> timedelta:
        """Get maximum allowed gap as timedelta."""
        return timedelta(seconds=self.get_max_gap_seconds())


@dataclass
class DomainConfig:
    """Complete domain configuration."""
    domain: str
    description: str

    # Temporal settings
    min_resolution: str       # e.g., "1d", "1ps"
    max_resolution: str       # e.g., "1y", "1000000y"
    business_calendar: Optional[str] = None
    week_end: Optional[str] = None

    # Parsed configs
    units: Dict[str, TemporalUnit] = field(default_factory=dict)
    frequencies: Dict[str, FrequencyConfig] = field(default_factory=dict)
    windows: Dict[str, WindowConfig] = field(default_factory=dict)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    # Domain-specific extras (e.g., EEG frequency bands)
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def default_frequency(self) -> Optional[FrequencyConfig]:
        """Get the default frequency for this domain."""
        for freq in self.frequencies.values():
            if freq.default:
                return freq
        # Return first if no default specified
        if self.frequencies:
            return list(self.frequencies.values())[0]
        return None

    @property
    def default_window(self) -> Optional[WindowConfig]:
        """Get the default window for this domain."""
        return self.windows.get('default')

    @property
    def has_business_calendar(self) -> bool:
        """Check if domain uses business calendar."""
        return self.business_calendar is not None

    def get_window(self, name: str) -> Optional[WindowConfig]:
        """Get a specific window configuration."""
        return self.windows.get(name)

    def get_frequency(self, name: str) -> Optional[FrequencyConfig]:
        """Get a specific frequency configuration."""
        return self.frequencies.get(name)

    def get_resample_freq(self, frequency_name: str) -> str:
        """
        Get pandas resample string for a frequency.

        Args:
            frequency_name: Name of the frequency (e.g., 'weekly')

        Returns:
            Pandas frequency string (e.g., 'W-FRI')
        """
        freq = self.frequencies.get(frequency_name)
        if freq and freq.resample_to:
            return freq.resample_to

        # Default mappings
        defaults = {
            'daily': 'D',
            'weekly': 'W',
            'monthly': 'MS',
            'quarterly': 'QS',
            'annual': 'YS',
            'yearly': 'YS',
            'hourly': 'h',
            'minute': 'min',
            'second': 's',
            'millisecond': 'ms',
            'microsecond': 'us',
            'nanosecond': 'ns',
        }
        return defaults.get(frequency_name, 'D')

    def get_max_gap(self) -> timedelta:
        """Get maximum allowed gap as timedelta."""
        return self.validation.get_max_gap_timedelta()

    def get_base_unit(self) -> str:
        """
        Get the natural base unit for this domain.

        Returns unit name based on min_resolution.
        """
        res = self.min_resolution.lower()

        if 'ps' in res or 'pico' in res:
            return 'nanoseconds'  # timedelta doesn't support picoseconds
        elif 'ns' in res or 'nano' in res:
            return 'nanoseconds'
        elif 'us' in res or 'micro' in res:
            return 'microseconds'
        elif 'ms' in res or 'milli' in res:
            return 'milliseconds'
        elif 's' in res and 'm' not in res:
            return 'seconds'
        elif 'm' in res and 'mo' not in res:
            return 'minutes'
        elif 'h' in res:
            return 'hours'
        elif 'd' in res:
            return 'days'
        elif 'y' in res:
            return 'years'
        else:
            return 'days'  # Default

    def __repr__(self):
        return (f"DomainConfig({self.domain}, "
                f"resolution={self.min_resolution}-{self.max_resolution}, "
                f"windows={list(self.windows.keys())})")


# =============================================================================
# LOADER FUNCTIONS
# =============================================================================

def _parse_window_config(name: str, info: Dict) -> WindowConfig:
    """Parse a window configuration from YAML."""
    return WindowConfig(
        name=name,
        label=info.get('label', name),
        days=info.get('days'),
        years=info.get('years'),
        hours=info.get('hours'),
        minutes=info.get('minutes'),
        seconds=info.get('seconds'),
        milliseconds=info.get('milliseconds'),
        microseconds=info.get('microseconds'),
        nanoseconds=info.get('nanoseconds'),
    )


def _parse_validation_config(raw: Dict) -> ValidationConfig:
    """Parse validation configuration from YAML."""
    return ValidationConfig(
        min_observations_per_window=raw.get('min_observations_per_window', 20),
        coverage_threshold=raw.get('coverage_threshold', 0.50),
        max_gap_days=raw.get('max_gap_days'),
        max_gap_years=raw.get('max_gap_years'),
        max_gap_hours=raw.get('max_gap_hours'),
        max_gap_minutes=raw.get('max_gap_minutes'),
        max_gap_seconds=raw.get('max_gap_seconds'),
        max_gap_milliseconds=raw.get('max_gap_milliseconds'),
        max_gap_microseconds=raw.get('max_gap_microseconds'),
        max_gap_nanoseconds=raw.get('max_gap_nanoseconds'),
    )


def load_domain_config(domain: str, config_dir: Path = None) -> DomainConfig:
    """
    Load domain configuration from YAML file.

    Args:
        domain: Domain name (e.g., 'cmapss', 'climate')
        config_dir: Directory containing YAML files (default: prism/domains/)

    Returns:
        DomainConfig object

    Raises:
        FileNotFoundError: If domain config file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    if config_dir is None:
        # Look in prism/domains/ relative to this file
        config_dir = Path(__file__).parent.parent / 'domains'

    yaml_path = config_dir / f"{domain}.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Domain config not found: {yaml_path}\n"
            f"Available domains: {list_available_domains(config_dir)}"
        )

    with open(yaml_path, 'r') as f:
        raw = yaml.safe_load(f)

    # Parse temporal section
    temporal = raw.get('temporal', {})

    # Parse units
    units = {}
    for name, info in temporal.get('units', {}).items():
        units[name] = TemporalUnit(name=name, seconds=info['seconds'])

    # Parse frequencies
    frequencies = {}
    for name, info in temporal.get('frequencies', {}).items():
        frequencies[name] = FrequencyConfig(
            name=name,
            resolution=info.get('resolution', name),
            resample_to=info.get('resample_to'),
            default=info.get('default', False),
            rare=info.get('rare', False),
        )

    # Parse windows
    windows = {}
    for name, info in raw.get('windows', {}).items():
        windows[name] = _parse_window_config(name, info)

    # Parse validation
    validation = _parse_validation_config(raw.get('validation', {}))

    # Collect extras (domain-specific fields)
    known_keys = {'domain', 'description', 'temporal', 'windows', 'validation'}
    extras = {k: v for k, v in raw.items() if k not in known_keys}

    return DomainConfig(
        domain=raw['domain'],
        description=raw.get('description', ''),
        min_resolution=temporal.get('min_resolution', '1d'),
        max_resolution=temporal.get('max_resolution', '1y'),
        business_calendar=temporal.get('business_calendar'),
        week_end=temporal.get('week_end'),
        units=units,
        frequencies=frequencies,
        windows=windows,
        validation=validation,
        extras=extras,
    )


def list_available_domains(config_dir: Path = None) -> List[str]:
    """
    List all available domain configurations.

    Returns:
        List of domain names
    """
    if config_dir is None:
        config_dir = Path(__file__).parent.parent / 'domains'

    if not config_dir.exists():
        return []

    return [
        f.stem for f in config_dir.glob('*.yaml')
        if not f.name.startswith('_')
    ]


# =============================================================================
# CACHING
# =============================================================================

_config_cache: Dict[str, DomainConfig] = {}


def get_domain_config(domain: str, use_cache: bool = True) -> DomainConfig:
    """
    Get domain configuration (cached by default).

    Args:
        domain: Domain name
        use_cache: Whether to use cached config

    Returns:
        DomainConfig object
    """
    if use_cache and domain in _config_cache:
        return _config_cache[domain]

    config = load_domain_config(domain)

    if use_cache:
        _config_cache[domain] = config

    return config


def clear_config_cache():
    """Clear the configuration cache."""
    _config_cache.clear()


def reload_domain_config(domain: str) -> DomainConfig:
    """
    Force reload a domain configuration.

    Args:
        domain: Domain name

    Returns:
        Fresh DomainConfig object
    """
    if domain in _config_cache:
        del _config_cache[domain]
    return get_domain_config(domain)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_window_days(domain: str, window_name: str = 'default') -> int:
    """
    Get window size in days for a domain.

    Args:
        domain: Domain name
        window_name: Window name (default: 'default')

    Returns:
        Window size in days
    """
    config = get_domain_config(domain)
    window = config.get_window(window_name)
    if window is None:
        window = config.default_window
    if window is None:
        raise RuntimeError(
            f"No window configured for domain '{domain}'. "
            "Configure domain-specific windows in config/domains.yaml or config/stride.yaml."
        )
    return int(window.to_base_units('days'))


def get_resample_frequency(domain: str, frequency_name: str) -> str:
    """
    Get pandas resample string for a domain and frequency.

    Args:
        domain: Domain name
        frequency_name: Frequency name (e.g., 'weekly')

    Returns:
        Pandas frequency string
    """
    config = get_domain_config(domain)
    return config.get_resample_freq(frequency_name)


def get_validation_thresholds(domain: str) -> Dict[str, Any]:
    """
    Get validation thresholds for a domain.

    Returns:
        Dict with min_observations, max_gap, coverage_threshold
    """
    config = get_domain_config(domain)
    v = config.validation

    return {
        'min_observations': v.min_observations_per_window,
        'max_gap_seconds': v.get_max_gap_seconds(),
        'max_gap_timedelta': v.get_max_gap_timedelta(),
        'coverage_threshold': v.coverage_threshold,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for inspecting domain configurations."""
    import argparse

    parser = argparse.ArgumentParser(description='PRISM Domain Configuration')
    parser.add_argument('domain', nargs='?', help='Domain to inspect')
    parser.add_argument('--list', action='store_true', help='List available domains')
    parser.add_argument('--validate', action='store_true', help='Validate config')

    args = parser.parse_args()

    if args.list or not args.domain:
        domains = list_available_domains()
        print("Available domains:")
        for d in domains:
            try:
                config = get_domain_config(d)
                print(f"  {d}: {config.description}")
            except Exception as e:
                print(f"  {d}: [ERROR] {e}")
        return

    try:
        config = get_domain_config(args.domain)

        print(f"Domain: {config.domain}")
        print(f"Description: {config.description}")
        print()
        print("Temporal:")
        print(f"  Resolution: {config.min_resolution} to {config.max_resolution}")
        print(f"  Business calendar: {config.business_calendar or 'None'}")
        print(f"  Week end: {config.week_end or 'None'}")
        print()
        print("Frequencies:")
        for name, freq in config.frequencies.items():
            default = " [default]" if freq.default else ""
            resample = f" -> {freq.resample_to}" if freq.resample_to else ""
            print(f"  {name}: {freq.resolution}{resample}{default}")
        print()
        print("Windows:")
        for name, window in config.windows.items():
            base = config.get_base_unit()
            size = window.to_base_units(base)
            print(f"  {name}: {size:.1f} {base} ({window.label})")
        print()
        print("Validation:")
        print(f"  Min observations: {config.validation.min_observations_per_window}")
        print(f"  Max gap: {config.validation.get_max_gap_timedelta()}")
        print(f"  Coverage threshold: {config.validation.coverage_threshold}")

        if config.extras:
            print()
            print("Extras:")
            for key, value in config.extras.items():
                print(f"  {key}: {value}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == '__main__':
    main()
