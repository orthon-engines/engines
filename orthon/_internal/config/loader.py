"""
PRISM Adaptive Clock Configuration Loader
==========================================

Load domain-specific configuration for adaptive windowing from domain.yaml.

This supplements the existing domain.py config with adaptive clock parameters
for auto-detecting window sizes based on data frequency.

Usage:
    from orthon._internal.config.loader import load_clock_config, load_delta_thresholds

    config = load_clock_config('cmapss')  # Adaptive clock parameters
    thresholds = load_delta_thresholds('cmapss')  # Layer transition thresholds
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


# Default config path (can be overridden)
CONFIG_PATH = Path(__file__).parent.parent.parent / 'config'


def get_config_path() -> Path:
    """Get config directory path."""
    # Try multiple locations
    candidates = [
        CONFIG_PATH,
        Path('config'),
        Path(__file__).parent / 'config',
    ]
    
    for path in candidates:
        if path.exists():
            return path
    
    return Path('config')


def load_clock_config(domain: str = None) -> Dict[str, Any]:
    """
    Load domain configuration from domain.yaml.
    
    Args:
        domain: Domain name (e.g., 'cmapss', 'femto')
                If None, returns defaults only
    
    Returns:
        Configuration dict with keys:
        - min_cycles
        - min_samples
        - max_samples
        - stride_fraction
        - convergence_threshold
        - use_spectral, use_autocorrelation, etc.
    """
    config_file = get_config_path() / 'domain.yaml'
    
    if not config_file.exists():
        print(f"[Config] No domain.yaml found at {config_file}, using defaults")
        return get_default_config()
    
    with open(config_file) as f:
        all_config = yaml.safe_load(f)
    
    defaults = all_config.get('defaults', {})
    
    if domain and domain in all_config.get('domains', {}):
        domain_config = all_config['domains'][domain]
        # Merge: domain-specific overrides defaults
        merged = {**defaults, **domain_config}
        return merged
    
    return defaults


def load_delta_thresholds(domain: str = None) -> Dict[str, float]:
    """
    Load delta thresholds for convergence/transition detection.
    
    Args:
        domain: Domain name (currently unused, thresholds are global)
    
    Returns:
        Dict with threshold values for each layer:
        - vector_convergence
        - geometry_divergence
        - geometry_coupling
        - mode_entropy
        - mode_affinity
        - state_velocity
        - state_acceleration
    """
    config_file = get_config_path() / 'domain.yaml'
    
    if not config_file.exists():
        return get_default_thresholds()
    
    with open(config_file) as f:
        all_config = yaml.safe_load(f)
    
    return all_config.get('delta_thresholds', get_default_thresholds())


def load_rul_thresholds() -> Dict[str, float]:
    """
    Load RUL correlation thresholds for ML-free prognostics.
    
    Returns:
        Dict with correlation thresholds:
        - divergence_correlation: threshold for direct divergence→RUL
        - coupling_correlation: threshold for coupling→RUL
        - entropy_correlation: threshold for entropy→RUL
    """
    config_file = get_config_path() / 'domain.yaml'
    
    if not config_file.exists():
        return {
            'divergence_correlation': 0.70,
            'coupling_correlation': 0.60,
            'entropy_correlation': 0.50,
        }
    
    with open(config_file) as f:
        all_config = yaml.safe_load(f)
    
    return all_config.get('rul_direct_thresholds', {
        'divergence_correlation': 0.70,
        'coupling_correlation': 0.60,
        'entropy_correlation': 0.50,
    })


def load_physics_thresholds() -> Dict[str, float]:
    """Load physics validation thresholds."""
    config_file = get_config_path() / 'domain.yaml'
    
    if not config_file.exists():
        return get_default_physics_thresholds()
    
    with open(config_file) as f:
        all_config = yaml.safe_load(f)
    
    return all_config.get('physics', get_default_physics_thresholds())


def get_default_config() -> Dict[str, Any]:
    """Default domain configuration."""
    return {
        'min_cycles': 3,
        'min_samples': 20,
        'max_samples': 1000,
        'stride_fraction': 0.33,
        'convergence_threshold': 0.05,
        'use_spectral': True,
        'use_autocorrelation': True,
        'use_zero_crossing': True,
        'use_activity': True,
        'laplace_n_values': 50,
        'laplace_freq_margin': 10,
    }


def get_default_thresholds() -> Dict[str, float]:
    """Default delta thresholds for all layers."""
    return {
        'vector_convergence': 0.05,
        'geometry_divergence': 0.10,
        'geometry_coupling': 0.15,
        'mode_entropy': 0.20,
        'mode_affinity': 0.10,
        'state_velocity': 0.10,
        'state_acceleration': 0.05,
    }


def get_default_physics_thresholds() -> Dict[str, float]:
    """Default physics validation thresholds."""
    return {
        'energy_conservation_tolerance': 0.05,
        'entropy_increase_tolerance': 0.01,
        'causality_lag_max': 10,
    }


def list_domains() -> list:
    """List all configured domains."""
    config_file = get_config_path() / 'domain.yaml'
    
    if not config_file.exists():
        return []
    
    with open(config_file) as f:
        all_config = yaml.safe_load(f)
    
    return list(all_config.get('domains', {}).keys())


def describe_domain(domain: str) -> str:
    """Get description for a domain."""
    config = load_clock_config(domain)
    return config.get('description', f'Domain: {domain}')


# =============================================================================
# Convenience functions for entry points
# =============================================================================

def get_window_config(domain: str) -> Dict[str, Any]:
    """
    Get window configuration for a domain.
    
    This is the main interface for entry points to get windowing parameters.
    
    Returns:
        Dict with:
        - min_cycles
        - min_samples
        - max_samples
        - stride_fraction
    """
    config = load_clock_config(domain)
    
    return {
        'min_cycles': config.get('min_cycles', 3),
        'min_samples': config.get('min_samples', 20),
        'max_samples': config.get('max_samples', 1000),
        'stride_fraction': config.get('stride_fraction', 0.33),
    }


def get_laplace_config(domain: str) -> Dict[str, Any]:
    """
    Get Laplace transform configuration for a domain.
    
    Returns:
        Dict with:
        - n_values: number of s-values
        - freq_margin: margin around frequency range
    """
    config = load_clock_config(domain)
    
    return {
        'n_values': config.get('laplace_n_values', 50),
        'freq_margin': config.get('laplace_freq_margin', 10),
    }


if __name__ == "__main__":
    # Test configuration loading
    print("Testing configuration loader...")
    print()
    
    # List domains
    domains = list_domains()
    print(f"Configured domains: {domains}")
    print()
    
    # Load config for each domain
    for domain in ['cmapss', 'femto', 'tep', 'unknown']:
        config = load_clock_config(domain)
        print(f"{domain}:")
        print(f"  min_cycles: {config.get('min_cycles')}")
        print(f"  min_samples: {config.get('min_samples')}")
        print(f"  max_samples: {config.get('max_samples')}")
        print()
    
    # Load thresholds
    thresholds = load_delta_thresholds()
    print("Delta thresholds:")
    for k, v in thresholds.items():
        print(f"  {k}: {v}")
