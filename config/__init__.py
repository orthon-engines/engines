"""
PRISM Configuration Module

Provides hierarchical configuration with defaults and domain/environment overrides.

Usage:
    from config import ConfigManager, get_config

    # Get global config (loads defaults)
    config = get_config()

    # Get config for specific domain
    config = get_config(domain='turbofan')

    # Access values
    min_signals = config.get('geometry.eigenvalue.min_signals')
    feature_groups = config.get('geometry.feature_groups')
"""

from .config_manager import ConfigManager, get_config

__all__ = ['ConfigManager', 'get_config']
