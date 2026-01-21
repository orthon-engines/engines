"""PRISM Configuration Module."""

from orthon._internal.config.windows import (
    WindowConfig,
    DEFAULT_WINDOW_CONFIG,
    DOMAIN_CONFIGS,
    get_config_for_domain,
)

from orthon._internal.config.domain import (
    DomainConfig,
    TemporalUnit,
    FrequencyConfig,
    ValidationConfig,
    get_domain_config,
    load_domain_config,
    list_available_domains,
    get_window_days,
    get_resample_frequency,
    get_validation_thresholds,
    clear_config_cache,
    reload_domain_config,
)

from orthon._internal.config.cascade import (
    CascadeConfig,
    WINDOW_CASCADE,
    FLAG_THRESHOLDS,
    EVENT_TYPES,
    get_domain_cascade_configs,
    get_cascade_config,
    get_cascade_config_for_domain,
    get_thresholds,
)

__all__ = [
    # Legacy window config
    'WindowConfig',
    'DEFAULT_WINDOW_CONFIG',
    'DOMAIN_CONFIGS',
    'get_config_for_domain',
    # Domain configuration (YAML-based)
    'DomainConfig',
    'TemporalUnit',
    'FrequencyConfig',
    'ValidationConfig',
    'get_domain_config',
    'load_domain_config',
    'list_available_domains',
    'get_window_days',
    'get_resample_frequency',
    'get_validation_thresholds',
    'clear_config_cache',
    'reload_domain_config',
    # Cascade geometry configuration
    'CascadeConfig',
    'WINDOW_CASCADE',
    'FLAG_THRESHOLDS',
    'EVENT_TYPES',
    'get_domain_cascade_configs',
    'get_cascade_config',
    'get_cascade_config_for_domain',
    'get_thresholds',
]
