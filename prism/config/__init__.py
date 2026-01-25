"""PRISM Configuration Module."""

from prism.config.dataset import (
    DatasetConfig,
    normalize_columns,
    denormalize_columns,
    INTERNAL_ENTITY,
    INTERNAL_AXIS,
    INTERNAL_SIGNAL,
    INTERNAL_VALUE,
    CMAPSS_CONFIG,
    ELECTROCHEM_CONFIG,
    FEMTO_CONFIG,
)

from prism.config.windows import (
    WindowConfig,
    DEFAULT_WINDOW_CONFIG,
    DOMAIN_CONFIGS,
    get_config_for_domain,
)

from prism.config.domain import (
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

from prism.config.cascade import (
    CascadeConfig,
    WINDOW_CASCADE,
    FLAG_THRESHOLDS,
    EVENT_TYPES,
    get_domain_cascade_configs,
    get_cascade_config,
    get_cascade_config_for_domain,
    get_thresholds,
)

from prism.config.thresholds import (
    # Signal Typology
    TYPOLOGY_CLASSIFICATION,
    TYPOLOGY_AXIS_THRESHOLDS,
    TYPOLOGY_TRANSITION,
    # Structural Geometry
    GEOMETRY_TOPOLOGY,
    GEOMETRY_STABILITY,
    GEOMETRY_LEADERSHIP,
    GEOMETRY_TRANSITION,
    # Dynamical Systems
    DYNAMICS_REGIME,
    DYNAMICS_STABILITY,
    DYNAMICS_TRAJECTORY,
    DYNAMICS_TRANSITION,
    # Transitions
    TRANSITION_NUMERIC,
    TRANSITION_SEVERITY,
    STATE_TRANSITION,
    # Causal Mechanics
    MECHANICS_ENERGY,
    MECHANICS_FLOW,
    MECHANICS_ORBIT,
    MECHANICS_TRANSITION,
    # Domain overrides
    DOMAIN_OVERRIDES,
    # Helper functions
    get_typology_thresholds,
    is_meaningful_change,
    classify_severity,
    get_transition_type,
    get_breaking_threshold,
    get_domain_thresholds,
    list_domains,
    get_domain_description,
    # Runtime override support (for Streamlit)
    set_runtime_overrides,
    clear_runtime_overrides,
    get_active_thresholds,
)

__all__ = [
    # Dataset configuration (axis-agnostic)
    'DatasetConfig',
    'normalize_columns',
    'denormalize_columns',
    'INTERNAL_ENTITY',
    'INTERNAL_AXIS',
    'INTERNAL_SIGNAL',
    'INTERNAL_VALUE',
    'CMAPSS_CONFIG',
    'ELECTROCHEM_CONFIG',
    'FEMTO_CONFIG',
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
    # ORTHON classification thresholds
    'TYPOLOGY_CLASSIFICATION',
    'TYPOLOGY_AXIS_THRESHOLDS',
    'TYPOLOGY_TRANSITION',
    'GEOMETRY_TOPOLOGY',
    'GEOMETRY_STABILITY',
    'GEOMETRY_LEADERSHIP',
    'GEOMETRY_TRANSITION',
    'DYNAMICS_REGIME',
    'DYNAMICS_STABILITY',
    'DYNAMICS_TRAJECTORY',
    'DYNAMICS_TRANSITION',
    'TRANSITION_NUMERIC',
    'TRANSITION_SEVERITY',
    'STATE_TRANSITION',
    'MECHANICS_ENERGY',
    'MECHANICS_FLOW',
    'MECHANICS_ORBIT',
    'MECHANICS_TRANSITION',
    'DOMAIN_OVERRIDES',
    'get_typology_thresholds',
    'is_meaningful_change',
    'classify_severity',
    'get_transition_type',
    'get_breaking_threshold',
    'get_domain_thresholds',
    'list_domains',
    'get_domain_description',
    'set_runtime_overrides',
    'clear_runtime_overrides',
    'get_active_thresholds',
]
