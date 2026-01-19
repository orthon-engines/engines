"""
PRISM Reports — Human-readable summaries of each pipeline stage.

Reports:
    observations_report     Summary of raw data loaded
    cohort_discovery_report What PRISM discovered about signal groupings
    signal_vector_report    51 behavioral metrics summary
    geometry_report         Coupling, structure, PCA analysis
    state_report            Velocity, acceleration, dynamics
    ml_acceleration_report  Ablation study and ML readiness

Usage:
    python -m reports.observations_report --domain cmapss
    python -m reports.cohort_discovery_report --domain cmapss
    python -m reports.signal_vector_report --domain cmapss
    python -m reports.geometry_report --domain cmapss
    python -m reports.state_report --domain cmapss
    python -m reports.ml_acceleration_report --domain cmapss --target RUL

All reports support:
    --domain NAME     Domain name (or use PRISM_DOMAIN env var)
    --output FILE     Save to markdown (.md) or JSON (.json)

Reports use the domain YAML to translate PRISM internal names back to
human-readable source names for presentation.
"""

from .report_utils import (
    ReportBuilder,
    load_domain_config,
    translate_signal_id,
    translate_signal_column,
    format_number,
    format_percentage,
    format_table,
)

__all__ = [
    'ReportBuilder',
    'load_domain_config',
    'translate_signal_id',
    'translate_signal_column',
    'format_number',
    'format_percentage',
    'format_table',
]
