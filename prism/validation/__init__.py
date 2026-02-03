"""
PRISM Validation Module

Validates pipeline prerequisites and input data before compute stages.

Exports:
    - check_prerequisites: Validate that required files exist for a stage
    - validate_input: Validate observations.parquet schema and data quality
    - PrerequisiteError: Raised when prerequisites are not met
    - ValidationError: Raised when input validation fails
    - StagePrerequisites: Stage dependency definitions
"""

from .prerequisites import (
    check_prerequisites,
    PrerequisiteError,
    StagePrerequisites,
    STAGE_PREREQUISITES,
)

from .input_validation import (
    validate_input,
    ValidationError,
    filter_constant_signals,
    InputValidationReport,
)

__all__ = [
    # Prerequisites
    'check_prerequisites',
    'PrerequisiteError',
    'StagePrerequisites',
    'STAGE_PREREQUISITES',
    # Input validation
    'validate_input',
    'ValidationError',
    'filter_constant_signals',
    'InputValidationReport',
]
