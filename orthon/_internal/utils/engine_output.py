"""
Engine Output Normalization

Centralized handling for engine return values.

DESIGN PRINCIPLE:
An engine failing to produce a value is DATA, not an error.
- Math undefined → NULL
- Insufficient samples → NULL
- Algorithm doesn't converge → NULL
- Numerical instability → NULL

Only actual Python exceptions (crashes) are errors.
"""

import logging
from typing import Any, Optional, Union
import numpy as np


logger = logging.getLogger(__name__)


def normalize_engine_value(value: Any) -> Optional[float]:
    """
    Normalize an engine output value.

    Engines may return:
        - float: valid result
        - None: explicit "no value"
        - np.nan: treated same as None
        - int: converted to float

    Returns:
        float if valid, None otherwise

    This is the SINGLE source of truth for engine output normalization.
    Do not duplicate this logic elsewhere.
    """
    if value is None:
        return None

    # Handle numpy types
    if isinstance(value, (np.floating, np.integer)):
        value = float(value)

    # Handle int
    if isinstance(value, int):
        value = float(value)

    # Check for NaN
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value

    # Non-numeric types become None
    return None


def is_null_result(value: Any) -> bool:
    """
    Check if a value represents a NULL result.

    Returns True for None, NaN, or Inf values.
    """
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return np.isnan(value) or np.isinf(value)
    return False


def safe_engine_call(engine_func, *args, engine_name: str = "unknown", **kwargs) -> Optional[float]:
    """
    Safely call an engine function, catching math-related exceptions.

    This wraps engine calls to ensure:
    - Math domain errors → None (not exception)
    - Value errors from algorithms → None
    - Real crashes → re-raised

    Args:
        engine_func: The engine function to call
        *args: Positional arguments for the function
        engine_name: Name for logging purposes
        **kwargs: Keyword arguments for the function

    Returns:
        Normalized engine value (float or None)
    """
    try:
        result = engine_func(*args, **kwargs)
        normalized = normalize_engine_value(result)

        if normalized is None:
            logger.debug(f"Engine {engine_name} returned NULL")

        return normalized

    except (ValueError, ZeroDivisionError, FloatingPointError) as e:
        # Math-related issues are NULL, not errors
        logger.debug(f"Engine {engine_name} math issue (NULL): {e}")
        return None

    except np.linalg.LinAlgError as e:
        # Linear algebra issues (singular matrix, etc.) are NULL
        logger.debug(f"Engine {engine_name} linalg issue (NULL): {e}")
        return None

    except RuntimeWarning as e:
        # Numerical warnings (overflow, etc.) are NULL
        logger.debug(f"Engine {engine_name} numerical warning (NULL): {e}")
        return None

    # All other exceptions propagate up as real errors
