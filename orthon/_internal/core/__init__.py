"""
PRISM Core — Types and utilities.

NOT computation. For computation, see prism.engines.
"""

from orthon._internal.core.domain_clock import DomainClock, DomainInfo, auto_detect_window

__all__ = [
    'DomainClock',
    'DomainInfo',
    'auto_detect_window',
]
