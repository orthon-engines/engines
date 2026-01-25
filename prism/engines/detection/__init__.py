"""
PRISM Detection Engines

Honest names for what these engines actually do:
- step_detector: Detects level shifts (step changes) in time series
- spike_detector: Detects impulse events (spikes) in time series

Note: These are NOT the mathematical functions they were previously named after.
- This is NOT the Heaviside function H(x) = 0 for x<0, 1 for x>=0
- This is NOT the Dirac delta distribution with integral = 1

These engines detect WHERE step/spike events occur in data.
The names reflect what they actually compute.
"""

from prism.engines.detection.step_detector import compute as detect_steps
from prism.engines.detection.spike_detector import compute as detect_spikes

__all__ = ['detect_steps', 'detect_spikes']
