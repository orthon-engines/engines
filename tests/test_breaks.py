"""
Tests for ENGINES Break Detection Engine.

Tests the Heaviside (step) and Dirac (impulse) detection.
"""

import numpy as np
import pytest
from engines.manifold.breaks import compute, summarize_breaks


class TestBreakDetection:
    """Test break detection on synthetic signals."""

    def test_heaviside_step_up(self):
        """Detect upward step (Heaviside)."""
        y = np.concatenate([np.ones(500) * 10, np.ones(500) * 20])
        breaks = compute(y, 'test_step')

        assert len(breaks) >= 1, "Should detect at least one break"

        # Find the main break near index 500
        main_break = min(breaks, key=lambda b: abs(b['I'] - 500))
        assert abs(main_break['I'] - 500) < 20, "Break should be near index 500"
        assert main_break['direction'] == 1, "Should be upward step"
        assert main_break['duration'] <= 5, "Step should be sharp"

    def test_heaviside_step_down(self):
        """Detect downward step (Heaviside)."""
        y = np.concatenate([np.ones(500) * 30, np.ones(500) * 10])
        breaks = compute(y, 'test_step_down')

        assert len(breaks) >= 1
        main_break = min(breaks, key=lambda b: abs(b['I'] - 500))
        assert main_break['direction'] == -1, "Should be downward step"

    def test_dirac_impulse(self):
        """Detect isolated spike (Dirac impulse)."""
        np.random.seed(42)
        y = np.random.randn(1000) * 0.5
        y[500] = 50.0  # Large spike

        breaks = compute(y, 'test_impulse', sensitivity=1.5)

        assert len(breaks) >= 1, "Should detect the spike"
        spike_break = min(breaks, key=lambda b: abs(b['I'] - 500))
        assert abs(spike_break['I'] - 500) < 5, "Should find spike at index 500"
        # Sharpness is MAD-normalized magnitude / duration
        # For a spike in noise, SNR should be high
        assert spike_break['snr'] > 10.0, "Impulse should have high SNR"

    def test_stationary_no_breaks(self):
        """Stationary signal should have few or no breaks."""
        np.random.seed(42)
        y = np.random.randn(1000)

        breaks = compute(y, 'test_stationary', sensitivity=0.5)

        # Should have very few breaks (noise may cause some)
        assert len(breaks) < 5, "Stationary signal should have few breaks"

    def test_multiple_regimes(self):
        """Detect multiple regime changes."""
        y = np.concatenate([
            np.ones(200) * 0,
            np.ones(200) * 10,
            np.ones(200) * 5,
            np.ones(200) * 15,
            np.ones(200) * 8,
        ])
        breaks = compute(y, 'test_regimes')

        assert len(breaks) >= 3, "Should detect at least 3 of 4 transitions"

    def test_gradual_shift(self):
        """Detect gradual level shift."""
        # Linear ramp from 0 to 10 over 100 samples, then constant
        ramp = np.linspace(0, 10, 100)
        y = np.concatenate([
            np.zeros(400),
            ramp,
            np.ones(500) * 10,
        ])
        breaks = compute(y, 'test_gradual')

        # Should detect the transition region
        assert len(breaks) >= 1

    def test_empty_signal(self):
        """Empty signal returns empty list."""
        breaks = compute(np.array([]), 'test_empty')
        assert breaks == []

    def test_short_signal(self):
        """Very short signal returns empty list."""
        breaks = compute(np.array([1, 2, 3]), 'test_short')
        assert breaks == []

    def test_constant_signal(self):
        """Constant signal has no breaks."""
        y = np.ones(1000) * 5.0
        breaks = compute(y, 'test_constant')
        assert len(breaks) == 0

    def test_nan_handling(self):
        """Signal with NaN values is handled."""
        y = np.ones(1000)
        y[500] = np.nan
        y[501] = np.nan

        # Should not crash
        breaks = compute(y, 'test_nan')
        # Constant signal minus NaN should have no breaks
        assert len(breaks) <= 1


class TestSummarizeBreaks:
    """Test break summary statistics."""

    def test_empty_breaks(self):
        """Empty breaks list gives zero summary."""
        summary = summarize_breaks([])

        assert summary['n_breaks'] == 0
        assert summary['mean_break_spacing'] == 0.0
        assert summary['mean_magnitude'] == 0.0

    def test_single_break(self):
        """Single break summary."""
        breaks = [{
            'I': 100,
            'magnitude': 5.0,
            'sharpness': 2.5,
            'direction': 1,
            'duration': 2,
            'pre_level': 10.0,
            'post_level': 15.0,
            'snr': 5.0,
        }]
        summary = summarize_breaks(breaks)

        assert summary['n_breaks'] == 1
        assert summary['mean_break_spacing'] == 0.0  # Only one break
        assert summary['mean_magnitude'] == 5.0
        assert summary['max_magnitude'] == 5.0
        assert summary['mean_sharpness'] == 2.5

    def test_multiple_breaks(self):
        """Multiple breaks summary with spacing."""
        breaks = [
            {'I': 100, 'magnitude': 3.0, 'sharpness': 1.0, 'direction': 1,
             'duration': 1, 'pre_level': 0, 'post_level': 3, 'snr': 3.0},
            {'I': 200, 'magnitude': 5.0, 'sharpness': 2.0, 'direction': -1,
             'duration': 2, 'pre_level': 3, 'post_level': -2, 'snr': 5.0},
            {'I': 350, 'magnitude': 4.0, 'sharpness': 1.5, 'direction': 1,
             'duration': 1, 'pre_level': -2, 'post_level': 2, 'snr': 4.0},
        ]
        summary = summarize_breaks(breaks)

        assert summary['n_breaks'] == 3
        assert summary['mean_break_spacing'] == 125.0  # (100 + 150) / 2
        assert summary['max_magnitude'] == 5.0
        assert summary['mean_sharpness'] == 1.5


class TestSensitivity:
    """Test sensitivity parameter effects."""

    def test_high_sensitivity_more_breaks(self):
        """Higher sensitivity should detect more breaks."""
        np.random.seed(42)
        y = np.random.randn(1000)
        y[300] = 3.0  # Small bump
        y[600] = 4.0  # Medium bump

        breaks_low = compute(y, 'test', sensitivity=0.5)
        breaks_high = compute(y, 'test', sensitivity=2.0)

        assert len(breaks_high) >= len(breaks_low), \
            "Higher sensitivity should find at least as many breaks"

    def test_conservative_sensitivity(self):
        """Conservative sensitivity ignores small deviations."""
        np.random.seed(42)
        y = np.random.randn(1000)

        breaks = compute(y, 'test', sensitivity=0.3)
        assert len(breaks) == 0, "Very conservative should find no breaks in noise"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
