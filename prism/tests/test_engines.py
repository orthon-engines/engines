"""
Test PRISM Engines
==================
"""

import numpy as np
import pytest


def test_hurst_random_walk():
    """Random walk should have H ≈ 0.5"""
    from prism.engines.core import hurst
    
    np.random.seed(42)
    y = np.cumsum(np.random.randn(1000))
    
    result = hurst.compute(y)
    
    assert 0.3 < result['hurst'] < 0.7
    assert result['r2'] > 0.5


def test_hurst_trending():
    """Strong trend should have H > 0.5"""
    from prism.engines.core import hurst
    
    y = np.arange(1000) + np.random.randn(1000) * 10
    
    result = hurst.compute(y)
    
    assert result['hurst'] > 0.6


def test_entropy_sample():
    """Sample entropy should be positive."""
    from prism.engines.core import entropy
    
    np.random.seed(42)
    y = np.random.randn(500)
    
    result = entropy.compute_sample_entropy(y)
    
    assert result['sample_entropy'] > 0


def test_entropy_permutation():
    """Permutation entropy should be in [0, 1]."""
    from prism.engines.core import entropy
    
    np.random.seed(42)
    y = np.random.randn(500)
    
    result = entropy.compute_permutation_entropy(y)
    
    assert 0 <= result['permutation_entropy'] <= 1


def test_work_order():
    """Work order parsing."""
    from prism.stream.protocol import parse_work_order, encode_work_order, WorkOrder
    
    wo = WorkOrder(
        signals={
            'signal_1': {'needs_hurst': True, 'needs_fft': False},
            'signal_2': {'needs_hurst': True, 'needs_garch': True}
        }
    )
    
    encoded = encode_work_order(wo)
    decoded = parse_work_order(encoded)
    
    assert decoded.needs('signal_1', 'hurst') == True
    assert decoded.needs('signal_1', 'fft') == False
    assert decoded.needs('signal_2', 'garch') == True


def test_signal_buffer():
    """Signal buffer accumulation."""
    from prism.stream.buffer import SignalBuffer
    
    buffer = SignalBuffer(min_samples=5)
    
    rows = [
        {'signal_id': 'sig1', 'index': i, 'value': float(i)}
        for i in range(10)
    ]
    
    buffer.add(rows)
    
    assert 'sig1' in buffer.ready_signals(min_samples=5)
    
    data = buffer.pop('sig1')
    assert len(data) == 10
    assert data[0] == 0
    assert data[9] == 9


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
