"""Create demo signals parquet for testing."""

import numpy as np


def create_demo_signals(output_path: str = 'demo_signals.parquet'):
    """Create demo parquet with test signals."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    np.random.seed(42)
    n = 1000
    
    signals = {
        'random_walk': np.cumsum(np.random.randn(n)),
        'sine_pure': np.sin(np.linspace(0, 20 * np.pi, n)),
        'trending': np.arange(n) + np.random.randn(n) * 10,
        'mean_reverting': np.zeros(n),
        'noisy': np.random.randn(n),
    }
    
    # Mean reverting (Ornstein-Uhlenbeck)
    theta, mu, sigma = 0.7, 0, 0.2
    for i in range(1, n):
        signals['mean_reverting'][i] = (
            signals['mean_reverting'][i-1] + 
            theta * (mu - signals['mean_reverting'][i-1]) + 
            sigma * np.random.randn()
        )
    
    # Build rows
    rows = []
    for signal_id, values in signals.items():
        for i, v in enumerate(values):
            rows.append({
                'entity_id': 'demo',
                'signal_id': signal_id,
                'index': float(i),
                'value': float(v)
            })
    
    # Create table
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_path)
    
    print(f"Created {output_path} with {len(rows)} rows")
    print(f"Signals: {list(signals.keys())}")


if __name__ == '__main__':
    create_demo_signals()
