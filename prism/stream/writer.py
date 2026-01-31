"""
Parquet Stream Writer
=====================

Write parquet output using Polars for efficiency.
"""

import io
from typing import Dict, Any, List, Optional


class ParquetStreamWriter:
    """
    Write parquet incrementally using Polars.
    
    Accumulates rows in memory, flushes to output when finalized.
    """
    
    def __init__(self):
        self.rows: List[Dict[str, Any]] = []
    
    def write_row(self, **kwargs) -> None:
        """Write a result row."""
        self.rows.append(kwargs)
    
    def finalize(self) -> bytes:
        """Complete the parquet file and return bytes."""
        if not self.rows:
            return b''
        
        try:
            import polars as pl
            
            # Create DataFrame from rows
            df = pl.DataFrame(self.rows)
            
            # Write to bytes buffer
            buffer = io.BytesIO()
            df.write_parquet(buffer)
            return buffer.getvalue()
            
        except Exception as e:
            raise ValueError(f"Failed to write parquet: {e}")


# Output schema for primitives.parquet
PRIMITIVES_SCHEMA = {
    'signal_id': str,
    'unit_id': str,
    'hurst': float,
    'hurst_r2': float,
    'lyapunov': float,
    'spectrum': list,  # float[]
    'garch_omega': float,
    'garch_alpha': float,
    'garch_beta': float,
    'sample_entropy': float,
    'permutation_entropy': float,
    'wavelet_energy': list,  # float[]
    'rqa_rr': float,
    'rqa_det': float,
    'rqa_lam': float,
    'pca_variance': list,  # float[]
    'umap_coords': list,  # float[]
}
