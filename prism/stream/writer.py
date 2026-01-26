"""
Parquet Stream Writer
=====================

Write parquet output incrementally as results are computed.
"""

import io
from typing import Dict, Any, List, Optional
import pyarrow as pa
import pyarrow.parquet as pq


class ParquetStreamWriter:
    """
    Write parquet incrementally.
    
    Accumulates rows in memory, flushes to output when finalized.
    For true streaming, would use row groups but pyarrow doesn't
    support streaming write easily.
    """
    
    def __init__(self):
        self.rows: List[Dict[str, Any]] = []
        self.schema: Optional[pa.Schema] = None
    
    def write_row(self, **kwargs) -> None:
        """Write a result row."""
        self.rows.append(kwargs)
    
    def finalize(self) -> bytes:
        """Complete the parquet file and return bytes."""
        if not self.rows:
            return b''
        
        # Build schema from first row
        fields = []
        sample = self.rows[0]
        
        for key, value in sample.items():
            if isinstance(value, float):
                fields.append(pa.field(key, pa.float64()))
            elif isinstance(value, int):
                fields.append(pa.field(key, pa.int64()))
            elif isinstance(value, (list, tuple)):
                fields.append(pa.field(key, pa.list_(pa.float64())))
            elif isinstance(value, dict):
                fields.append(pa.field(key, pa.string()))  # JSON string
            else:
                fields.append(pa.field(key, pa.string()))
        
        schema = pa.schema(fields)
        
        # Convert rows to columnar format
        columns = {key: [] for key in sample.keys()}
        for row in self.rows:
            for key in sample.keys():
                value = row.get(key)
                # Convert dicts to JSON strings
                if isinstance(value, dict):
                    import json
                    value = json.dumps(value)
                columns[key].append(value)
        
        # Create table
        arrays = []
        for field in schema:
            col_data = columns[field.name]
            
            if pa.types.is_list(field.type):
                # Handle list columns
                arrays.append(pa.array(col_data, type=field.type))
            else:
                arrays.append(pa.array(col_data, type=field.type))
        
        table = pa.Table.from_arrays(arrays, schema=schema)
        
        # Write to bytes
        buffer = io.BytesIO()
        pq.write_table(table, buffer)
        return buffer.getvalue()


# Output schema for primitives.parquet
PRIMITIVES_SCHEMA = {
    'signal_id': str,
    'entity_id': str,
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
