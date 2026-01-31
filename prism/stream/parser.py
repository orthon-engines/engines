"""
Chunk Parser
============

Parse incoming data chunks (parquet/csv) into rows.
Uses Polars for efficient parquet handling.
"""

import io
from typing import List, Dict, Any


def parse_chunk(chunk: bytes, format: str = 'parquet') -> List[Dict[str, Any]]:
    """
    Parse a chunk of data into rows.
    
    Args:
        chunk: Raw bytes
        format: 'parquet' or 'csv'
    
    Returns:
        List of row dicts with keys: unit_id, signal_id, index, value
    """
    if format == 'parquet':
        return _parse_parquet(chunk)
    elif format == 'csv':
        return _parse_csv(chunk)
    else:
        raise ValueError(f"Unknown format: {format}")


def _parse_parquet(chunk: bytes) -> List[Dict[str, Any]]:
    """Parse parquet bytes using Polars."""
    try:
        import polars as pl
        
        buffer = io.BytesIO(chunk)
        df = pl.read_parquet(buffer)
        
        # Convert to list of dicts
        return df.to_dicts()
        
    except Exception as e:
        raise ValueError(f"Failed to parse parquet: {e}")


def _parse_csv(chunk: bytes) -> List[Dict[str, Any]]:
    """Parse CSV bytes using Polars."""
    try:
        import polars as pl
        
        text = chunk.decode('utf-8')
        df = pl.read_csv(io.StringIO(text))
        
        return df.to_dicts()
        
    except Exception as e:
        raise ValueError(f"Failed to parse CSV: {e}")


def detect_format(chunk: bytes) -> str:
    """Detect format from magic bytes."""
    # Parquet magic bytes
    if chunk[:4] == b'PAR1':
        return 'parquet'
    
    # CSV heuristic (starts with header)
    try:
        text = chunk[:1000].decode('utf-8')
        if 'signal_id' in text.lower() or 'value' in text.lower():
            return 'csv'
    except:
        pass
    
    # Default to parquet
    return 'parquet'
