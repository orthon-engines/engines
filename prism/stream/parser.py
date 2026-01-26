"""
Chunk Parser
============

Parse incoming data chunks (parquet/csv) into rows.
"""

import io
from typing import List, Dict, Any, Optional


def parse_chunk(chunk: bytes, format: str = 'parquet') -> List[Dict[str, Any]]:
    """
    Parse a chunk of data into rows.
    
    Args:
        chunk: Raw bytes
        format: 'parquet' or 'csv'
    
    Returns:
        List of row dicts with keys: entity_id, signal_id, index, value
    """
    if format == 'parquet':
        return _parse_parquet(chunk)
    elif format == 'csv':
        return _parse_csv(chunk)
    else:
        raise ValueError(f"Unknown format: {format}")


def _parse_parquet(chunk: bytes) -> List[Dict[str, Any]]:
    """Parse parquet bytes."""
    try:
        import pyarrow.parquet as pq
        
        buffer = io.BytesIO(chunk)
        table = pq.read_table(buffer)
        
        # Convert to list of dicts
        df = table.to_pandas()
        return df.to_dict('records')
        
    except Exception as e:
        raise ValueError(f"Failed to parse parquet: {e}")


def _parse_csv(chunk: bytes) -> List[Dict[str, Any]]:
    """Parse CSV bytes."""
    try:
        import csv
        
        text = chunk.decode('utf-8')
        reader = csv.DictReader(io.StringIO(text))
        
        rows = []
        for row in reader:
            rows.append({
                'entity_id': row.get('entity_id', ''),
                'signal_id': row.get('signal_id', ''),
                'index': float(row.get('index', 0)),
                'value': float(row.get('value', 0))
            })
        
        return rows
        
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
