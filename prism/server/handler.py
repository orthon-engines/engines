"""
Stream Compute Handler
======================

Main entry point for stream computation.
Bytes in → Compute → Bytes out. Nothing stored.
"""

import numpy as np
from typing import Dict, Any, Optional, AsyncIterator

from prism.stream import (
    parse_chunk,
    detect_format,
    SignalBuffer,
    ParquetStreamWriter,
    WorkOrder,
    parse_work_order
)
from prism.engines import core


def compute_signal(signal_id: str, y: np.ndarray, work_order: WorkOrder) -> Dict[str, Any]:
    """
    Compute requested engines for a single signal.
    
    Args:
        signal_id: Signal identifier
        y: Signal data array
        work_order: What to compute
    
    Returns:
        Dict with signal_id and computed metrics
    """
    result = {'signal_id': signal_id}
    
    # Get engines needed for this signal
    engines = work_order.get_engines(signal_id)
    
    # If no work order, use defaults
    if not engines and not work_order.signals:
        engines = ['hurst', 'entropy']
    
    if len(y) < 10:
        # Not enough data
        return result
    
    # Compute each requested engine
    for engine in engines:
        try:
            if engine == 'hurst':
                res = core.hurst.compute(y)
                result['hurst'] = res.get('hurst')
                result['hurst_r2'] = res.get('r2')
            
            elif engine == 'lyapunov':
                res = core.lyapunov.compute(y)
                result['lyapunov'] = res.get('lyapunov_exponent')
            
            elif engine == 'fft':
                res = core.fft.compute(y)
                # Store dominant frequency and power
                result['fft_dominant_freq'] = res.get('dominant_freq')
                result['fft_power'] = res.get('total_power')
            
            elif engine == 'garch':
                res = core.garch.compute(y)
                result['garch_omega'] = res.get('omega')
                result['garch_alpha'] = res.get('alpha')
                result['garch_beta'] = res.get('beta')
            
            elif engine == 'entropy':
                res = core.entropy.compute(y, method='sample')
                result['sample_entropy'] = res.get('sample_entropy')
                res = core.entropy.compute(y, method='permutation')
                result['permutation_entropy'] = res.get('permutation_entropy')
            
            elif engine == 'wavelet':
                res = core.wavelet.compute(y)
                result['wavelet_energy'] = res.get('energy_by_level')
            
            elif engine == 'rqa':
                res = core.rqa.compute(y)
                result['rqa_rr'] = res.get('recurrence_rate')
                result['rqa_det'] = res.get('determinism')
                result['rqa_lam'] = res.get('laminarity')
        
        except Exception as e:
            # Log but don't fail - partial results are OK
            result[f'{engine}_error'] = str(e)
    
    return result


def stream_compute_sync(
    data: bytes,
    work_order_header: Optional[str] = None,
    format: Optional[str] = None
) -> bytes:
    """
    Synchronous stream compute.
    
    Args:
        data: Input data (parquet or csv bytes)
        work_order_header: Base64 encoded work order JSON
        format: 'parquet' or 'csv' (auto-detected if None)
    
    Returns:
        Parquet bytes with computed primitives
    """
    # Parse work order
    work_order = parse_work_order(work_order_header)
    
    # Detect format
    if format is None:
        format = detect_format(data)
    
    # Parse all data
    rows = parse_chunk(data, format)
    
    # Buffer signals
    buffer = SignalBuffer(max_memory_mb=100)
    buffer.add(rows)
    
    # Compute
    writer = ParquetStreamWriter()
    
    for signal_id in buffer.remaining():
        y = buffer.pop(signal_id)
        entity_id = buffer.get_entity_id(signal_id)
        
        result = compute_signal(signal_id, y, work_order)
        if entity_id:
            result['entity_id'] = entity_id
        
        writer.write_row(**result)
    
    return writer.finalize()


async def stream_compute_async(
    upload_stream: AsyncIterator[bytes],
    work_order: WorkOrder,
    format: str = 'parquet'
) -> AsyncIterator[bytes]:
    """
    Async streaming compute.
    
    Process data as it arrives. Yield results as computed.
    Never hold more than one chunk in memory.
    
    Args:
        upload_stream: Async iterator of data chunks
        work_order: What to compute
        format: Input format
    
    Yields:
        Parquet bytes chunks (accumulated, then finalized)
    """
    buffer = SignalBuffer(max_memory_mb=100)
    writer = ParquetStreamWriter()
    
    # Process chunks as they arrive
    async for chunk in upload_stream:
        rows = parse_chunk(chunk, format)
        buffer.add(rows)
        
        # Compute any signals that are ready
        for signal_id in buffer.ready_signals():
            y = buffer.pop(signal_id)
            result = compute_signal(signal_id, y, work_order)
            writer.write_row(**result)
    
    # Finalize remaining signals
    for signal_id in buffer.remaining():
        y = buffer.pop(signal_id)
        result = compute_signal(signal_id, y, work_order)
        writer.write_row(**result)
    
    # Yield final parquet
    yield writer.finalize()


# Lambda handler
def lambda_handler(event, context):
    """AWS Lambda entry point."""
    import base64
    
    # Get body
    body = event.get('body', '')
    if event.get('isBase64Encoded'):
        body = base64.b64decode(body)
    elif isinstance(body, str):
        body = body.encode('utf-8')
    
    # Get work order
    headers = event.get('headers', {})
    work_order_header = headers.get('x-work-order') or headers.get('X-Work-Order')
    
    # Compute
    result = stream_compute_sync(body, work_order_header)
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/octet-stream'
        },
        'body': base64.b64encode(result).decode('utf-8'),
        'isBase64Encoded': True
    }
