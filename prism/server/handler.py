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
                result['lyapunov_method'] = res.get('method')
            
            elif engine == 'fft':
                res = core.fft.compute(y)
                result['fft_centroid'] = res.get('centroid')
                result['fft_bandwidth'] = res.get('bandwidth')
            
            elif engine == 'garch':
                res = core.garch.compute(y)
                result['garch_omega'] = res.get('omega')
                result['garch_alpha'] = res.get('alpha')
                result['garch_beta'] = res.get('beta')
                result['garch_persistence'] = res.get('persistence')
            
            elif engine == 'entropy':
                res = core.entropy.compute_sample_entropy(y)
                result['sample_entropy'] = res.get('sample_entropy')
                res = core.entropy.compute_permutation_entropy(y)
                result['permutation_entropy'] = res.get('permutation_entropy')
            
            elif engine == 'wavelet':
                res = core.wavelet.compute(y)
                result['wavelet_dominant_scale'] = res.get('dominant_scale')
                result['wavelet_scale_entropy'] = res.get('scale_entropy')
            
            elif engine == 'rqa':
                res = core.rqa.compute(y)
                result['rqa_recurrence_rate'] = res.get('recurrence_rate')
                result['rqa_determinism'] = res.get('determinism')
                result['rqa_laminarity'] = res.get('laminarity')
                result['rqa_entropy'] = res.get('diagonal_entropy')
            
            elif engine == 'granger':
                # Granger needs pairwise - skip for single signal
                pass
            
            elif engine == 'cointegration':
                # Cointegration needs pairwise - skip for single signal
                pass
            
            elif engine == 'dtw':
                # DTW needs pairwise - skip for single signal
                pass
        
        except Exception as e:
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
    work_order = parse_work_order(work_order_header)
    
    if format is None:
        format = detect_format(data)
    
    rows = parse_chunk(data, format)
    
    buffer = SignalBuffer(max_memory_mb=100)
    buffer.add(rows)
    
    writer = ParquetStreamWriter()
    
    for signal_id in buffer.remaining():
        y = buffer.pop(signal_id)
        unit_id = buffer.get_unit_id(signal_id)
        
        result = compute_signal(signal_id, y, work_order)
        if unit_id:
            result['unit_id'] = unit_id
        
        writer.write_row(**result)
    
    return writer.finalize()


async def stream_compute_async(
    upload_stream: AsyncIterator[bytes],
    work_order: WorkOrder,
    format: str = 'parquet'
) -> AsyncIterator[bytes]:
    """Async streaming compute."""
    buffer = SignalBuffer(max_memory_mb=100)
    writer = ParquetStreamWriter()
    
    async for chunk in upload_stream:
        rows = parse_chunk(chunk, format)
        buffer.add(rows)
        
        for signal_id in buffer.ready_signals():
            y = buffer.pop(signal_id)
            result = compute_signal(signal_id, y, work_order)
            writer.write_row(**result)
    
    for signal_id in buffer.remaining():
        y = buffer.pop(signal_id)
        result = compute_signal(signal_id, y, work_order)
        writer.write_row(**result)
    
    yield writer.finalize()


def lambda_handler(event, context):
    """AWS Lambda entry point."""
    import base64
    
    body = event.get('body', '')
    if event.get('isBase64Encoded'):
        body = base64.b64decode(body)
    elif isinstance(body, str):
        body = body.encode('utf-8')
    
    headers = event.get('headers', {})
    work_order_header = headers.get('x-work-order') or headers.get('X-Work-Order')
    
    result = stream_compute_sync(body, work_order_header)
    
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/octet-stream'},
        'body': base64.b64encode(result).decode('utf-8'),
        'isBase64Encoded': True
    }
