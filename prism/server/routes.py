"""
PRISM API Routes
================

HTTP endpoints for stream compute.
"""

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse
import time

from prism.stream import parse_work_order
from prism.server.handler import stream_compute_sync

app = FastAPI(
    title="PRISM",
    description="Stateless stream compute for signal primitives",
    version="2.0.0"
)


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "version": "2.0.0",
        "architecture": "stream"
    }


@app.post("/compute")
async def compute(request: Request):
    """
    Stream compute endpoint.
    
    Headers:
        Content-Type: application/octet-stream
        X-Work-Order: Base64 encoded JSON (optional)
    
    Body:
        Parquet or CSV bytes
    
    Returns:
        Parquet bytes with computed primitives
    """
    start = time.time()
    
    # Get work order from header
    work_order_header = request.headers.get("X-Work-Order")
    
    # Read body
    body = await request.body()
    
    if not body:
        raise HTTPException(status_code=400, detail="Empty body")
    
    try:
        # Compute
        result = stream_compute_sync(body, work_order_header)
        
        duration = time.time() - start
        
        return Response(
            content=result,
            media_type="application/octet-stream",
            headers={
                "X-Compute-Duration": str(duration),
                "X-Architecture": "stream"
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/engines")
async def list_engines():
    """List available engines."""
    return {
        "core": [
            "hurst", "lyapunov", "fft", "pca", "umap",
            "garch", "entropy", "wavelet", "rqa",
            "granger", "transfer_entropy", "cointegration",
            "dtw", "dmd", "embedding", "mutual_info",
            "clustering", "mst", "copula", "divergence"
        ],
        "physics": [
            "phase_equilibria", "activity_models",
            "reaction_kinetics", "separations",
            "electrochemistry"
        ]
    }


@app.get("/schema")
async def output_schema():
    """Output parquet schema."""
    return {
        "columns": {
            "signal_id": "string",
            "entity_id": "string (optional)",
            "hurst": "float64 (optional)",
            "hurst_r2": "float64 (optional)",
            "lyapunov": "float64 (optional)",
            "fft_dominant_freq": "float64 (optional)",
            "fft_power": "float64 (optional)",
            "garch_omega": "float64 (optional)",
            "garch_alpha": "float64 (optional)",
            "garch_beta": "float64 (optional)",
            "sample_entropy": "float64 (optional)",
            "permutation_entropy": "float64 (optional)",
            "wavelet_energy": "list<float64> (optional)",
            "rqa_rr": "float64 (optional)",
            "rqa_det": "float64 (optional)",
            "rqa_lam": "float64 (optional)"
        },
        "note": "Only requested fields are populated. Others NULL."
    }
