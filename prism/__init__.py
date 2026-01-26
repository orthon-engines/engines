"""
PRISM - Stream Compute Engine
=============================

Stateless stream processing for signal primitives.

    BYTES IN → COMPUTE → BYTES OUT
               (nothing stored)

Architecture:
    - engines/: Irreducible algorithms (hurst, fft, garch, etc.)
    - stream/: Streaming infrastructure (parser, buffer, writer)
    - server/: HTTP/Lambda handlers

Usage:
    # Start server
    uvicorn prism.server.routes:app --host 0.0.0.0 --port 8080
    
    # Or use Lambda
    from prism.server import lambda_handler
"""

__version__ = "2.0.0"
__architecture__ = "stream"

from . import engines
from . import stream
from . import server

__all__ = ['engines', 'stream', 'server', '__version__']
