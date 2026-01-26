"""
PRISM Server
============

HTTP/Lambda handlers for stream compute.
"""

from .handler import stream_compute_sync, lambda_handler
from .routes import app

__all__ = ['stream_compute_sync', 'lambda_handler', 'app']
