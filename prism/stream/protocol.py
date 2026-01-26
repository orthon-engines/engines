"""
Work Order Protocol
===================

ORTHON SQL generates work orders. PRISM reads them.
"""

import json
import base64
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class WorkOrder:
    """
    Work order specifying what to compute for each signal.
    
    Example:
        {
            "signals": {
                "P_inlet": {"needs_hurst": true, "needs_fft": false},
                "Pump_vibration": {"needs_fft": true, "needs_wavelet": true}
            },
            "system": {
                "needs_umap": true,
                "needs_pca": false,
                "umap_signals": ["P_inlet", "P_outlet"]
            }
        }
    """
    signals: Dict[str, Dict[str, bool]] = field(default_factory=dict)
    system: Dict[str, Any] = field(default_factory=dict)
    
    def needs(self, signal_id: str, engine: str) -> bool:
        """Check if signal needs a specific engine."""
        if signal_id not in self.signals:
            return False
        return self.signals[signal_id].get(f'needs_{engine}', False)
    
    def get_engines(self, signal_id: str) -> list:
        """Get list of engines needed for a signal."""
        if signal_id not in self.signals:
            return []
        return [
            k.replace('needs_', '')
            for k, v in self.signals[signal_id].items()
            if k.startswith('needs_') and v
        ]
    
    def needs_system(self, engine: str) -> bool:
        """Check if system-level computation is needed."""
        return self.system.get(f'needs_{engine}', False)


def parse_work_order(header: Optional[str]) -> WorkOrder:
    """
    Parse X-Work-Order header (base64 encoded JSON).
    
    Args:
        header: Base64 encoded JSON string, or None for defaults
    
    Returns:
        WorkOrder instance
    """
    if not header:
        # Default: compute all engines for all signals
        return WorkOrder()
    
    try:
        decoded = base64.b64decode(header).decode('utf-8')
        data = json.loads(decoded)
        return WorkOrder(
            signals=data.get('signals', {}),
            system=data.get('system', {})
        )
    except Exception as e:
        raise ValueError(f"Invalid work order: {e}")


def encode_work_order(work_order: WorkOrder) -> str:
    """Encode work order as base64 JSON for header."""
    data = {
        'signals': work_order.signals,
        'system': work_order.system
    }
    json_str = json.dumps(data)
    return base64.b64encode(json_str.encode('utf-8')).decode('utf-8')


# Default engines if no work order provided
DEFAULT_ENGINES = [
    'hurst',
    'fft',
    'garch',
    'entropy',
]
