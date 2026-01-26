"""
Signal Buffer
=============

Accumulates per-signal data until ready for computation.
Memory-bounded: tracks memory usage, evicts when needed.
"""

import numpy as np
from typing import Dict, List, Optional, Iterator
from dataclasses import dataclass, field


@dataclass
class SignalData:
    """Accumulated data for a single signal."""
    values: List[float] = field(default_factory=list)
    indices: List[float] = field(default_factory=list)
    entity_id: Optional[str] = None
    
    def add(self, index: float, value: float):
        self.indices.append(index)
        self.values.append(value)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array sorted by index."""
        if not self.values:
            return np.array([])
        
        # Sort by index
        pairs = sorted(zip(self.indices, self.values))
        return np.array([v for _, v in pairs])
    
    def __len__(self) -> int:
        return len(self.values)
    
    @property
    def memory_bytes(self) -> int:
        """Approximate memory usage."""
        return len(self.values) * 16  # 8 bytes each for value and index


class SignalBuffer:
    """
    Accumulates per-signal data until complete.
    
    Memory-bounded: evicts signals when memory limit reached.
    
    Usage:
        buffer = SignalBuffer(max_memory_mb=100)
        buffer.add(rows)
        
        for signal_id in buffer.ready_signals(min_samples=1000):
            data = buffer.pop(signal_id)
            # compute...
    """
    
    def __init__(self, max_memory_mb: float = 100.0, min_samples: int = 100):
        self.signals: Dict[str, SignalData] = {}
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.min_samples = min_samples
        self._memory_used = 0
    
    def add(self, rows: List[dict]) -> None:
        """
        Add rows to buffer, grouped by signal_id.
        
        Expected row format:
            {'entity_id': str, 'signal_id': str, 'index': float, 'value': float}
        """
        for row in rows:
            signal_id = row.get('signal_id')
            if not signal_id:
                continue
            
            if signal_id not in self.signals:
                self.signals[signal_id] = SignalData(entity_id=row.get('entity_id'))
            
            old_size = self.signals[signal_id].memory_bytes
            self.signals[signal_id].add(
                index=float(row.get('index', 0)),
                value=float(row.get('value', 0))
            )
            self._memory_used += self.signals[signal_id].memory_bytes - old_size
        
        # Evict if over memory limit
        self._maybe_evict()
    
    def ready_signals(self, min_samples: Optional[int] = None) -> List[str]:
        """Return signal_ids with enough data to compute."""
        min_samples = min_samples or self.min_samples
        return [
            sid for sid, data in self.signals.items()
            if len(data) >= min_samples
        ]
    
    def pop(self, signal_id: str) -> np.ndarray:
        """Remove and return signal data as numpy array."""
        if signal_id not in self.signals:
            return np.array([])
        
        data = self.signals.pop(signal_id)
        self._memory_used -= data.memory_bytes
        return data.to_array()
    
    def remaining(self) -> List[str]:
        """Return all remaining signal_ids."""
        return list(self.signals.keys())
    
    def get_entity_id(self, signal_id: str) -> Optional[str]:
        """Get entity_id for a signal."""
        if signal_id in self.signals:
            return self.signals[signal_id].entity_id
        return None
    
    def _maybe_evict(self) -> None:
        """Evict oldest/smallest signals if over memory limit."""
        if self._memory_used <= self.max_memory_bytes:
            return
        
        # Sort by size (evict smallest first - they're probably complete)
        by_size = sorted(
            self.signals.items(),
            key=lambda x: len(x[1])
        )
        
        while self._memory_used > self.max_memory_bytes * 0.8 and by_size:
            sid, data = by_size.pop(0)
            self._memory_used -= data.memory_bytes
            del self.signals[sid]
    
    @property
    def memory_mb(self) -> float:
        """Current memory usage in MB."""
        return self._memory_used / (1024 * 1024)
    
    def __len__(self) -> int:
        return len(self.signals)
