"""
Base engine class with self-configuration.

Engines own their configuration (window requirements, outputs, dependencies).
This decouples engine internals from manifest bloat.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import math
import yaml
import numpy as np


@dataclass
class EngineRequirements:
    """Engine windowing requirements."""
    base_window: int
    min_window: int
    max_window: int
    scaling: str = "linear"  # linear, sqrt, log

    def compute_effective_window(self, window_factor: float) -> int:
        """
        Compute effective window size given a signal's window_factor.

        Args:
            window_factor: Multiplier from typology (1.0 = base, 2.0 = double, etc.)

        Returns:
            Effective window size, clamped to [min_window, max_window]
        """
        if self.scaling == "linear":
            effective = self.base_window * window_factor
        elif self.scaling == "sqrt":
            effective = self.base_window * math.sqrt(window_factor)
        elif self.scaling == "log":
            effective = self.base_window * math.log2(1 + window_factor)
        else:
            effective = self.base_window * window_factor

        # Clamp to engine limits
        effective = max(self.min_window, min(self.max_window, int(effective)))
        return effective


@dataclass
class EngineConfig:
    """Full engine configuration loaded from config.yaml."""
    name: str
    version: str
    requirements: EngineRequirements
    outputs: List[str]
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_engine_config(config_path: Path) -> EngineConfig:
    """Load engine configuration from YAML file."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    requirements = EngineRequirements(
        base_window=raw['requirements']['base_window'],
        min_window=raw['requirements']['min_window'],
        max_window=raw['requirements']['max_window'],
        scaling=raw['requirements'].get('scaling', 'linear'),
    )

    return EngineConfig(
        name=raw['engine'],
        version=raw.get('version', '1.0'),
        requirements=requirements,
        outputs=raw.get('outputs', []),
        dependencies=raw.get('dependencies', []),
        metadata=raw.get('metadata', {}),
    )


class BaseEngine(ABC):
    """
    Base class for all ENGINES engines.

    Subclasses must:
    1. Define engine_name property
    2. Implement compute() method
    3. Have a config.yaml in their directory
    """

    _config: Optional[EngineConfig] = None

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Return engine name (must match config.yaml 'engine' field)."""
        pass

    @property
    def config(self) -> EngineConfig:
        """Get or load engine configuration."""
        if self._config is None:
            self._config = self._load_config()
        return self._config

    def _load_config(self) -> EngineConfig:
        """Load config.yaml from engine's directory."""
        # Look for config in the same directory as the engine module
        engine_file = Path(__file__).parent / "signal" / f"{self.engine_name}.py"
        config_path = engine_file.with_suffix('.yaml')

        # Also check for config.yaml in engine directory
        if not config_path.exists():
            engine_dir = Path(__file__).parent / "signal" / self.engine_name
            config_path = engine_dir / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Engine config not found for '{self.engine_name}'. "
                f"Expected at: {config_path}"
            )

        return load_engine_config(config_path)

    def get_window(self, window_factor: float = 1.0) -> int:
        """Get effective window size for a signal."""
        return self.config.requirements.compute_effective_window(window_factor)

    def get_outputs(self) -> List[str]:
        """Return list of output column names."""
        return self.config.outputs

    def get_min_samples(self) -> int:
        """Return minimum samples required for this engine."""
        return self.config.requirements.min_window

    @abstractmethod
    def compute(self, data: np.ndarray) -> Dict[str, float]:
        """
        Compute engine features for a window of data.

        Args:
            data: Input signal data (1D numpy array)

        Returns:
            Dict mapping output names to computed values
        """
        pass

    def compute_with_factor(
        self,
        data: np.ndarray,
        window_factor: float = 1.0
    ) -> Dict[str, float]:
        """
        Compute features, adjusting for window factor.

        This method handles window sizing internally - the caller provides
        the full available data, and the engine extracts the appropriate
        window based on the factor.

        Args:
            data: Full signal data available
            window_factor: Signal-specific window multiplier

        Returns:
            Dict mapping output names to computed values
        """
        effective_window = self.get_window(window_factor)

        if len(data) < effective_window:
            # Return NaN for all outputs
            return {name: np.nan for name in self.get_outputs()}

        # Use the last `effective_window` samples
        window_data = data[-effective_window:]
        return self.compute(window_data)
