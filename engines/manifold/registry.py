"""
Engine Registry - discovers and loads all available engines.

The registry provides:
1. Auto-discovery of engines with config.yaml files
2. Lazy loading of engine instances
3. Window size computation for engine + signal combinations
"""

from pathlib import Path
from typing import Dict, List, Optional, Callable
import yaml

from .base import EngineConfig, EngineRequirements, load_engine_config


class EngineRegistry:
    """
    Registry of available ENGINES engines.

    Discovers engines by scanning for config.yaml files in the engines directory.
    Provides lazy loading of engine compute functions.
    """

    def __init__(self, engines_dir: Optional[Path] = None):
        if engines_dir is None:
            engines_dir = Path(__file__).parent / "signal"

        self.engines_dir = engines_dir
        self._configs: Dict[str, EngineConfig] = {}
        self._compute_funcs: Dict[str, Callable] = {}

        self._discover_engines()

    def _discover_engines(self):
        """Find all engines with config.yaml files."""
        # Look for .yaml files alongside .py files
        for config_path in self.engines_dir.glob("*.yaml"):
            engine_name = config_path.stem
            if engine_name.startswith("_"):
                continue

            try:
                self._configs[engine_name] = load_engine_config(config_path)
            except Exception as e:
                print(f"Warning: Failed to load config for {engine_name}: {e}")

        # Also look for config.yaml in subdirectories
        for config_path in self.engines_dir.glob("*/config.yaml"):
            engine_dir = config_path.parent
            engine_name = engine_dir.name

            if engine_name.startswith("_"):
                continue

            try:
                self._configs[engine_name] = load_engine_config(config_path)
            except Exception as e:
                print(f"Warning: Failed to load config for {engine_name}: {e}")

    def list_engines(self) -> List[str]:
        """List all available engine names."""
        return sorted(self._configs.keys())

    def has_engine(self, engine_name: str) -> bool:
        """Check if engine exists in registry."""
        return engine_name in self._configs

    def get_config(self, engine_name: str) -> EngineConfig:
        """Get configuration for an engine."""
        if engine_name not in self._configs:
            available = ", ".join(self.list_engines())
            raise KeyError(
                f"Unknown engine: '{engine_name}'. Available: {available}"
            )
        return self._configs[engine_name]

    def get_compute_func(self, engine_name: str) -> Callable:
        """
        Get compute function for an engine.

        Lazily imports the engine module on first access.
        """
        if engine_name not in self._compute_funcs:
            # Try to import from the signal module
            try:
                import importlib
                module = importlib.import_module(
                    f"engines.manifold.signal.{engine_name}"
                )
                self._compute_funcs[engine_name] = module.compute
            except (ImportError, AttributeError) as e:
                raise ImportError(
                    f"Could not load compute function for '{engine_name}': {e}"
                )

        return self._compute_funcs[engine_name]

    def get_window_for_signal(
        self,
        engine_name: str,
        window_factor: float = 1.0
    ) -> int:
        """
        Compute effective window for engine + signal combination.

        Args:
            engine_name: Name of the engine
            window_factor: Signal-specific window multiplier from typology

        Returns:
            Effective window size
        """
        config = self.get_config(engine_name)
        return config.requirements.compute_effective_window(window_factor)

    def get_min_samples(self, engine_name: str) -> int:
        """Get minimum samples required for an engine."""
        config = self.get_config(engine_name)
        return config.requirements.min_window

    def get_all_outputs(self) -> Dict[str, List[str]]:
        """Get outputs for all engines."""
        return {name: config.outputs for name, config in self._configs.items()}

    def get_outputs(self, engine_name: str) -> List[str]:
        """Get outputs for a specific engine."""
        config = self.get_config(engine_name)
        return config.outputs

    def validate_manifest_engines(self, engine_names: List[str]) -> Dict[str, List[str]]:
        """
        Validate that requested engines exist.

        Returns:
            Dict with 'available' and 'missing' engine lists
        """
        available = [e for e in engine_names if self.has_engine(e)]
        missing = [e for e in engine_names if not self.has_engine(e)]

        return {
            'available': available,
            'missing': missing,
            'coverage': len(available) / len(engine_names) if engine_names else 1.0,
        }


# Global registry instance (lazy initialized)
_registry: Optional[EngineRegistry] = None


def get_registry() -> EngineRegistry:
    """Get or create global engine registry."""
    global _registry
    if _registry is None:
        _registry = EngineRegistry()
    return _registry


def reset_registry():
    """Reset the global registry (for testing)."""
    global _registry
    _registry = None
