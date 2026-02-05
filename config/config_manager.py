"""
PRISM Configuration Manager

Loads and merges configuration from YAML files with hierarchical overrides:
    defaults/ < domains/{domain}.yaml < environments/{env}.yaml

Usage:
    from config.config_manager import ConfigManager

    # Load with defaults only
    config = ConfigManager()

    # Load with domain overrides
    config = ConfigManager(domain='turbofan')

    # Load with domain and environment overrides
    config = ConfigManager(domain='turbofan', environment='production')

    # Access values
    min_signals = config.get('geometry.eigenvalue.min_signals')
    feature_groups = config.get('geometry.feature_groups')

    # Access with default fallback
    threshold = config.get('dynamics.collapse.threshold_velocity', default=-0.1)
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import copy


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file, return empty dict if not found."""
    if not path.exists():
        return {}

    try:
        import yaml
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # PyYAML not available, use simple parser
        return _parse_yaml_simple(path)


def _parse_yaml_simple(path: Path) -> Dict[str, Any]:
    """
    Simple YAML parser for basic configs (no external dependencies).
    Handles: scalars, lists, nested dicts with proper indentation.
    """
    result = {}
    stack = [(0, result, None)]  # (indent_level, current_dict, list_key)

    with open(path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        # Skip comments and empty lines
        if not stripped or stripped.startswith('#'):
            i += 1
            continue

        # Calculate indent
        indent = len(line) - len(line.lstrip())

        # Pop stack to correct level
        while len(stack) > 1 and stack[-1][0] >= indent:
            stack.pop()

        current = stack[-1][1]
        list_key = stack[-1][2]

        # Handle list items
        if stripped.startswith('- '):
            item_value = stripped[2:].strip()

            # Find the list we're adding to
            if list_key and list_key in current:
                if not isinstance(current[list_key], list):
                    current[list_key] = []
                current[list_key].append(_parse_value(item_value))
            i += 1
            continue

        # Parse key: value
        if ':' in stripped:
            key, _, value = stripped.partition(':')
            key = key.strip()
            value = value.strip()

            if value:
                # Inline value
                if value.startswith('['):
                    # List value: [a, b, c]
                    items = value[1:-1].split(',')
                    current[key] = [item.strip().strip('"\'') for item in items if item.strip()]
                else:
                    # Scalar value
                    current[key] = _parse_value(value)
            else:
                # Check if next non-empty line starts with '- ' (list) or is a nested dict
                next_idx = i + 1
                while next_idx < len(lines):
                    next_line = lines[next_idx]
                    next_stripped = next_line.lstrip()
                    if next_stripped and not next_stripped.startswith('#'):
                        break
                    next_idx += 1

                if next_idx < len(lines) and lines[next_idx].lstrip().startswith('- '):
                    # It's a list
                    current[key] = []
                    next_indent = len(lines[next_idx]) - len(lines[next_idx].lstrip())
                    stack.append((next_indent, current, key))
                else:
                    # Nested dict
                    current[key] = {}
                    stack.append((indent + 2, current[key], None))

        i += 1

    return result


def _parse_value(value: str) -> Any:
    """Parse a YAML value string to appropriate Python type."""
    # Remove quotes
    if (value.startswith('"') and value.endswith('"')) or \
       (value.startswith("'") and value.endswith("'")):
        return value[1:-1]

    # Booleans
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False
    if value.lower() in ('null', 'none', '~'):
        return None

    # Numbers
    try:
        if '.' in value or 'e' in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        pass

    return value


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries. Override values take precedence.

    Lists are replaced, not merged.
    """
    result = copy.deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        elif isinstance(value, list):
            # Lists are replaced, not merged
            result[key] = copy.deepcopy(value)
        elif isinstance(value, dict):
            result[key] = copy.deepcopy(value)
        else:
            result[key] = value

    return result


class ConfigManager:
    """
    Configuration manager with hierarchical loading.

    Load order (later overrides earlier):
        1. defaults/*.yaml
        2. domains/{domain}.yaml
        3. environments/{environment}.yaml

    Args:
        domain: Optional domain name (e.g., 'turbofan', 'chemical_reactor')
        environment: Optional environment name (e.g., 'production', 'development')
        config_dir: Optional path to config directory (defaults to ./config)
    """

    def __init__(
        self,
        domain: Optional[str] = None,
        environment: Optional[str] = None,
        config_dir: Optional[Union[str, Path]] = None
    ):
        self.domain = domain
        self.environment = environment

        # Find config directory
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            # Look for config dir relative to this file or cwd
            possible_paths = [
                Path(__file__).parent,  # Same dir as this file
                Path(__file__).parent.parent / 'config',  # Parent/config
                Path.cwd() / 'config',  # CWD/config
            ]
            self.config_dir = next(
                (p for p in possible_paths if p.exists() and (p / 'defaults').exists()),
                Path(__file__).parent
            )

        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load and merge all configuration files."""
        # 1. Load defaults
        defaults_dir = self.config_dir / 'defaults'
        if defaults_dir.exists():
            for yaml_file in defaults_dir.glob('*.yaml'):
                config_name = yaml_file.stem
                self._config[config_name] = _load_yaml(yaml_file)

        # 2. Load domain overrides
        if self.domain:
            domain_file = self.config_dir / 'domains' / f'{self.domain}.yaml'
            if domain_file.exists():
                domain_config = _load_yaml(domain_file)
                for key, value in domain_config.items():
                    if key in self._config and isinstance(self._config[key], dict):
                        self._config[key] = _deep_merge(self._config[key], value)
                    else:
                        self._config[key] = value

        # 3. Load environment overrides
        if self.environment:
            env_file = self.config_dir / 'environments' / f'{self.environment}.yaml'
            if env_file.exists():
                env_config = _load_yaml(env_file)
                for key, value in env_config.items():
                    if key in self._config and isinstance(self._config[key], dict):
                        self._config[key] = _deep_merge(self._config[key], value)
                    else:
                        self._config[key] = value

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get a config value by dot-separated path.

        Args:
            path: Dot-separated path (e.g., 'geometry.eigenvalue.min_signals')
            default: Default value if path not found

        Returns:
            Config value or default
        """
        keys = path.split('.')
        current = self._config

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire config section.

        Args:
            section: Section name (e.g., 'geometry', 'dynamics')

        Returns:
            Section dict or empty dict
        """
        return self._config.get(section, {})

    def reload(self) -> None:
        """Reload configuration from files."""
        self._config = {}
        self._load_config()

    @property
    def all(self) -> Dict[str, Any]:
        """Return the complete merged configuration."""
        return copy.deepcopy(self._config)


# Global config instance (lazy-loaded)
_global_config: Optional[ConfigManager] = None


def get_config(
    domain: Optional[str] = None,
    environment: Optional[str] = None,
    reload: bool = False
) -> ConfigManager:
    """
    Get the global config instance.

    Args:
        domain: Optional domain override
        environment: Optional environment override
        reload: Force reload of config

    Returns:
        ConfigManager instance
    """
    global _global_config

    if _global_config is None or reload or domain or environment:
        _global_config = ConfigManager(domain=domain, environment=environment)

    return _global_config
