"""
PRISM Universal Intake — One File, We Figure It Out

Upload ONE file. We detect:
  - Project name & global constants (from header comments)
  - Entity grouping (entity_id column)
  - Per-entity constants (columns that don't vary within entity)
  - Signals (columns that vary)
  - Units (from column name suffixes)

=============================================================================
FILE FORMAT
=============================================================================

MINIMAL (just data, we infer everything):
```
timestamp,flow_gpm,pressure_psi,temp_F
2024-01-01 08:00,50,120,150
2024-01-01 08:01,51,121,151
```

WITH HEADER (global constants that apply to ALL data):
```
# PROJECT: MyPumpStation
# fluid_density: 850 kg/m3
# viscosity: 5 cP
#
timestamp,flow_gpm,pressure_psi
2024-01-01,50,120
```

MULTI-ENTITY (constants as columns):
```
timestamp,entity_id,diameter_in,flow_gpm,pressure_psi
2024-01-01,P-101,4,50,120
2024-01-01,P-102,6,100,115
2024-01-02,P-101,4,51,121
```
-> diameter_in is constant per entity (auto-detected)
-> flow_gpm, pressure_psi are signals (they vary)

=============================================================================
RULES
=============================================================================

1. Global constants   -> Header comments (# key: value unit)
2. Per-entity constants -> Columns that don't change within entity_id
3. Signals            -> Columns that change over time
4. Units              -> Inferred from column suffixes (_gpm, _psi, _F, etc.)

Read the README. Or don't. We'll try our best either way.

=============================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
import re


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Signal:
    """A time-varying measurement"""
    name: str
    unit: Optional[str] = None
    category: Optional[str] = None  # pressure, flow, temperature, etc.


@dataclass
class Constant:
    """A fixed value (global or per-entity)"""
    name: str
    value: Any
    unit: Optional[str] = None
    scope: str = "global"  # "global" or "per_entity"


@dataclass
class Entity:
    """A distinct system/equipment being monitored"""
    id: str
    constants: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntakeResult:
    """Everything we extracted from the file"""
    # Metadata
    project: Optional[str] = None
    source_file: Optional[str] = None

    # Global constants (apply to all data)
    global_constants: Dict[str, Constant] = field(default_factory=dict)

    # Entities (if multi-entity data)
    entities: Dict[str, Entity] = field(default_factory=dict)
    entity_column: Optional[str] = None

    # Per-entity constant columns (same value within each entity)
    per_entity_constants: List[str] = field(default_factory=list)

    # Signal columns (vary over time)
    signals: List[Signal] = field(default_factory=list)

    # Time column
    time_column: Optional[str] = None

    # The actual data
    columns: List[str] = field(default_factory=list)
    rows: List[List[Any]] = field(default_factory=list)
    row_count: int = 0

    # Issues
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_config(self) -> Dict[str, Any]:
        """Convert intake result to PRISM config format"""
        config = {}

        if self.project:
            config['project'] = self.project

        # Window settings (default, can be overridden)
        config['window_size'] = 50
        config['window_stride'] = 25

        # Signals with units
        config['signals'] = []
        for sig in self.signals:
            sig_config = {'name': sig.name}
            if sig.unit:
                sig_config['units'] = sig.unit
            if sig.category:
                sig_config['physical_quantity'] = sig.category
            config['signals'].append(sig_config)

        # Constants
        if self.global_constants:
            config['constants'] = {}
            for name, const in self.global_constants.items():
                config['constants'][name] = const.value

        # Pipe network (if we detected diameter columns)
        pipe_signals = [s for s in self.signals if s.category == 'length' and 'diam' in s.name.lower()]
        if pipe_signals and self.per_entity_constants:
            # Build pipe network from per-entity diameter constants
            config['pipe_network'] = {'pipes': []}
            for entity_id, entity in self.entities.items():
                for const_name, const_val in entity.constants.items():
                    if 'diam' in const_name.lower():
                        pipe = {
                            'signal': entity_id,
                            'diameter': float(const_val) if isinstance(const_val, (int, float)) else const_val,
                        }
                        # Infer unit from column name
                        unit, _ = infer_unit_from_name(const_name)
                        if unit:
                            pipe['diameter_unit'] = unit
                        config['pipe_network']['pipes'].append(pipe)

        return config


# =============================================================================
# UNIT INFERENCE
# =============================================================================

# Column suffix -> (unit, category)
SUFFIX_TO_UNIT = {
    # Pressure
    '_psi': ('psi', 'pressure'),
    '_psia': ('psia', 'pressure'),  # absolute pressure
    '_psig': ('psig', 'pressure'),  # gauge pressure
    '_bar': ('bar', 'pressure'),
    '_pa': ('Pa', 'pressure'),
    '_kpa': ('kPa', 'pressure'),
    '_mpa': ('MPa', 'pressure'),
    '_atm': ('atm', 'pressure'),
    '_mmhg': ('mmHg', 'pressure'),
    '_inh2o': ('inH2O', 'pressure'),

    # Temperature
    '_f': ('degF', 'temperature'),
    '_c': ('degC', 'temperature'),
    '_k': ('K', 'temperature'),
    '_degf': ('degF', 'temperature'),
    '_degc': ('degC', 'temperature'),
    '_degr': ('degR', 'temperature'),
    '_r': ('degR', 'temperature'),  # Rankine (aerospace)

    # Volumetric Flow
    '_gpm': ('gpm', 'volumetric_flow'),
    '_lpm': ('L/min', 'volumetric_flow'),
    '_cfm': ('cfm', 'volumetric_flow'),
    '_m3s': ('m3/s', 'volumetric_flow'),
    '_m3h': ('m3/h', 'volumetric_flow'),
    '_bpd': ('bbl/d', 'volumetric_flow'),

    # Mass Flow
    '_kgs': ('kg/s', 'mass_flow'),
    '_lbs': ('lb/s', 'mass_flow'),
    '_lbm_s': ('lb/s', 'mass_flow'),  # aerospace notation
    '_lbms': ('lb/s', 'mass_flow'),

    # Velocity
    '_mps': ('m/s', 'velocity'),
    '_fps': ('ft/s', 'velocity'),
    '_mph': ('mph', 'velocity'),
    '_kph': ('km/h', 'velocity'),

    # Length
    '_m': ('m', 'length'),
    '_mm': ('mm', 'length'),
    '_cm': ('cm', 'length'),
    '_in': ('in', 'length'),
    '_ft': ('ft', 'length'),

    # Mass
    '_kg': ('kg', 'mass'),
    '_g': ('g', 'mass'),
    '_lb': ('lb', 'mass'),

    # Density
    '_kgm3': ('kg/m3', 'density'),

    # Viscosity
    '_cp': ('cP', 'dynamic_viscosity'),
    '_cst': ('cSt', 'kinematic_viscosity'),
    '_pas': ('Pa*s', 'dynamic_viscosity'),

    # Electrical
    '_v': ('V', 'voltage'),
    '_kv': ('kV', 'voltage'),
    '_mv': ('mV', 'voltage'),
    '_a': ('A', 'current'),
    '_ma': ('mA', 'current'),
    '_ohm': ('ohm', 'resistance'),
    '_w': ('W', 'power'),
    '_kw': ('kW', 'power'),
    '_mw': ('MW', 'power'),
    '_hp': ('hp', 'power'),

    # Time
    '_s': ('s', 'time'),
    '_ms': ('ms', 'time'),
    '_min': ('min', 'time'),
    '_hr': ('hr', 'time'),

    # Frequency
    '_hz': ('Hz', 'frequency'),
    '_rpm': ('rpm', 'frequency'),

    # Angle
    '_deg': ('deg', 'angle'),
    '_rad': ('rad', 'angle'),

    # Ratio/percentage
    '_pct': ('%', 'ratio'),
    '_percent': ('%', 'ratio'),

    # Energy (specific)
    '_jkg': ('J/kg', 'specific_enthalpy'),
    '_kjkg': ('kJ/kg', 'specific_enthalpy'),
    '_btulb': ('BTU/lb', 'specific_enthalpy'),
    '_btu_lb': ('BTU/lb', 'specific_enthalpy'),
}

# Patterns in column names (not just suffixes)
NAME_PATTERNS = [
    (r'temperature|temp(?!o)', 'temperature'),
    (r'pressure|press', 'pressure'),
    (r'flow(?:_?rate)?', 'volumetric_flow'),
    (r'velocity|speed', 'velocity'),
    (r'diameter|diam', 'length'),
    (r'voltage|volt', 'voltage'),
    (r'current(?!_)', 'current'),
    (r'power', 'power'),
    (r'resistance|resist', 'resistance'),
    (r'density|dens', 'density'),
    (r'viscosity|visc', 'dynamic_viscosity'),
    (r'mass(?!_flow)', 'mass'),
    (r'position|pos(?:_|$)', 'length'),
    (r'level', 'length'),
    (r'rpm|speed_rpm', 'frequency'),
    (r'efficiency|eff(?:_|$)', 'ratio'),
]

# Entity ID column patterns
ENTITY_PATTERNS = [
    r'^entity_?id$',
    r'^unit_?id$',
    r'^equipment_?id$',
    r'^asset_?id$',
    r'^machine_?id$',
    r'^engine_?id$',
    r'^pump_?id$',
    r'^pipe_?id$',
    r'^tag$',
    r'^id$',
    r'^name$',
    r'^unit$',
    r'^entity$',
    r'^asset$',
    r'^equipment$',
]

# Timestamp column patterns
TIME_PATTERNS = [
    r'^timestamp$',
    r'^time$',
    r'^datetime$',
    r'^date$',
    r'^t$',
    r'^ts$',
    r'^cycle$',
    r'^step$',
    r'^index$',
    r'_time$',
    r'_timestamp$',
]


def infer_unit_from_name(column_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Infer unit and category from column name.

    Returns:
        (unit, category) or (None, None)
    """
    name_lower = column_name.lower()

    # Check suffixes first (most specific)
    for suffix, (unit, category) in SUFFIX_TO_UNIT.items():
        if name_lower.endswith(suffix):
            return unit, category

    # Check name patterns
    for pattern, category in NAME_PATTERNS:
        if re.search(pattern, name_lower, re.IGNORECASE):
            return None, category  # Category known, unit unknown

    return None, None


def is_entity_column(column_name: str) -> bool:
    """Check if column looks like an entity identifier"""
    name_lower = column_name.lower()
    return any(re.match(p, name_lower) for p in ENTITY_PATTERNS)


def is_time_column(column_name: str) -> bool:
    """Check if column looks like a timestamp"""
    name_lower = column_name.lower()
    return any(re.match(p, name_lower) for p in TIME_PATTERNS)


# =============================================================================
# HEADER PARSING
# =============================================================================

def parse_header_line(line: str) -> Tuple[Optional[str], Optional[Any], Optional[str]]:
    """
    Parse a header comment line.

    Returns:
        (key, value, unit) or (None, None, None)
    """
    line = line.strip()
    if not line.startswith('#'):
        return None, None, None

    line = line.lstrip('#').strip()
    if not line:
        return None, None, None

    # PROJECT: name
    match = re.match(r'^PROJECT\s*[:\s]\s*(.+)$', line, re.IGNORECASE)
    if match:
        return '__project__', match.group(1).strip(), None

    # key: value unit  OR  key = value unit
    match = re.match(r'^(\w+)\s*[:\s=]\s*(.+)$', line)
    if match:
        key = match.group(1).strip().lower()
        value_str = match.group(2).strip()

        # Try to parse "123.4 kg/m3" format
        val_match = re.match(r'^([+-]?[\d.]+(?:[eE][+-]?\d+)?)\s*(.*)$', value_str)
        if val_match:
            try:
                value = float(val_match.group(1))
                unit = val_match.group(2).strip() or None
                return key, value, unit
            except ValueError:
                pass

        # Plain string value
        return key, value_str, None

    return None, None, None


def parse_headers(lines: List[str]) -> Tuple[Dict[str, Any], int]:
    """
    Parse header comments to extract metadata.

    Returns:
        (metadata_dict, first_data_line_index)
    """
    metadata = {
        '__project__': None,
        '__constants__': {},
    }

    data_start = 0

    for i, line in enumerate(lines):
        line = line.strip()

        # Empty line - skip
        if not line:
            continue

        # Not a comment - this is where data starts
        if not line.startswith('#'):
            data_start = i
            break

        # Parse the header line
        key, value, unit = parse_header_line(line)

        if key == '__project__':
            metadata['__project__'] = value
        elif key and key not in ('project', 'data', 'constants'):
            metadata['__constants__'][key] = {
                'value': value,
                'unit': unit,
            }

    return metadata, data_start


# =============================================================================
# DATA ANALYSIS
# =============================================================================

def detect_column_types(columns: List[str], rows: List[List[Any]]) -> Dict[str, str]:
    """
    Classify each column as: time, entity, constant, signal

    Returns:
        {column_name: column_type}
    """
    result = {}

    for col_idx, col_name in enumerate(columns):
        # Check if it's a time column
        if is_time_column(col_name):
            result[col_name] = 'time'
            continue

        # Check if it's an entity column
        if is_entity_column(col_name):
            result[col_name] = 'entity'
            continue

        # Get all values for this column
        values = [row[col_idx] for row in rows if col_idx < len(row)]

        # Check if all values are the same (global constant)
        unique = set(str(v) for v in values if v is not None)
        if len(unique) == 1:
            result[col_name] = 'constant'
            continue

        # Default to signal
        result[col_name] = 'signal'

    return result


def detect_per_entity_constants(columns: List[str], rows: List[List[Any]],
                                 entity_col_idx: Optional[int]) -> List[str]:
    """
    Find columns that are constant within each entity but vary across entities.

    These are per-entity constants (like pipe diameter).
    """
    if entity_col_idx is None:
        return []

    per_entity_constants = []

    for col_idx, col_name in enumerate(columns):
        if col_idx == entity_col_idx:
            continue

        # Skip time columns
        if is_time_column(col_name):
            continue

        # Group values by entity
        entity_values: Dict[str, Set] = {}
        for row in rows:
            if len(row) <= max(col_idx, entity_col_idx):
                continue

            entity_id = str(row[entity_col_idx])
            value = row[col_idx]

            if entity_id not in entity_values:
                entity_values[entity_id] = set()
            entity_values[entity_id].add(str(value) if value is not None else None)

        # Check if each entity has only one unique value
        is_per_entity_constant = all(len(vals) == 1 for vals in entity_values.values())

        # But must have different values across entities (otherwise it's global)
        all_values = set()
        for vals in entity_values.values():
            all_values.update(vals)

        if is_per_entity_constant and len(all_values) > 1:
            per_entity_constants.append(col_name)

    return per_entity_constants


# =============================================================================
# MAIN INTAKE FUNCTION
# =============================================================================

def read_file(filepath: str) -> IntakeResult:
    """
    Read a data file and extract everything we need.

    Supports:
        - CSV (with optional header comments)
        - TSV
        - Parquet (via pyarrow/pandas)
    """
    result = IntakeResult(source_file=filepath)
    path = Path(filepath)

    if not path.exists():
        result.errors.append(f"File not found: {filepath}")
        return result

    # Determine format and read
    suffix = path.suffix.lower()

    if suffix in ('.csv', '.tsv', '.txt'):
        return read_csv(filepath)
    elif suffix == '.parquet':
        return read_parquet(filepath)
    else:
        result.errors.append(f"Unknown file format: {suffix}")
        return result


def read_csv(filepath: str) -> IntakeResult:
    """Read CSV/TSV file with optional header comments"""
    result = IntakeResult(source_file=filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()

    if not all_lines:
        result.errors.append("File is empty")
        return result

    # Parse header comments
    metadata, data_start = parse_headers(all_lines)

    result.project = metadata.get('__project__')

    # Convert header constants
    for key, info in metadata.get('__constants__', {}).items():
        result.global_constants[key] = Constant(
            name=key,
            value=info['value'],
            unit=info['unit'],
            scope='global'
        )

    # Parse data section
    data_lines = all_lines[data_start:]
    if not data_lines:
        result.errors.append("No data found after headers")
        return result

    # Detect delimiter
    first_line = data_lines[0].strip()
    if '\t' in first_line:
        delimiter = '\t'
    else:
        delimiter = ','

    # Parse column headers
    result.columns = [c.strip().strip('"\'') for c in first_line.split(delimiter)]

    # Parse data rows
    for line in data_lines[1:]:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        values = []
        for val in line.split(delimiter):
            val = val.strip().strip('"\'')
            # Try to convert to number
            try:
                if '.' in val or 'e' in val.lower():
                    values.append(float(val))
                else:
                    values.append(int(val))
            except ValueError:
                values.append(val if val else None)

        result.rows.append(values)

    result.row_count = len(result.rows)

    # Analyze columns
    _analyze_columns(result)

    return result


def read_parquet(filepath: str) -> IntakeResult:
    """Read Parquet file"""
    result = IntakeResult(source_file=filepath)

    try:
        import pandas as pd
        df = pd.read_parquet(filepath)

        result.columns = list(df.columns)
        result.rows = df.values.tolist()
        result.row_count = len(df)

        # Check for metadata in parquet
        # (could be stored in schema metadata)

        # Analyze columns
        _analyze_columns(result)

    except ImportError:
        result.errors.append("pandas/pyarrow required for parquet files")
    except Exception as e:
        result.errors.append(f"Error reading parquet: {e}")

    return result


def _analyze_columns(result: IntakeResult):
    """Analyze columns to detect types, units, entities"""

    if not result.columns or not result.rows:
        return

    # Find entity column
    entity_col_idx = None
    for i, col in enumerate(result.columns):
        if is_entity_column(col):
            result.entity_column = col
            entity_col_idx = i
            break

    # Find time column
    for col in result.columns:
        if is_time_column(col):
            result.time_column = col
            break

    # Detect column types
    col_types = detect_column_types(result.columns, result.rows)

    # Find per-entity constants
    result.per_entity_constants = detect_per_entity_constants(
        result.columns, result.rows, entity_col_idx
    )

    # Update column types for per-entity constants
    for col in result.per_entity_constants:
        col_types[col] = 'per_entity_constant'

    # Build signal list
    for col in result.columns:
        col_type = col_types.get(col, 'signal')

        # Skip time and entity columns
        if col_type in ('time', 'entity'):
            continue

        unit, category = infer_unit_from_name(col)

        signal = Signal(
            name=col,
            unit=unit,
            category=category,
        )
        result.signals.append(signal)

    # Extract entities and their constants
    if entity_col_idx is not None:
        for row in result.rows:
            if len(row) <= entity_col_idx:
                continue

            entity_id = str(row[entity_col_idx])

            if entity_id not in result.entities:
                result.entities[entity_id] = Entity(id=entity_id)

            # Get per-entity constant values
            for const_col in result.per_entity_constants:
                col_idx = result.columns.index(const_col)
                if col_idx < len(row):
                    result.entities[entity_id].constants[const_col] = row[col_idx]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def summarize(result: IntakeResult) -> str:
    """Print a nice summary of what we found"""
    lines = []
    lines.append("=" * 60)
    lines.append("INTAKE SUMMARY")
    lines.append("=" * 60)

    if result.project:
        lines.append(f"Project: {result.project}")
    lines.append(f"Source:  {result.source_file}")
    lines.append(f"Rows:    {result.row_count:,}")
    lines.append(f"Columns: {len(result.columns)}")

    if result.time_column:
        lines.append(f"Time:    {result.time_column}")

    # Global constants
    if result.global_constants:
        lines.append("")
        lines.append("GLOBAL CONSTANTS:")
        for name, const in result.global_constants.items():
            unit_str = f" {const.unit}" if const.unit else ""
            lines.append(f"  {name}: {const.value}{unit_str}")

    # Entities
    if result.entities:
        lines.append("")
        lines.append(f"ENTITIES ({len(result.entities)}):")
        lines.append(f"  Grouped by: {result.entity_column}")
        for eid, entity in list(result.entities.items())[:5]:
            const_str = ", ".join(f"{k}={v}" for k, v in entity.constants.items())
            lines.append(f"  {eid}: {const_str}")
        if len(result.entities) > 5:
            lines.append(f"  ... and {len(result.entities) - 5} more")

    # Per-entity constants
    if result.per_entity_constants:
        lines.append("")
        lines.append("PER-ENTITY CONSTANTS (columns):")
        for col in result.per_entity_constants:
            unit, cat = infer_unit_from_name(col)
            unit_str = f" [{unit}]" if unit else ""
            lines.append(f"  {col}{unit_str}")

    # Signals
    signal_cols = [s for s in result.signals if s.name not in result.per_entity_constants]
    if signal_cols:
        lines.append("")
        lines.append("SIGNALS:")
        for sig in signal_cols:
            unit_str = f" [{sig.unit}]" if sig.unit else ""
            cat_str = f" ({sig.category})" if sig.category else ""
            lines.append(f"  {sig.name}{unit_str}{cat_str}")

    # Warnings/Errors
    if result.warnings:
        lines.append("")
        lines.append("[!] WARNINGS:")
        for w in result.warnings:
            lines.append(f"  {w}")

    if result.errors:
        lines.append("")
        lines.append("[X] ERRORS:")
        for e in result.errors:
            lines.append(f"  {e}")

    lines.append("=" * 60)
    return "\n".join(lines)


def intake(filepath: str) -> IntakeResult:
    """
    Main entry point: read file and return structured result.

    Alias for read_file() with a friendlier name.
    """
    return read_file(filepath)


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    import tempfile
    import os

    print("=" * 60)
    print("PRISM Intake - Self Test")
    print("=" * 60)

    # Test 1: Minimal CSV
    print("\n--- Test 1: Minimal CSV (no header) ---")
    csv1 = """timestamp,flow_gpm,pressure_psi,temp_F
2024-01-01 08:00,50,120,150
2024-01-01 08:01,51,121,151
2024-01-01 08:02,49,119,149"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv1)
        f.flush()
        result = read_file(f.name)
        print(summarize(result))
        os.unlink(f.name)

    # Test 2: With header constants
    print("\n--- Test 2: With header constants ---")
    csv2 = """# PROJECT: PumpStation_Alpha
# fluid_density: 850 kg/m3
# viscosity: 5 cP
# pipe_material: steel
#
timestamp,flow_gpm,pressure_psi
2024-01-01,50,120
2024-01-01,51,121
2024-01-01,49,119"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv2)
        f.flush()
        result = read_file(f.name)
        print(summarize(result))
        print("\nGenerated config:")
        print(result.to_config())
        os.unlink(f.name)

    # Test 3: Multi-entity with per-entity constants
    print("\n--- Test 3: Multi-entity (per-entity constants) ---")
    csv3 = """timestamp,entity_id,diameter_in,material,flow_gpm,pressure_psi
2024-01-01,P-101,4,steel,50,120
2024-01-01,P-102,6,copper,100,115
2024-01-02,P-101,4,steel,51,121
2024-01-02,P-102,6,copper,99,114
2024-01-03,P-101,4,steel,52,122
2024-01-03,P-102,6,copper,101,116"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv3)
        f.flush()
        result = read_file(f.name)
        print(summarize(result))
        os.unlink(f.name)

    # Test 4: C-MAPSS style (engine_id + cycle)
    print("\n--- Test 4: C-MAPSS style ---")
    csv4 = """# PROJECT: CMAPSS_FD001
#
engine_id,cycle,setting1,setting2,sensor1,sensor2,sensor3
1,1,0.0,0.0,518.67,641.82,1589.70
1,2,0.0,0.0,518.67,642.15,1591.82
1,3,0.0,0.0,518.67,642.35,1587.99
2,1,0.0,0.0,518.67,641.71,1590.43
2,2,0.0,0.0,518.67,642.03,1582.79"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv4)
        f.flush()
        result = read_file(f.name)
        print(summarize(result))
        os.unlink(f.name)

    print("\n[OK] All tests complete")
