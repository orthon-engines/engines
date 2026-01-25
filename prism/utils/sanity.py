"""
PRISM Data Sanity Checker — "That's Not What You Think It Is"

Auto-detects when data doesn't match what the column name claims.
Catches unit mismatches, impossible values, and obvious mistakes
BEFORE you waste compute on garbage.

The philosophy: Be helpful first, sarcastic second.

Usage:
    from prism.sanity import check_dataframe, check_column

    issues = check_dataframe(df)
    for issue in issues:
        print(issue.message)
        print(issue.suggestion)
        if issue.humor:
            print(f"  ({issue.humor})")
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import re
import math


class Severity(Enum):
    """How bad is the problem?"""
    INFO = "info"           # FYI, might be fine
    WARNING = "warning"     # Probably wrong
    ERROR = "error"         # Definitely wrong
    ABSURD = "absurd"       # So wrong it's funny


@dataclass
class SanityIssue:
    """A detected data problem"""
    column: str
    severity: Severity
    code: str               # Machine-readable code
    message: str            # Human-readable explanation
    suggestion: str         # How to fix it
    humor: Optional[str]    # Optional comedic relief
    details: Dict[str, Any] # Debug info


# =============================================================================
# PHYSICAL REASONABLENESS BOUNDS
# =============================================================================

# What's physically possible? (in SI units)
PHYSICAL_BOUNDS = {
    # Temperature (Kelvin)
    'temperature': {
        'min': 0,           # Absolute zero
        'max': 1e8,         # Core of the sun is ~1.5e7 K
        'typical_min': 200,  # -73C, cold but earthly
        'typical_max': 500,  # 227C, hot but industrial
        'suspiciously_human': (309, 311),  # 98-100F body temp
    },

    # Pressure (Pascals)
    'pressure': {
        'min': 0,           # Vacuum
        'max': 1e12,        # Diamond anvil territory
        'typical_min': 1e4,  # 0.1 atm
        'typical_max': 1e8,  # 1000 atm, deep sea/industrial
        'suspiciously_atmospheric': (99000, 103000),  # ~1 atm
    },

    # Flow rate (m3/s)
    'volumetric_flow': {
        'min': 0,
        'max': 1e6,         # Amazon river is ~2e5 m3/s
        'typical_min': 1e-8, # Tiny lab flow
        'typical_max': 1,    # Industrial scale
        'glacier_pace': 1e-10,  # Suspiciously slow
    },

    # Velocity (m/s)
    'velocity': {
        'min': -1e8,        # Negative OK (direction)
        'max': 3e8,         # Speed of light
        'typical_min': -100,
        'typical_max': 100,
        'supersonic': 343,  # Mach 1
    },

    # Length/diameter (meters)
    'length': {
        'min': 0,
        'max': 1e12,        # ~Saturn orbit
        'typical_min': 1e-6, # Microns
        'typical_max': 1e3,  # Kilometers
        'sus_ratio': (0.01, 0.99),  # Looks like a percentage
    },

    # Density (kg/m3)
    'density': {
        'min': 0,
        'max': 2.3e17,      # Neutron star
        'typical_min': 0.1,  # Light gas
        'typical_max': 2e4,  # Heavy metals
        'water_ish': (990, 1010),
    },

    # Viscosity (Pa*s)
    'dynamic_viscosity': {
        'min': 0,
        'max': 1e9,         # Pitch drop experiment
        'typical_min': 1e-6, # Air
        'typical_max': 1e3,  # Very thick fluids
        'water_ish': (0.0008, 0.0012),
    },

    # Electrical
    'voltage': {
        'min': -1e6,
        'max': 1e6,
        'typical_min': -500,
        'typical_max': 500,
        'household': (110, 130, 220, 240),  # Common values
    },

    'current': {
        'min': -1e6,
        'max': 1e6,
        'typical_min': -100,
        'typical_max': 100,
    },

    'resistance': {
        'min': 0,
        'max': 1e15,        # Good insulators
        'typical_min': 1e-3,
        'typical_max': 1e9,
    },

    # Power (Watts)
    'power': {
        'min': -1e12,       # Can be negative (regenerative)
        'max': 1e12,
        'typical_min': -1e6,
        'typical_max': 1e6,
    },

    # Dimensionless ratios
    'ratio': {
        'min': 0,
        'max': 1,
        'typical_min': 0,
        'typical_max': 1,
    },

    'percentage': {
        'min': 0,
        'max': 100,
        'typical_min': 0,
        'typical_max': 100,
    },
}


# =============================================================================
# COLUMN NAME PATTERNS
# =============================================================================

# What unit type does this column name suggest?
NAME_PATTERNS = [
    # Temperature
    (r'temp|temperature|deg[CF]', 'temperature', 'K'),
    (r'_[CF]$|_kelvin|_celsius|_fahrenheit', 'temperature', 'K'),

    # Pressure
    (r'press|pressure|psi|bar|atm|pascal|kpa|mpa', 'pressure', 'Pa'),
    (r'_psi$|_bar$|_pa$|_kpa$', 'pressure', 'Pa'),

    # Flow
    (r'flow|gpm|cfm|lpm|m3.?s|bbl|scfm', 'volumetric_flow', 'm3/s'),
    (r'_gpm$|_cfm$|_lpm$', 'volumetric_flow', 'm3/s'),

    # Velocity
    (r'veloc|speed|mph|fps|mps|m.?s', 'velocity', 'm/s'),
    (r'_velocity$|_speed$', 'velocity', 'm/s'),

    # Length/diameter/position
    (r'diam|length|width|height|depth|thick|position|distance', 'length', 'm'),
    (r'_in$|_ft$|_m$|_mm$|_cm$', 'length', 'm'),

    # Density
    (r'dens|density|rho', 'density', 'kg/m3'),

    # Viscosity
    (r'visc|viscosity|cp$|cst$|poise', 'dynamic_viscosity', 'Pa*s'),

    # Electrical
    (r'volt|voltage|potential|_v$|_kv$', 'voltage', 'V'),
    (r'current|_amp|_a$|_ma$', 'current', 'A'),
    (r'resist|_ohm', 'resistance', 'ohm'),
    (r'power|watt|_w$|_kw$|_mw$', 'power', 'W'),

    # Ratios/percentages
    (r'ratio|fraction|efficiency|yield', 'ratio', ''),
    (r'percent|pct|%', 'percentage', '%'),
    (r'_pct$|_percent$', 'percentage', '%'),
]


def infer_type_from_name(column_name: str) -> Optional[Tuple[str, str]]:
    """
    Guess the physical quantity type from column name.

    Returns:
        (quantity_type, expected_unit) or None
    """
    name_lower = column_name.lower()

    for pattern, qty_type, unit in NAME_PATTERNS:
        if re.search(pattern, name_lower, re.IGNORECASE):
            return qty_type, unit

    return None


# =============================================================================
# VALUE ANALYSIS
# =============================================================================

def analyze_values(values: List[float]) -> Dict[str, Any]:
    """Get statistical summary of values"""
    clean = [v for v in values if v is not None and not math.isnan(v) and not math.isinf(v)]

    if not clean:
        return {'empty': True}

    sorted_vals = sorted(clean)
    n = len(clean)

    return {
        'empty': False,
        'count': n,
        'min': min(clean),
        'max': max(clean),
        'mean': sum(clean) / n,
        'median': sorted_vals[n // 2],
        'p05': sorted_vals[int(n * 0.05)] if n > 20 else sorted_vals[0],
        'p95': sorted_vals[int(n * 0.95)] if n > 20 else sorted_vals[-1],
        'range': max(clean) - min(clean),
        'all_positive': min(clean) >= 0,
        'all_integers': all(v == int(v) for v in clean),
        'all_zero_to_one': all(0 <= v <= 1 for v in clean),
        'all_zero_to_hundred': all(0 <= v <= 100 for v in clean),
        'has_negatives': min(clean) < 0,
        'std': (sum((v - sum(clean)/n)**2 for v in clean) / n) ** 0.5 if n > 1 else 0,
        'unique_count': len(set(clean)),
    }


def detect_data_type(values: List[Any]) -> str:
    """Detect if values are numeric, string, mixed, etc."""
    sample = values[:1000]  # Check first 1000

    types = set()
    for v in sample:
        if v is None:
            continue
        elif isinstance(v, bool):
            types.add('bool')
        elif isinstance(v, int):
            types.add('int')
        elif isinstance(v, float):
            types.add('float')
        elif isinstance(v, str):
            types.add('str')
        else:
            types.add('other')

    if types == {'int'} or types == {'float'} or types == {'int', 'float'}:
        return 'numeric'
    elif types == {'str'}:
        return 'string'
    elif types == {'bool'}:
        return 'boolean'
    elif 'str' in types and ('int' in types or 'float' in types):
        return 'mixed'
    else:
        return 'unknown'


# =============================================================================
# SANITY CHECKS
# =============================================================================

def check_column(column_name: str, values: List[Any],
                 declared_unit: Optional[str] = None) -> List[SanityIssue]:
    """
    Check a single column for sanity issues.

    Args:
        column_name: Name of the column
        values: List of values
        declared_unit: Unit the user claims this is (optional)
    """
    issues = []

    # Check data type
    dtype = detect_data_type(values)

    # Infer what this column SHOULD be from its name
    inferred = infer_type_from_name(column_name)

    # Handle non-numeric data
    if dtype == 'string':
        # Check for bra sizes, zip codes, etc.
        sample = [v for v in values[:100] if v is not None]

        # Bra size pattern: number + letter(s)
        bra_pattern = r'^\d{2,3}[A-G]{1,2}$'
        bra_count = sum(1 for v in sample if re.match(bra_pattern, str(v)))

        if bra_count > len(sample) * 0.5:
            issues.append(SanityIssue(
                column=column_name,
                severity=Severity.ABSURD,
                code='BRA_SIZES_DETECTED',
                message=f"Column '{column_name}' appears to contain bra sizes.",
                suggestion="This is probably not the industrial sensor data you meant to upload.",
                humor="Reynolds number for underwire: undefined. Please check your file.",
                details={'pattern_matches': bra_count, 'sample_size': len(sample)}
            ))
            return issues  # No point checking further

        # If column name suggests numeric but data is strings
        if inferred:
            issues.append(SanityIssue(
                column=column_name,
                severity=Severity.ERROR,
                code='EXPECTED_NUMERIC',
                message=f"Column '{column_name}' looks like it should be {inferred[0]}, but contains text.",
                suggestion=f"Check your data export. This column should contain numeric {inferred[0]} values.",
                humor=None,
                details={'data_type': dtype, 'expected_type': inferred[0]}
            ))
        return issues

    if dtype == 'mixed':
        issues.append(SanityIssue(
            column=column_name,
            severity=Severity.ERROR,
            code='MIXED_TYPES',
            message=f"Column '{column_name}' contains both numbers and text.",
            suggestion="Clean your data. Pick one type.",
            humor="Schrodinger's column: simultaneously numeric and not.",
            details={'data_type': dtype}
        ))
        return issues

    # Numeric analysis
    numeric_values = [float(v) for v in values if v is not None
                      and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))]

    if not numeric_values:
        issues.append(SanityIssue(
            column=column_name,
            severity=Severity.WARNING,
            code='ALL_NULL',
            message=f"Column '{column_name}' contains no valid numeric data.",
            suggestion="Check your data source. This column is empty or all NaN.",
            humor=None,
            details={'count': len(values)}
        ))
        return issues

    stats = analyze_values(numeric_values)

    # Check against physical bounds if we know the type
    if inferred:
        qty_type, expected_unit = inferred
        bounds = PHYSICAL_BOUNDS.get(qty_type, {})

        # Check for impossible values
        if 'min' in bounds and stats['min'] < bounds['min']:
            issues.append(SanityIssue(
                column=column_name,
                severity=Severity.ERROR,
                code='BELOW_PHYSICAL_MIN',
                message=f"Column '{column_name}' has values below physical minimum.",
                suggestion=f"Minimum value {stats['min']:.4g} is below {bounds['min']} for {qty_type}.",
                humor="Unless you've discovered negative temperature, check your signs." if qty_type == 'temperature' else None,
                details={'min_value': stats['min'], 'physical_min': bounds['min']}
            ))

        if 'max' in bounds and stats['max'] > bounds['max']:
            issues.append(SanityIssue(
                column=column_name,
                severity=Severity.ERROR,
                code='ABOVE_PHYSICAL_MAX',
                message=f"Column '{column_name}' has values above physical maximum.",
                suggestion=f"Maximum value {stats['max']:.4g} exceeds {bounds['max']} for {qty_type}.",
                humor=None,
                details={'max_value': stats['max'], 'physical_max': bounds['max']}
            ))

        # Check for suspicious patterns
        if qty_type == 'temperature' and 'suspiciously_human' in bounds:
            low, high = bounds['suspiciously_human']
            if low <= stats['mean'] <= high and stats['range'] < 5:
                issues.append(SanityIssue(
                    column=column_name,
                    severity=Severity.WARNING,
                    code='BODY_TEMPERATURE',
                    message=f"Column '{column_name}' values look like body temperatures, not process temperatures.",
                    suggestion="Mean value {:.1f} with range {:.1f} looks like human body temp data.".format(stats['mean'], stats['range']),
                    humor="Is your pump running a fever? These look like patient vitals, not industrial data.",
                    details={'mean': stats['mean'], 'range': stats['range']}
                ))

        if qty_type == 'volumetric_flow' and 'glacier_pace' in bounds:
            if stats['max'] < bounds['glacier_pace']:
                issues.append(SanityIssue(
                    column=column_name,
                    severity=Severity.WARNING,
                    code='IMPOSSIBLY_SLOW_FLOW',
                    message=f"Column '{column_name}' flow rates are impossibly slow.",
                    suggestion="Values suggest flow slower than continental drift. Check units or sensor.",
                    humor="At this rate, your fluid will arrive sometime next ice age.",
                    details={'max_flow': stats['max']}
                ))

        # Check for ratio/percentage confusion
        if qty_type not in ('ratio', 'percentage'):
            if stats['all_zero_to_one'] and stats['range'] < 1:
                issues.append(SanityIssue(
                    column=column_name,
                    severity=Severity.WARNING,
                    code='LOOKS_LIKE_RATIO',
                    message=f"Column '{column_name}' values are all between 0-1.",
                    suggestion=f"Name suggests {qty_type}, but values look like ratios or percentages. Unit mismatch?",
                    humor=None,
                    details={'min': stats['min'], 'max': stats['max']}
                ))
            elif stats['all_zero_to_hundred'] and qty_type not in ('temperature',):
                # Could be percentage masquerading as actual values
                if stats['max'] <= 100 and stats['min'] >= 0:
                    issues.append(SanityIssue(
                        column=column_name,
                        severity=Severity.INFO,
                        code='COULD_BE_PERCENTAGE',
                        message=f"Column '{column_name}' values are all 0-100.",
                        suggestion=f"Could be percentages? Verify units.",
                        humor=None,
                        details={'min': stats['min'], 'max': stats['max']}
                    ))

    # Check for all-constant data (sensor failure?)
    if stats['std'] == 0 and stats['count'] > 10:
        issues.append(SanityIssue(
            column=column_name,
            severity=Severity.WARNING,
            code='CONSTANT_VALUE',
            message=f"Column '{column_name}' has the same value for all {stats['count']} rows.",
            suggestion="Stuck sensor? Placeholder data? Check your data source.",
            humor="This sensor has achieved enlightenment: perfect constancy. Or it's broken.",
            details={'constant_value': stats['mean'], 'count': stats['count']}
        ))

    # Check for suspiciously low variance
    elif stats['count'] > 100 and stats['std'] > 0:
        cv = stats['std'] / abs(stats['mean']) if stats['mean'] != 0 else float('inf')
        if cv < 0.0001:
            issues.append(SanityIssue(
                column=column_name,
                severity=Severity.INFO,
                code='VERY_LOW_VARIANCE',
                message=f"Column '{column_name}' has extremely low variance.",
                suggestion=f"Coefficient of variation is {cv:.2e}. Normal for this signal?",
                humor=None,
                details={'cv': cv, 'std': stats['std'], 'mean': stats['mean']}
            ))

    # Check for too many unique values (possible ID column)
    if stats['unique_count'] == stats['count'] and stats['count'] > 100:
        if stats['all_integers']:
            issues.append(SanityIssue(
                column=column_name,
                severity=Severity.INFO,
                code='POSSIBLY_ID_COLUMN',
                message=f"Column '{column_name}' has all unique integer values.",
                suggestion="This might be an ID or index column, not a measurement.",
                humor=None,
                details={'unique_count': stats['unique_count']}
            ))

    return issues


def check_dataframe(df, column_units: Optional[Dict[str, str]] = None) -> List[SanityIssue]:
    """
    Check entire dataframe for sanity issues.

    Args:
        df: pandas DataFrame or dict of {column: values}
        column_units: Optional dict mapping column names to declared units

    Returns:
        List of SanityIssue objects
    """
    all_issues = []
    column_units = column_units or {}

    # Handle DataFrame (pandas or polars)
    if hasattr(df, 'columns'):
        columns = {}
        for col in df.columns:
            series = df[col]
            # Polars uses .to_list(), pandas uses .tolist()
            if hasattr(series, 'to_list'):
                columns[col] = series.to_list()
            else:
                columns[col] = series.tolist()
    else:
        columns = df

    for col_name, values in columns.items():
        declared_unit = column_units.get(col_name)
        issues = check_column(col_name, values, declared_unit)
        all_issues.extend(issues)

    # Cross-column checks
    all_issues.extend(check_cross_column(columns))

    return all_issues


def check_cross_column(columns: Dict[str, List]) -> List[SanityIssue]:
    """Check for issues that span multiple columns"""
    issues = []

    # Find temperature columns
    temp_cols = [c for c in columns.keys()
                 if infer_type_from_name(c) and infer_type_from_name(c)[0] == 'temperature']

    # Check for Fahrenheit/Celsius confusion between columns
    if len(temp_cols) >= 2:
        for i, col1 in enumerate(temp_cols):
            for col2 in temp_cols[i+1:]:
                vals1 = [v for v in columns[col1] if v is not None and not (isinstance(v, float) and math.isnan(v))]
                vals2 = [v for v in columns[col2] if v is not None and not (isinstance(v, float) and math.isnan(v))]

                if vals1 and vals2:
                    mean1, mean2 = sum(vals1)/len(vals1), sum(vals2)/len(vals2)

                    # Check for F/C confusion: if ratio is ~1.8 and diff is ~32-ish scaled
                    if mean1 > 0 and mean2 > 0:
                        ratio = mean1 / mean2 if mean1 > mean2 else mean2 / mean1
                        if 1.7 < ratio < 1.9:
                            issues.append(SanityIssue(
                                column=f"{col1} vs {col2}",
                                severity=Severity.WARNING,
                                code='POSSIBLE_F_C_CONFUSION',
                                message=f"Temperature columns '{col1}' and '{col2}' might be in different units.",
                                suggestion="One might be Fahrenheit, the other Celsius. Check your data source.",
                                humor=None,
                                details={'ratio': ratio, 'mean1': mean1, 'mean2': mean2}
                            ))

    return issues


# =============================================================================
# FUN ERROR MESSAGES
# =============================================================================

HUMOR_BY_CODE = {
    'BRA_SIZES_DETECTED': [
        "Reynolds number for underwire: undefined.",
        "Victoria's Secret does not have industrial applications. Yet.",
        "This data is unsupported. Much like the garments it describes.",
    ],
    'BODY_TEMPERATURE': [
        "Is your pump running a fever?",
        "These look like patient vitals, not process data.",
        "Your heat exchanger appears to be human.",
    ],
    'IMPOSSIBLY_SLOW_FLOW': [
        "At this rate, your fluid will arrive sometime next ice age.",
        "Slower than a glacier with commitment issues.",
        "This flow rate violates no physics laws, just patience.",
    ],
    'CONSTANT_VALUE': [
        "This sensor has achieved enlightenment: perfect constancy.",
        "Flatline. Check the patient. I mean, sensor.",
        "Either very stable process or very dead instrument.",
    ],
    'MIXED_TYPES': [
        "Schrodinger's column: simultaneously numeric and not.",
        "Pick a type. Any type. But just one.",
    ],
}


def get_random_humor(code: str) -> Optional[str]:
    """Get a random humorous message for an error code"""
    import random
    options = HUMOR_BY_CODE.get(code, [])
    return random.choice(options) if options else None


# =============================================================================
# REPORTING
# =============================================================================

def format_report(issues: List[SanityIssue], verbose: bool = False) -> str:
    """Format issues into a human-readable report"""
    if not issues:
        return "All columns passed sanity checks."

    lines = ["=" * 60, "DATA SANITY REPORT", "=" * 60, ""]

    # Group by severity
    by_severity = {}
    for issue in issues:
        by_severity.setdefault(issue.severity, []).append(issue)

    # Report in order of severity
    for severity in [Severity.ABSURD, Severity.ERROR, Severity.WARNING, Severity.INFO]:
        if severity not in by_severity:
            continue

        emoji = {'absurd': '[!?]', 'error': '[X]', 'warning': '[!]', 'info': '[i]'}[severity.value]
        lines.append(f"{emoji} {severity.value.upper()} ({len(by_severity[severity])})")
        lines.append("-" * 40)

        for issue in by_severity[severity]:
            lines.append(f"  [{issue.column}] {issue.message}")
            lines.append(f"    -> {issue.suggestion}")
            if issue.humor:
                lines.append(f"    ({issue.humor})")
            if verbose:
                lines.append(f"    Code: {issue.code}")
                lines.append(f"    Details: {issue.details}")
            lines.append("")

    lines.append("=" * 60)

    absurd = len(by_severity.get(Severity.ABSURD, []))
    errors = len(by_severity.get(Severity.ERROR, []))
    warnings = len(by_severity.get(Severity.WARNING, []))

    if absurd > 0:
        lines.append("[!?] This data is... creative. Please verify your upload.")
    elif errors > 0:
        lines.append(f"[X] {errors} error(s) found. Fix before proceeding.")
    elif warnings > 0:
        lines.append(f"[!] {warnings} warning(s). Review recommended.")

    return "\n".join(lines)


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PRISM Sanity Checker - Self Test")
    print("=" * 60)

    # Test 1: Normal data
    print("\n--- Test 1: Normal pressure data ---")
    issues = check_column("pressure_psi", [100, 105, 98, 102, 99, 101, 103])
    print(f"Issues found: {len(issues)}")

    # Test 2: Body temperatures masquerading as process temps
    print("\n--- Test 2: Body temperature confusion ---")
    issues = check_column("temperature_F", [98.6, 99.1, 98.4, 98.8, 99.0, 98.5])
    for issue in issues:
        print(f"  {issue.severity.value}: {issue.message}")
        if issue.humor:
            print(f"    ({issue.humor})")

    # Test 3: Bra sizes
    print("\n--- Test 3: Bra sizes ---")
    issues = check_column("diameter_in", ["34B", "36C", "32A", "38D", "34C"])
    for issue in issues:
        print(f"  {issue.severity.value}: {issue.message}")
        if issue.humor:
            print(f"    ({issue.humor})")

    # Test 4: Constant value (dead sensor)
    print("\n--- Test 4: Stuck sensor ---")
    issues = check_column("flow_gpm", [50.0] * 100)
    for issue in issues:
        print(f"  {issue.severity.value}: {issue.message}")

    # Test 5: Ratio confusion
    print("\n--- Test 5: Ratio vs actual value ---")
    issues = check_column("pressure_psi", [0.12, 0.15, 0.11, 0.14, 0.13])
    for issue in issues:
        print(f"  {issue.severity.value}: {issue.message}")

    # Test 6: Impossibly slow flow
    print("\n--- Test 6: Glacier-speed flow ---")
    issues = check_column("flow_gpm", [1e-15, 2e-15, 1.5e-15])
    for issue in issues:
        print(f"  {issue.severity.value}: {issue.message}")
        if issue.humor:
            print(f"    ({issue.humor})")

    # Test 7: Full dataframe check
    print("\n--- Test 7: Full dataframe report ---")
    test_df = {
        'timestamp': list(range(100)),
        'temperature_F': [98.6 + i*0.01 for i in range(100)],  # Body temps
        'pressure_psi': [0.5] * 100,  # Looks like ratio
        'flow_gpm': [100 + i for i in range(100)],  # Normal
        'diameter_in': ["34B", "36C"] * 50,  # Bra sizes
    }
    issues = check_dataframe(test_df)
    print(format_report(issues))
