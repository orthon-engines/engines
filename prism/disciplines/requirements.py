"""
Discipline Requirements Checker

Validates that data meets discipline requirements and generates help text.
"""

from typing import Dict, List, Any, Set, Optional
from prism.disciplines.registry import DISCIPLINES


def get_requirements_text(discipline_key: str) -> str:
    """
    Generate human-readable requirements for ORTHON UI.

    Parameters
    ----------
    discipline_key : str
        Key from DISCIPLINES dict

    Returns
    -------
    str
        Markdown-formatted requirements text
    """
    disc = DISCIPLINES.get(discipline_key)
    if not disc:
        return "Unknown discipline"

    lines = [
        f"# {disc['name']}",
        f"{disc['description']}",
        "",
    ]

    # Signals
    signals = disc.get('signals', {})
    if signals.get('required'):
        lines.append("**Required signals:**")
        for s in signals['required']:
            lines.append(f"  - {s}")
    if signals.get('required_any'):
        lines.append("**Required (at least one):**")
        for s in signals['required_any']:
            lines.append(f"  - {s}")
    if signals.get('optional'):
        lines.append("**Optional signals:**")
        for s in signals['optional']:
            lines.append(f"  - {s}")

    lines.append("")

    # Constants
    constants = disc.get('constants', {})
    if constants.get('required'):
        lines.append("**Required constants:**")
        for name, info in constants['required'].items():
            desc = info.get('description', '')
            desc_str = f" - {desc}" if desc else ""
            lines.append(f"  - {name} [{info['unit']}]{desc_str}")
    if constants.get('optional'):
        lines.append("**Optional constants (enable more engines):**")
        for name, info in constants['optional'].items():
            default = f" (default: {info['default']})" if 'default' in info else ""
            desc = info.get('description', '')
            desc_str = f" - {desc}" if desc else ""
            lines.append(f"  - {name} [{info['unit']}]{default}{desc_str}")

    lines.append("")

    # Subdisciplines
    if disc.get('subdisciplines'):
        lines.append("**Subdisciplines:**")
        for sub_key, sub in disc['subdisciplines'].items():
            lines.append(f"\n### {sub['name']}")

            sub_signals = sub.get('signals', {})
            if sub_signals.get('required'):
                lines.append("Required signals: " + ", ".join(sub_signals['required']))
            if sub_signals.get('required_any'):
                lines.append("Required (at least one): " + ", ".join(sub_signals['required_any']))

            sub_constants = sub.get('constants', {})
            if sub_constants.get('required'):
                const_list = [f"{k} [{v['unit']}]" for k, v in sub_constants['required'].items()]
                lines.append("Required constants: " + ", ".join(const_list))

            lines.append("Engines: " + ", ".join(sub.get('engines', [])))

    lines.append("")

    # Main engines
    if disc.get('engines'):
        lines.append("**Engines:**")
        for engine in disc['engines']:
            lines.append(f"  - {engine}")

    return "\n".join(lines)


def check_requirements(
    discipline_key: str,
    available_signals: Optional[Set[str]] = None,
    available_constants: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Check which requirements are met for a discipline.

    Parameters
    ----------
    discipline_key : str
        Discipline name
    available_signals : set, optional
        Set of signal names in the data
    available_constants : dict, optional
        Dict of constant name -> value

    Returns
    -------
    dict
        valid: bool - whether minimum requirements met
        missing_signals: list - signals still needed
        missing_constants: list - constants still needed
        available_engines: list - engines that can run
        unavailable_engines: list - engines that can't run
        warnings: list - non-fatal issues
    """
    disc = DISCIPLINES.get(discipline_key)
    if not disc:
        return {"valid": False, "error": "Unknown discipline"}

    available_signals = available_signals or set()
    available_constants = available_constants or {}

    result = {
        "valid": True,
        "missing_signals": [],
        "missing_constants": [],
        "available_engines": [],
        "unavailable_engines": [],
        "warnings": [],
    }

    # Check signals
    signals = disc.get('signals', {})

    if signals.get('required'):
        for s in signals['required']:
            if s not in available_signals:
                result['missing_signals'].append(s)
                result['valid'] = False

    if signals.get('required_any'):
        if not any(s in available_signals for s in signals['required_any']):
            result['missing_signals'].append(f"one of: {', '.join(signals['required_any'])}")
            result['valid'] = False

    # Check constants
    constants = disc.get('constants', {})

    if constants.get('required'):
        for name in constants['required']:
            if name not in available_constants:
                result['missing_constants'].append(name)
                result['valid'] = False

    # Check optional constants for warnings
    if constants.get('optional'):
        missing_optional = [
            name for name in constants['optional']
            if name not in available_constants and 'default' not in constants['optional'][name]
        ]
        if missing_optional:
            result['warnings'].append(
                f"Optional constants not provided (some engines may not run): {', '.join(missing_optional)}"
            )

    # Check subdisciplines
    if disc.get('subdisciplines'):
        for sub_key, sub in disc['subdisciplines'].items():
            sub_signals = sub.get('signals', {})
            sub_constants = sub.get('constants', {})

            sub_valid = True

            # Check subdiscipline required signals
            if sub_signals.get('required'):
                for s in sub_signals['required']:
                    if s not in available_signals:
                        sub_valid = False

            if sub_signals.get('required_any'):
                if not any(s in available_signals for s in sub_signals['required_any']):
                    sub_valid = False

            # Check subdiscipline required constants
            if sub_constants.get('required'):
                for name in sub_constants['required']:
                    if name not in available_constants:
                        sub_valid = False

            if sub_valid:
                result['available_engines'].extend(sub.get('engines', []))
            else:
                result['unavailable_engines'].extend(sub.get('engines', []))
                result['warnings'].append(
                    f"Subdiscipline '{sub['name']}' requirements not met"
                )

    # Main discipline engines (always available if valid)
    if result['valid']:
        result['available_engines'].extend(disc.get('engines', []))
    else:
        result['unavailable_engines'].extend(disc.get('engines', []))

    # Deduplicate
    result['available_engines'] = list(set(result['available_engines']))
    result['unavailable_engines'] = list(set(result['unavailable_engines']))

    return result


def get_available_engines(
    discipline_key: str,
    available_signals: Optional[Set[str]] = None,
    available_constants: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Get list of engines that can run given available data.

    Parameters
    ----------
    discipline_key : str
        Discipline name
    available_signals : set
        Available signal names
    available_constants : dict
        Available constants

    Returns
    -------
    list
        Engine names that can run
    """
    result = check_requirements(discipline_key, available_signals, available_constants)
    return result.get('available_engines', [])


def get_all_requirements() -> Dict[str, Dict[str, Any]]:
    """
    Get requirements for all disciplines.

    Returns
    -------
    dict
        discipline_name -> requirements dict
    """
    all_reqs = {}
    for key in DISCIPLINES:
        all_reqs[key] = {
            'text': get_requirements_text(key),
            'signals': DISCIPLINES[key].get('signals', {}),
            'constants': DISCIPLINES[key].get('constants', {}),
            'engines': DISCIPLINES[key].get('engines', []),
        }
    return all_reqs


def suggest_discipline(
    available_signals: Set[str],
    available_constants: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Suggest disciplines based on available data.

    Parameters
    ----------
    available_signals : set
        Available signal names
    available_constants : dict
        Available constants

    Returns
    -------
    list
        Discipline names that could run, sorted by how many requirements are met
    """
    suggestions = []

    for key, disc in DISCIPLINES.items():
        result = check_requirements(key, available_signals, available_constants)
        if result['valid']:
            n_engines = len(result['available_engines'])
            suggestions.append((key, n_engines))

    # Sort by number of available engines (descending)
    suggestions.sort(key=lambda x: x[1], reverse=True)

    return [s[0] for s in suggestions]
