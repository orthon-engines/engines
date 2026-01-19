#!/usr/bin/env python3
"""
PRISM Cohort Discovery Report — What PRISM found in your data.

This is the "give me unlabeled data, I'll tell you what you have" demo.

Shows:
- Number of cohorts discovered
- Signals in each cohort with human-readable names
- Automatic interpretation of each cohort
- Comparison to expected cohorts (if defined in YAML)

Usage:
    python -m reports.cohort_discovery_report
    python -m reports.cohort_discovery_report --domain cmapss
    python -m reports.cohort_discovery_report --output report.md
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

import polars as pl

from prism.db.parquet_store import get_parquet_path
from report_utils import (
    ReportBuilder, 
    load_domain_config, 
    translate_signal_id,
    format_number,
)


# =============================================================================
# Cohort Interpretation
# =============================================================================

def interpret_cohort(signals: List[str], config: Dict[str, Any] = None) -> str:
    """
    Attempt to interpret what a cohort represents based on signal names.
    
    Uses common industrial naming conventions and YAML categories if available.
    """
    # Check YAML categories first
    if config and 'signals' in config:
        categories = {}
        for source_name, signal_config in config['signals'].items():
            prism_id = signal_config.get('prism_id', source_name)
            if prism_id in signals:
                cat = signal_config.get('category', 'unknown')
                categories[cat] = categories.get(cat, 0) + 1
        
        if categories:
            # Return most common category
            top_cat = max(categories, key=categories.get)
            if categories[top_cat] >= len(signals) * 0.5:  # Majority rule
                return f"{top_cat.replace('_', ' ').title()} sensors"
    
    # Fallback: interpret from signal names
    signals_lower = [s.lower() for s in signals]
    signals_str = ' '.join(signals_lower)
    
    # Temperature indicators
    if any(t in signals_str for t in ['temp', 't2', 't24', 't30', 't50', 'htbleed']):
        if any(h in signals_str for h in ['t30', 't50', 'hot']):
            return "Hot section temperatures"
        if any(c in signals_str for c in ['t2', 't24', 'inlet', 'cold']):
            return "Inlet/cold section temperatures"
        return "Temperature sensors"
    
    # Pressure indicators
    if any(p in signals_str for p in ['press', 'p2', 'p15', 'p30', 'ps30', 'bpr']):
        return "Pressure measurements"
    
    # Speed/RPM indicators
    if any(n in signals_str for n in ['nf', 'nc', 'nrf', 'nrc', 'rpm', 'speed']):
        if all('f' in s.lower() for s in signals if 'n' in s.lower()):
            return "Fan spool speed"
        if all('c' in s.lower() for s in signals if 'n' in s.lower()):
            return "Core spool speed"
        return "Rotational speed sensors"
    
    # Flow indicators
    if any(w in signals_str for w in ['w31', 'w32', 'flow', 'mass']):
        return "Mass flow measurements"
    
    # Operational parameters
    if any(o in signals_str for o in ['op1', 'op2', 'op3', 'cmd', 'dmd', 'far', 'epr']):
        return "Operational/control parameters"
    
    # Failure/health indicators
    if any(r in signals_str for r in ['rul', 'health', 'fail', 'life', 'target']):
        return "Health/failure indicators"
    
    # Chemical process indicators
    if any(c in signals_str for c in ['ph', 'conc', 'react', 'feed', 'prod', 'level']):
        return "Process chemistry"
    
    # Vibration
    if any(v in signals_str for v in ['vib', 'accel', 'vel', 'disp']):
        return "Vibration sensors"
    
    return "Sensor group (unknown type)"


def compare_to_expected(
    discovered: Dict[str, List[str]], 
    expected: Dict[str, List[str]],
) -> List[str]:
    """
    Compare discovered cohorts to expected groupings from YAML.
    
    Returns list of comparison notes.
    """
    notes = []
    
    # Check each expected cohort
    for expected_name, expected_signals in expected.items():
        expected_set = set(expected_signals)
        
        # Find best matching discovered cohort
        best_match = None
        best_overlap = 0
        
        for cohort_id, discovered_signals in discovered.items():
            discovered_set = set(discovered_signals)
            overlap = len(expected_set & discovered_set)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = cohort_id
        
        if best_match:
            discovered_set = set(discovered[best_match])
            
            # Perfect match?
            if expected_set == discovered_set:
                notes.append(f"✓ {expected_name}: Perfect match (cohort {best_match})")
            else:
                missing = expected_set - discovered_set
                extra = discovered_set - expected_set
                
                if not missing and extra:
                    notes.append(f"~ {expected_name}: Cohort {best_match} includes extra signals: {extra}")
                elif missing and not extra:
                    notes.append(f"~ {expected_name}: Cohort {best_match} missing: {missing}")
                else:
                    notes.append(f"~ {expected_name}: Partial match with cohort {best_match} ({best_overlap}/{len(expected_set)} signals)")
        else:
            notes.append(f"✗ {expected_name}: No matching cohort found")
    
    return notes


# =============================================================================
# Report Generation
# =============================================================================

def generate_cohort_discovery_report(domain: str = None) -> ReportBuilder:
    """Generate cohort discovery report."""
    
    config = load_domain_config(domain) if domain else {}
    report = ReportBuilder("Cohort Discovery Report", domain=domain)
    
    # Load cohort members
    cohort_path = get_parquet_path('config', 'cohort_members')
    if not Path(cohort_path).exists():
        report.add_section("Error", f"Cohort data not found: {cohort_path}\nRun cohort discovery first.")
        return report
    
    cohort_members = pl.read_parquet(cohort_path)
    
    # Find columns
    cohort_col = None
    signal_col = None
    
    for col in ['cohort_id', 'cohort', 'cluster_id']:
        if col in cohort_members.columns:
            cohort_col = col
            break
    
    for col in ['signal_id', 'signal_type', 'sensor_id']:
        if col in cohort_members.columns:
            signal_col = col
            break
    
    if not cohort_col or not signal_col:
        report.add_section("Error", f"Could not find cohort/signal columns in {cohort_members.columns}")
        return report
    
    # ==========================================================================
    # Key Metrics
    # ==========================================================================
    n_cohorts = cohort_members[cohort_col].n_unique()
    n_signals = cohort_members[signal_col].n_unique()
    
    report.add_metric("Cohorts Discovered", n_cohorts)
    report.add_metric("Signals Clustered", n_signals)
    report.add_metric("Avg Signals/Cohort", round(n_signals / n_cohorts, 1))
    
    # ==========================================================================
    # Opening Statement
    # ==========================================================================
    report.add_section(
        "What PRISM Discovered",
        f"PRISM analyzed {n_signals} signals and discovered {n_cohorts} distinct behavioral groups.\n"
        "These cohorts represent signals that exhibit similar dynamics — likely measuring\n"
        "the same physical subsystem or responding to the same underlying process.\n\n"
        "**Key Insight:** PRISM identified these groupings from data alone, without labels."
    )
    
    # ==========================================================================
    # Cohort Details
    # ==========================================================================
    cohort_groups = (
        cohort_members
        .group_by(cohort_col)
        .agg(pl.col(signal_col).unique().alias('signals'))
        .sort(cohort_col)
    )
    
    discovered_cohorts = {}  # For comparison later
    rows = []
    
    for row in cohort_groups.iter_rows(named=True):
        cohort_id = str(row[cohort_col])
        signals = sorted(row['signals'])
        discovered_cohorts[cohort_id] = signals
        
        # Translate signal names
        human_names = [translate_signal_id(s, config) for s in signals]
        
        # Interpret
        interpretation = interpret_cohort(signals, config)
        
        # Format signal list (truncate if too long)
        if len(human_names) <= 5:
            signal_list = ", ".join(human_names)
        else:
            signal_list = ", ".join(human_names[:5]) + f" (+{len(human_names)-5} more)"
        
        rows.append([
            cohort_id,
            interpretation,
            str(len(signals)),
            signal_list,
        ])
    
    report.add_table(
        "Discovered Cohorts",
        ["Cohort", "Interpretation", "Signals", "Members"],
        rows,
        alignments=['l', 'l', 'r', 'l'],
    )
    
    # ==========================================================================
    # Detailed Signal Breakdown
    # ==========================================================================
    detail_rows = []
    for row in cohort_groups.iter_rows(named=True):
        cohort_id = str(row[cohort_col])
        signals = sorted(row['signals'])
        
        for signal in signals:
            human_name = translate_signal_id(signal, config, include_description=True)
            detail_rows.append([cohort_id, signal, human_name])
    
    report.add_table(
        "Signal-to-Cohort Mapping",
        ["Cohort", "Signal ID", "Signal Name"],
        detail_rows,
        alignments=['l', 'l', 'l'],
    )
    
    # ==========================================================================
    # Comparison to Expected (if YAML has expected_cohorts)
    # ==========================================================================
    expected = config.get('expected_cohorts', {})
    if expected:
        comparison_notes = compare_to_expected(discovered_cohorts, expected)
        report.add_section(
            "Comparison to Expected Cohorts",
            "Checking discovered cohorts against domain knowledge:\n\n" + 
            "\n".join(comparison_notes)
        )
    
    # ==========================================================================
    # Key Findings
    # ==========================================================================
    findings = []
    
    # Check for singleton cohorts
    singletons = [c for c, s in discovered_cohorts.items() if len(s) == 1]
    if singletons:
        findings.append(f"- {len(singletons)} signals are isolated (unique behavior): cohorts {singletons}")
    
    # Check for large cohorts
    large = [(c, len(s)) for c, s in discovered_cohorts.items() if len(s) > n_signals * 0.3]
    if large:
        findings.append(f"- {len(large)} cohort(s) contain >30% of signals — may indicate dominant operational mode")
    
    # Check for failure-related cohorts
    failure_cohorts = [c for c, s in discovered_cohorts.items() 
                       if any('rul' in sig.lower() or 'fail' in sig.lower() or 'health' in sig.lower() 
                              for sig in s)]
    if failure_cohorts:
        findings.append(f"- {len(failure_cohorts)} cohort(s) contain failure/health indicators — suggests distinct failure patterns")
    
    if findings:
        report.add_section("Key Findings", "\n".join(findings))
    
    # ==========================================================================
    # The Pitch
    # ==========================================================================
    report.add_section(
        "What This Means",
        "PRISM discovered the physical structure of your system from unlabeled data.\n\n"
        "\"Give me your mystery sensors — I'll tell you what you have.\"\n\n"
        "These cohorts can now be used to:\n"
        "- Validate sensor installations (signals that should cluster together, do they?)\n"
        "- Identify redundant sensors (same cohort = measuring same thing)\n"
        "- Focus monitoring on leading indicators (which cohort changes first before failure?)\n"
        "- Simplify dashboards (one metric per cohort instead of per sensor)"
    )
    
    return report


def main():
    parser = argparse.ArgumentParser(description='PRISM Cohort Discovery Report')
    parser.add_argument('--domain', type=str, default=None, help='Domain name')
    parser.add_argument('--output', type=str, default=None, help='Output file (md or json)')
    args = parser.parse_args()
    
    if not args.domain:
        import os
        args.domain = os.environ.get('PRISM_DOMAIN')
    
    report = generate_cohort_discovery_report(args.domain)
    
    if args.output:
        if args.output.endswith('.json'):
            report.save_json(args.output)
        else:
            report.save_markdown(args.output)
        print(f"Report saved to: {args.output}")
    else:
        print(report.to_text())


if __name__ == "__main__":
    main()
