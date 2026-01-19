#!/usr/bin/env python3
"""
PRISM State Report — Velocity, acceleration, and temporal dynamics.

Shows:
- State velocity (rate of change in coupling)
- Acceleration (change in velocity — critical for prediction)
- Trajectory classification (stable, drifting, accelerating)
- Early warning indicators

Usage:
    python -m reports.state_report
    python -m reports.state_report --domain cmapss
    python -m reports.state_report --output report.md
"""

import argparse
from pathlib import Path
from typing import Dict, Any

import polars as pl

from prism.db.parquet_store import get_parquet_path
from report_utils import (
    ReportBuilder, 
    load_domain_config, 
    translate_signal_id,
    format_number,
    format_percentage,
    get_time_unit,
)


# =============================================================================
# State Interpretation
# =============================================================================

def classify_trajectory(velocity: float, acceleration: float) -> str:
    """Classify trajectory based on velocity and acceleration."""
    if velocity is None or acceleration is None:
        return "UNKNOWN"
    
    # Magnitude thresholds (these should be calibrated per domain)
    v_threshold = 0.1
    a_threshold = 0.05
    
    if abs(velocity) < v_threshold and abs(acceleration) < a_threshold:
        return "STABLE"
    
    if velocity > 0 and acceleration > 0:
        return "ACCELERATING_DEGRADATION"
    
    if velocity > 0 and acceleration <= 0:
        return "DRIFTING_DEGRADATION"
    
    if velocity < 0 and acceleration < 0:
        return "ACCELERATING_RECOVERY"
    
    if velocity < 0 and acceleration >= 0:
        return "STABILIZING"
    
    return "TRANSITIONAL"


def interpret_trajectory(classification: str) -> str:
    """Provide human interpretation of trajectory."""
    interpretations = {
        "STABLE": "System in steady state. No significant change detected.",
        "ACCELERATING_DEGRADATION": "⚠️ CRITICAL: Degradation rate is increasing. Failure approaching faster.",
        "DRIFTING_DEGRADATION": "CAUTION: Slow degradation. Monitor closely.",
        "ACCELERATING_RECOVERY": "Positive: System recovering, rate increasing.",
        "STABILIZING": "System was degrading but is now stabilizing.",
        "TRANSITIONAL": "System in transition between states.",
        "UNKNOWN": "Insufficient data for trajectory classification.",
    }
    return interpretations.get(classification, "Unknown state")


def calculate_warning_level(velocity: float, acceleration: float) -> tuple:
    """Calculate warning level and estimated time to threshold."""
    if velocity is None:
        return "UNKNOWN", None
    
    # Warning levels based on velocity magnitude
    v_mag = abs(velocity) if velocity else 0
    
    if v_mag < 0.05:
        level = "GREEN"
    elif v_mag < 0.1:
        level = "YELLOW"
    elif v_mag < 0.2:
        level = "ORANGE"
    else:
        level = "RED"
    
    # Boost warning if accelerating in bad direction
    if acceleration and velocity:
        if velocity > 0 and acceleration > 0:  # Accelerating degradation
            if level == "YELLOW":
                level = "ORANGE"
            elif level == "ORANGE":
                level = "RED"
    
    return level, None  # Time estimation requires domain calibration


# =============================================================================
# Report Generation
# =============================================================================

def generate_state_report(domain: str = None) -> ReportBuilder:
    """Generate state report."""
    
    config = load_domain_config(domain) if domain else {}
    time_unit = get_time_unit(config)
    report = ReportBuilder("State Report", domain=domain)
    
    # Load state data
    state_path = get_parquet_path('state', 'signal_pair')
    cohort_state_path = get_parquet_path('state', 'cohort')
    
    state = None
    cohort_state = None
    
    if Path(state_path).exists():
        state = pl.read_parquet(state_path)
    
    if Path(cohort_state_path).exists():
        cohort_state = pl.read_parquet(cohort_state_path)
    
    if state is None and cohort_state is None:
        report.add_section("Error", "No state data found. Run state entry point first.")
        return report
    
    # Use whichever data is available
    df = state if state is not None else cohort_state
    
    # ==========================================================================
    # Key Metrics
    # ==========================================================================
    n_rows = len(df)
    n_windows = df['window_id'].n_unique() if 'window_id' in df.columns else 0
    
    report.add_metric("State Records", n_rows)
    report.add_metric("Windows", n_windows)
    
    # ==========================================================================
    # Velocity Analysis
    # ==========================================================================
    velocity_cols = [c for c in df.columns if 'velocity' in c.lower()]
    
    if velocity_cols:
        v_col = velocity_cols[0]
        
        velocity_stats = df.select([
            pl.col(v_col).mean().alias('mean'),
            pl.col(v_col).std().alias('std'),
            pl.col(v_col).min().alias('min'),
            pl.col(v_col).max().alias('max'),
        ]).row(0, named=True)
        
        report.add_metric("Mean Velocity", format_number(velocity_stats['mean'], 4))
        report.add_metric("Velocity Std", format_number(velocity_stats['std'], 4))
        
        # Latest velocity
        if 'window_id' in df.columns:
            latest = df.filter(pl.col('window_id') == df['window_id'].max())
            latest_v = latest[v_col].mean() if len(latest) > 0 else None
            
            if latest_v is not None:
                report.add_metric("Current Velocity", format_number(latest_v, 4))
        
        # Velocity distribution
        report.add_section(
            "Velocity Analysis",
            f"**{v_col}** measures the rate of change in coupling dynamics.\n\n"
            f"- Mean: {format_number(velocity_stats['mean'], 4)}\n"
            f"- Range: {format_number(velocity_stats['min'], 4)} → {format_number(velocity_stats['max'], 4)}\n\n"
            "Positive velocity = coupling strengthening (can be good or bad depending on context)\n"
            "Negative velocity = coupling weakening (often precedes failure)\n\n"
            "**Key insight:** Velocity magnitude matters more than sign. Rapid change in either direction warrants attention."
        )
    
    # ==========================================================================
    # Acceleration Analysis
    # ==========================================================================
    accel_cols = [c for c in df.columns if 'acceleration' in c.lower() or 'accel' in c.lower()]
    
    if accel_cols:
        a_col = accel_cols[0]
        
        accel_stats = df.select([
            pl.col(a_col).mean().alias('mean'),
            pl.col(a_col).std().alias('std'),
            pl.col(a_col).min().alias('min'),
            pl.col(a_col).max().alias('max'),
        ]).row(0, named=True)
        
        report.add_section(
            "Acceleration Analysis",
            f"**{a_col}** measures the change in velocity — the second derivative.\n\n"
            f"- Mean: {format_number(accel_stats['mean'], 4)}\n"
            f"- Range: {format_number(accel_stats['min'], 4)} → {format_number(accel_stats['max'], 4)}\n\n"
            "**Why acceleration matters:**\n"
            "Systems don't fail at constant velocity. They ACCELERATE toward failure.\n"
            "Positive acceleration + positive velocity = exponential degradation."
        )
        
        # Combined trajectory analysis
        if velocity_cols:
            latest_v = df[velocity_cols[0]].mean()
            latest_a = df[a_col].mean()
            
            trajectory = classify_trajectory(latest_v, latest_a)
            interpretation = interpret_trajectory(trajectory)
            warning_level, _ = calculate_warning_level(latest_v, latest_a)
            
            report.add_section(
                "Trajectory Classification",
                f"**Current State:** {trajectory}\n\n"
                f"{interpretation}\n\n"
                f"**Warning Level:** {warning_level}"
            )
    
    # ==========================================================================
    # Per-Entity State (if available)
    # ==========================================================================
    entity_col = None
    for col in ['entity_id', 'cohort_id']:
        if col in df.columns:
            entity_col = col
            break
    
    if entity_col and velocity_cols:
        v_col = velocity_cols[0]
        
        entity_state = (
            df
            .group_by(entity_col)
            .agg([
                pl.col(v_col).last().alias('latest_velocity'),
                pl.col(v_col).mean().alias('mean_velocity'),
            ])
            .sort(pl.col('latest_velocity').abs(), descending=True)
        )
        
        rows = []
        for row in entity_state.head(10).iter_rows(named=True):
            entity_id = str(row[entity_col])
            if entity_col == 'signal_id':
                entity_name = translate_signal_id(entity_id, config)
            else:
                entity_name = entity_id
            
            latest_v = row['latest_velocity']
            level, _ = calculate_warning_level(latest_v, None)
            
            rows.append([
                entity_id,
                entity_name,
                format_number(row['mean_velocity'], 4),
                format_number(latest_v, 4),
                level,
            ])
        
        report.add_table(
            f"State by {entity_col} (sorted by velocity magnitude)",
            [entity_col, "Name", "Mean Velocity", "Latest Velocity", "Warning"],
            rows,
            alignments=['l', 'l', 'r', 'r', 'l'],
        )
    
    # ==========================================================================
    # Temporal Trend
    # ==========================================================================
    if 'window_id' in df.columns and velocity_cols:
        v_col = velocity_cols[0]
        
        trend = (
            df
            .group_by('window_id')
            .agg([
                pl.col(v_col).mean().alias('velocity'),
            ])
            .sort('window_id')
        )
        
        if len(trend) >= 5:
            # Check if velocity is trending upward (accelerating degradation)
            first_half = trend.head(len(trend) // 2)['velocity'].mean()
            second_half = trend.tail(len(trend) // 2)['velocity'].mean()
            
            if first_half is not None and second_half is not None:
                delta = second_half - first_half
                direction = "INCREASING ↑" if delta > 0.01 else ("DECREASING ↓" if delta < -0.01 else "STABLE →")
                
                report.add_section(
                    "Velocity Trend",
                    f"First half avg: {format_number(first_half, 4)}\n"
                    f"Second half avg: {format_number(second_half, 4)}\n"
                    f"Trend: {direction} (Δ = {format_number(delta, 4)})\n\n"
                    "Increasing velocity trend = system dynamics are changing faster over time."
                )
    
    # ==========================================================================
    # Early Warning Summary
    # ==========================================================================
    warnings = []
    
    # High velocity
    if velocity_cols:
        max_v = df[velocity_cols[0]].abs().max()
        if max_v and max_v > 0.2:
            warnings.append(f"⚠️ High velocity detected: |v| = {format_number(max_v, 3)}")
    
    # Accelerating
    if accel_cols and velocity_cols:
        # Check for positive accel + positive velocity
        problem_rows = df.filter(
            (pl.col(velocity_cols[0]) > 0.05) & 
            (pl.col(accel_cols[0]) > 0.01)
        )
        if len(problem_rows) > 0:
            warnings.append(f"⚠️ {len(problem_rows)} windows show accelerating degradation")
    
    if warnings:
        report.add_section("⚠️ Early Warnings", "\n".join(warnings))
    else:
        report.add_section("Status", "✓ No critical state warnings")
    
    # ==========================================================================
    # Key Insight
    # ==========================================================================
    report.add_section(
        "The PRISM Insight",
        "**Systems don't just decouple before failure — they decouple with accelerating velocity.**\n\n"
        "Traditional monitoring: \"This sensor crossed a threshold.\"\n"
        "PRISM monitoring: \"These sensors are decoupling at an accelerating rate.\"\n\n"
        "The second derivative (acceleration) is often the earliest warning signal."
    )
    
    return report


def main():
    parser = argparse.ArgumentParser(description='PRISM State Report')
    parser.add_argument('--domain', type=str, default=None, help='Domain name')
    parser.add_argument('--output', type=str, default=None, help='Output file (md or json)')
    args = parser.parse_args()
    
    if not args.domain:
        import os
        args.domain = os.environ.get('PRISM_DOMAIN')
    
    report = generate_state_report(args.domain)
    
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
