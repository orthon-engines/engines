#!/usr/bin/env python3
"""
PRISM Observations Report — Summary of raw data loaded.

Shows:
- Data volume (rows, entities, signals)
- Timestamp range and coverage
- Signal inventory with human-readable names
- Data quality metrics (nulls, outliers)

Usage:
    python -m reports.observations_report
    python -m reports.observations_report --domain cmapss
    python -m reports.observations_report --output report.md
"""

import argparse
import sys
from pathlib import Path

import polars as pl

from prism.db.parquet_store import get_parquet_path
from report_utils import (
    ReportBuilder, 
    load_domain_config, 
    translate_signal_id,
    format_number,
    get_time_unit,
    get_entity_description,
)


def generate_observations_report(domain: str = None) -> ReportBuilder:
    """Generate observations report."""
    
    # Load config
    config = load_domain_config(domain) if domain else {}
    time_unit = get_time_unit(config)
    entity_desc = get_entity_description(config)
    
    report = ReportBuilder("Observations Report", domain=domain)
    
    # Load observations
    obs_path = get_parquet_path('raw', 'observations')
    if not Path(obs_path).exists():
        report.add_section("Error", f"Observations not found: {obs_path}")
        return report
    
    df = pl.read_parquet(obs_path)
    
    # ==========================================================================
    # Key Metrics
    # ==========================================================================
    n_rows = len(df)
    n_entities = df['entity_id'].n_unique() if 'entity_id' in df.columns else 0
    n_signals = df['signal_id'].n_unique() if 'signal_id' in df.columns else 0
    
    report.add_metric("Total Observations", n_rows)
    report.add_metric(f"Entities ({entity_desc}s)", n_entities)
    report.add_metric("Signals", n_signals)
    
    if 'timestamp' in df.columns:
        ts_min = df['timestamp'].min()
        ts_max = df['timestamp'].max()
        ts_range = ts_max - ts_min
        report.add_metric("Timestamp Range", f"{ts_min} → {ts_max}", time_unit)
        report.add_metric("Duration", ts_range, time_unit)
    
    # Observations per entity
    obs_per_entity = n_rows / n_entities if n_entities > 0 else 0
    report.add_metric("Avg Observations/Entity", round(obs_per_entity))
    
    # ==========================================================================
    # Entity Summary
    # ==========================================================================
    if 'entity_id' in df.columns:
        entity_stats = (
            df
            .group_by('entity_id')
            .agg([
                pl.count().alias('n_obs'),
                pl.col('timestamp').min().alias('ts_min'),
                pl.col('timestamp').max().alias('ts_max'),
            ])
            .with_columns(
                (pl.col('ts_max') - pl.col('ts_min')).alias('duration')
            )
            .sort('entity_id')
        )
        
        # Show first 10 entities
        rows = []
        for row in entity_stats.head(10).iter_rows(named=True):
            rows.append([
                str(row['entity_id']),
                format_number(row['n_obs'], 0),
                format_number(row['ts_min'], 1),
                format_number(row['ts_max'], 1),
                format_number(row['duration'], 1),
            ])
        
        if len(entity_stats) > 10:
            rows.append(["...", f"({len(entity_stats) - 10} more)", "", "", ""])
        
        report.add_table(
            f"Entities ({entity_desc}s)",
            ["Entity ID", "Observations", f"Start ({time_unit})", f"End ({time_unit})", "Duration"],
            rows,
            alignments=['l', 'r', 'r', 'r', 'r'],
        )
    
    # ==========================================================================
    # Signal Inventory
    # ==========================================================================
    if 'signal_id' in df.columns:
        signal_stats = (
            df
            .group_by('signal_id')
            .agg([
                pl.count().alias('n_obs'),
                pl.col('value').mean().alias('mean'),
                pl.col('value').std().alias('std'),
                pl.col('value').min().alias('min'),
                pl.col('value').max().alias('max'),
                pl.col('value').is_null().sum().alias('n_nulls'),
            ])
            .sort('signal_id')
        )
        
        rows = []
        for row in signal_stats.iter_rows(named=True):
            # Translate signal ID to human name
            human_name = translate_signal_id(row['signal_id'], config)
            
            rows.append([
                row['signal_id'],
                human_name,
                format_number(row['n_obs'], 0),
                format_number(row['mean'], 2),
                format_number(row['std'], 2),
                format_number(row['min'], 2),
                format_number(row['max'], 2),
                str(row['n_nulls']),
            ])
        
        report.add_table(
            "Signal Inventory",
            ["Signal ID", "Name", "Count", "Mean", "Std", "Min", "Max", "Nulls"],
            rows,
            alignments=['l', 'l', 'r', 'r', 'r', 'r', 'r', 'r'],
        )
    
    # ==========================================================================
    # Data Quality
    # ==========================================================================
    quality_issues = []
    
    # Check for nulls
    for col in ['entity_id', 'signal_id', 'timestamp', 'value']:
        if col in df.columns:
            null_count = df.select(pl.col(col).is_null().sum()).item()
            if null_count > 0:
                quality_issues.append(f"- {col}: {null_count:,} null values")
    
    # Check for duplicate rows
    n_dupes = len(df) - len(df.unique())
    if n_dupes > 0:
        quality_issues.append(f"- {n_dupes:,} duplicate rows")
    
    # Check timestamp monotonicity
    if 'timestamp' in df.columns and 'entity_id' in df.columns:
        violations = (
            df
            .sort(['entity_id', 'timestamp'])
            .with_columns(
                pl.col('timestamp').diff().over('entity_id').alias('ts_diff')
            )
            .filter(pl.col('ts_diff') < 0)
        )
        if len(violations) > 0:
            quality_issues.append(f"- {len(violations):,} timestamp ordering violations")
    
    if quality_issues:
        report.add_section("Data Quality Issues", "\n".join(quality_issues))
    else:
        report.add_section("Data Quality", "✓ No issues detected")
    
    # ==========================================================================
    # Target Variable (if present)
    # ==========================================================================
    target_col = config.get('target', {}).get('source_column')
    if target_col and target_col in df.columns:
        target_stats = df.select([
            pl.col(target_col).mean().alias('mean'),
            pl.col(target_col).std().alias('std'),
            pl.col(target_col).min().alias('min'),
            pl.col(target_col).max().alias('max'),
        ]).row(0, named=True)
        
        target_desc = config.get('target', {}).get('description', 'Target')
        report.add_section(
            f"Target Variable: {target_col}",
            f"{target_desc}\n"
            f"  Mean: {format_number(target_stats['mean'], 2)}\n"
            f"  Std:  {format_number(target_stats['std'], 2)}\n"
            f"  Range: {format_number(target_stats['min'], 2)} → {format_number(target_stats['max'], 2)}"
        )
    
    return report


def main():
    parser = argparse.ArgumentParser(description='PRISM Observations Report')
    parser.add_argument('--domain', type=str, default=None, help='Domain name')
    parser.add_argument('--output', type=str, default=None, help='Output file (md or json)')
    args = parser.parse_args()
    
    # Get domain from env if not specified
    if not args.domain:
        import os
        args.domain = os.environ.get('PRISM_DOMAIN')
    
    report = generate_observations_report(args.domain)
    
    # Output
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
