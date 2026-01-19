#!/usr/bin/env python3
"""
PRISM Signal Vector Report — 51 behavioral metrics summary.

Shows:
- Metrics computed per signal
- Top discriminating features (highest variance)
- Hilbert transform analysis (dominant in fault detection)
- Per-cohort metric patterns

Usage:
    python -m reports.signal_vector_report
    python -m reports.signal_vector_report --domain cmapss
    python -m reports.signal_vector_report --output report.md
"""

import argparse
from pathlib import Path
from typing import Dict, Any

import polars as pl
import numpy as np

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
# Metric Categories
# =============================================================================

METRIC_CATEGORIES = {
    'entropy': ['sample_entropy', 'permutation_entropy', 'spectral_entropy', 'approx_entropy'],
    'memory': ['hurst_exponent', 'dfa_alpha', 'autocorr_decay'],
    'hilbert': ['inst_freq_mean', 'inst_freq_std', 'inst_phase_coherence', 'inst_amplitude_mean'],
    'spectral': ['dominant_freq', 'spectral_centroid', 'spectral_rolloff', 'band_power_ratio'],
    'nonlinear': ['lyapunov_exp', 'correlation_dim', 'recurrence_rate', 'determinism'],
    'volatility': ['garch_alpha', 'garch_beta', 'garch_omega', 'volatility_persistence'],
    'distribution': ['skewness', 'kurtosis', 'iqr', 'coefficient_of_variation'],
    'trend': ['trend_slope', 'trend_strength', 'seasonality_strength', 'residual_variance'],
}


def categorize_metric(metric_name: str) -> str:
    """Determine category for a metric name."""
    metric_lower = metric_name.lower()
    
    for category, keywords in METRIC_CATEGORIES.items():
        if any(kw in metric_lower for kw in keywords):
            return category
    
    return 'other'


# =============================================================================
# Report Generation
# =============================================================================

def generate_signal_vector_report(domain: str = None) -> ReportBuilder:
    """Generate signal vector report."""
    
    config = load_domain_config(domain) if domain else {}
    time_unit = get_time_unit(config)
    report = ReportBuilder("Signal Vector Report", domain=domain)
    
    # Load vector data
    vector_path = get_parquet_path('vector', 'signal')
    if not Path(vector_path).exists():
        report.add_section("Error", f"Vector data not found: {vector_path}\nRun signal_vector entry point first.")
        return report
    
    vector = pl.read_parquet(vector_path)
    
    # Identify metric columns (exclude metadata)
    exclude_cols = {'entity_id', 'signal_id', 'signal_type', 'timestamp', 
                    'window_id', 'window_start', 'window_end', 'cohort_id', 'n_obs'}
    metric_cols = [c for c in vector.columns if c not in exclude_cols]
    
    # ==========================================================================
    # Key Metrics
    # ==========================================================================
    n_rows = len(vector)
    n_signals = vector['signal_id'].n_unique() if 'signal_id' in vector.columns else 0
    n_windows = vector['window_id'].n_unique() if 'window_id' in vector.columns else 0
    n_metrics = len(metric_cols)
    
    report.add_metric("Vector Rows", n_rows)
    report.add_metric("Signals", n_signals)
    report.add_metric("Windows", n_windows)
    report.add_metric("Metrics Computed", n_metrics)
    
    # ==========================================================================
    # Metric Categories
    # ==========================================================================
    category_counts = {}
    for metric in metric_cols:
        cat = categorize_metric(metric)
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    rows = []
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        rows.append([cat.title(), str(count)])
    
    report.add_table(
        "Metrics by Category",
        ["Category", "Count"],
        rows,
        alignments=['l', 'r'],
    )
    
    # ==========================================================================
    # Top Discriminating Metrics (by variance)
    # ==========================================================================
    # Higher variance = more discriminating power
    variance_stats = []
    for col in metric_cols:
        if vector[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
            var = vector[col].var()
            mean = vector[col].mean()
            cv = (var ** 0.5) / abs(mean) if mean and mean != 0 else 0
            variance_stats.append({
                'metric': col,
                'variance': var,
                'mean': mean,
                'cv': cv,  # Coefficient of variation
            })
    
    # Sort by CV (coefficient of variation) — better than raw variance
    variance_stats.sort(key=lambda x: -(x['cv'] or 0))
    
    rows = []
    for stat in variance_stats[:15]:  # Top 15
        category = categorize_metric(stat['metric'])
        rows.append([
            stat['metric'],
            category,
            format_number(stat['mean'], 4),
            format_number(stat['variance'], 4),
            format_number(stat['cv'], 2),
        ])
    
    report.add_table(
        "Top Discriminating Metrics (by coefficient of variation)",
        ["Metric", "Category", "Mean", "Variance", "CV"],
        rows,
        alignments=['l', 'l', 'r', 'r', 'r'],
    )
    
    # ==========================================================================
    # Hilbert Transform Analysis
    # ==========================================================================
    hilbert_metrics = [c for c in metric_cols if 'hilbert' in c.lower() or 'inst_' in c.lower()]
    
    if hilbert_metrics:
        report.add_section(
            "Hilbert Transform Features",
            "Hilbert transform metrics capture instantaneous frequency and phase information.\n"
            "These often dominate fault detection (47% importance in turbofan analysis).\n\n"
            f"Found {len(hilbert_metrics)} Hilbert-related metrics: {', '.join(hilbert_metrics[:5])}"
        )
        
        # Stats on Hilbert metrics
        rows = []
        for metric in hilbert_metrics[:10]:
            if metric in vector.columns:
                stats = vector.select([
                    pl.col(metric).mean().alias('mean'),
                    pl.col(metric).std().alias('std'),
                    pl.col(metric).min().alias('min'),
                    pl.col(metric).max().alias('max'),
                ]).row(0, named=True)
                
                rows.append([
                    metric,
                    format_number(stats['mean'], 4),
                    format_number(stats['std'], 4),
                    format_number(stats['min'], 4),
                    format_number(stats['max'], 4),
                ])
        
        if rows:
            report.add_table(
                "Hilbert Metric Statistics",
                ["Metric", "Mean", "Std", "Min", "Max"],
                rows,
                alignments=['l', 'r', 'r', 'r', 'r'],
            )
    
    # ==========================================================================
    # Per-Signal Summary
    # ==========================================================================
    if 'signal_id' in vector.columns:
        # Average metrics per signal
        signal_summary = (
            vector
            .group_by('signal_id')
            .agg([
                pl.count().alias('n_windows'),
            ] + [
                pl.col(c).mean().alias(f'{c}_mean')
                for c in metric_cols[:5]  # Just first 5 metrics for summary
            ])
            .sort('signal_id')
        )
        
        rows = []
        for row in signal_summary.head(15).iter_rows(named=True):
            signal_name = translate_signal_id(row['signal_id'], config)
            rows.append([
                row['signal_id'],
                signal_name,
                str(row['n_windows']),
            ])
        
        if len(signal_summary) > 15:
            rows.append(["...", f"({len(signal_summary) - 15} more)", ""])
        
        report.add_table(
            "Per-Signal Summary",
            ["Signal ID", "Name", "Windows"],
            rows,
            alignments=['l', 'l', 'r'],
        )
    
    # ==========================================================================
    # Key Insights
    # ==========================================================================
    insights = []
    
    # Check for highly variable metrics
    high_cv = [s for s in variance_stats if s['cv'] and s['cv'] > 1.0]
    if high_cv:
        insights.append(f"- {len(high_cv)} metrics have CV > 1.0 — strong discriminating power")
    
    # Check for near-constant metrics
    low_cv = [s for s in variance_stats if s['cv'] is not None and s['cv'] < 0.01]
    if low_cv:
        insights.append(f"- {len(low_cv)} metrics have CV < 0.01 — may be uninformative")
    
    # Hilbert dominance
    if hilbert_metrics:
        hilbert_in_top = sum(1 for s in variance_stats[:10] if any(h in s['metric'].lower() for h in ['hilbert', 'inst_']))
        if hilbert_in_top >= 3:
            insights.append(f"- {hilbert_in_top}/10 top metrics are Hilbert-related — frequency modulation is key signal")
    
    if insights:
        report.add_section("Key Insights", "\n".join(insights))
    
    return report


def main():
    parser = argparse.ArgumentParser(description='PRISM Signal Vector Report')
    parser.add_argument('--domain', type=str, default=None, help='Domain name')
    parser.add_argument('--output', type=str, default=None, help='Output file (md or json)')
    args = parser.parse_args()
    
    if not args.domain:
        import os
        args.domain = os.environ.get('PRISM_DOMAIN')
    
    report = generate_signal_vector_report(args.domain)
    
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
