"""
Summary Engine

Generates executive summaries and reports.
Produces text summaries, key findings, and recommendations.

Key insight: Executives need one page, not 50 parquets.
"""

import numpy as np
import polars as pl
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class SummaryConfig:
    """Configuration for summary engine."""
    report_title: str = 'PRISM Health Report'
    include_recommendations: bool = True
    top_n_entities: int = 10
    top_n_concerns: int = 5


class SummaryEngine:
    """
    Executive Summary Engine.

    Generates human-readable reports from analysis results.

    Outputs:
    - report_section: Section name
    - content: Section content
    - priority: Display order (1 = highest)
    """

    ENGINE_TYPE = "statistics"

    def __init__(self, config: Optional[SummaryConfig] = None):
        self.config = config or SummaryConfig()

    def generate_executive_summary(
        self,
        health_df: pl.DataFrame,
        rankings_df: pl.DataFrame = None,
        anomaly_df: pl.DataFrame = None
    ) -> List[Dict[str, Any]]:
        """
        Generate executive summary sections.

        Parameters
        ----------
        health_df : pl.DataFrame
            Health DataFrame
        rankings_df : pl.DataFrame, optional
            Rankings DataFrame
        anomaly_df : pl.DataFrame, optional
            Anomaly DataFrame

        Returns
        -------
        list of dict
            Report sections
        """
        sections = []

        # Get latest window data
        if 'window_id' in health_df.columns:
            latest_window = health_df['window_id'].max()
            latest_health = health_df.filter(pl.col('window_id') == latest_window)
        else:
            latest_window = 0
            latest_health = health_df

        n_entities = latest_health['unit_id'].n_unique()
        avg_health = latest_health['health_score'].mean() if 'health_score' in latest_health.columns else 0

        # Count by risk level
        if 'risk_level' in latest_health.columns:
            n_critical = latest_health.filter(pl.col('risk_level') == 'CRITICAL').height
            n_high = latest_health.filter(pl.col('risk_level') == 'HIGH').height
            n_moderate = latest_health.filter(pl.col('risk_level') == 'MODERATE').height
            n_low = latest_health.filter(pl.col('risk_level') == 'LOW').height
        else:
            n_critical = n_high = n_moderate = n_low = 0

        # Determine overall status
        if n_critical > 0:
            overall_status = 'CRITICAL ATTENTION REQUIRED'
        elif n_high > 0:
            overall_status = 'HIGH RISK - ACTION NEEDED'
        elif n_moderate > n_entities * 0.3:
            overall_status = 'MODERATE - MONITOR CLOSELY'
        else:
            overall_status = 'NORMAL OPERATIONS'

        # ----- EXECUTIVE SUMMARY -----
        exec_summary = f"""EXECUTIVE SUMMARY
=================
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Fleet Overview:
- Total Entities Monitored: {n_entities}
- Average Health Score: {avg_health:.1f}/100
- Critical Risk Entities: {n_critical}
- High Risk Entities: {n_high}
- Moderate Risk Entities: {n_moderate}
- Low Risk (Healthy): {n_low}

Overall Fleet Status: {overall_status}"""

        sections.append({
            'report_section': 'EXECUTIVE_SUMMARY',
            'content': exec_summary,
            'priority': 1
        })

        # ----- CRITICAL ALERTS -----
        if 'risk_level' in latest_health.columns:
            critical_entities = latest_health.filter(
                pl.col('risk_level').is_in(['CRITICAL', 'HIGH'])
            ).sort('health_score')

            if len(critical_entities) > 0:
                alerts = "CRITICAL ALERTS\n===============\n"
                for row in critical_entities.head(self.config.top_n_entities).iter_rows(named=True):
                    alerts += f"\n* {row['unit_id']}: "
                    alerts += f"Health {row.get('health_score', 'N/A'):.0f}, "
                    alerts += f"Risk: {row.get('risk_level', 'N/A')}, "
                    alerts += f"Concern: {row.get('primary_concern', 'Unknown')}"

                sections.append({
                    'report_section': 'CRITICAL_ALERTS',
                    'content': alerts,
                    'priority': 2
                })

        # ----- TOP CONCERNS -----
        if 'primary_concern' in latest_health.columns:
            concern_counts = latest_health.group_by('primary_concern').agg([
                pl.count().alias('count')
            ]).sort('count', descending=True)

            concerns = "TOP CONCERNS ACROSS FLEET\n=========================\n"
            for row in concern_counts.head(self.config.top_n_concerns).iter_rows(named=True):
                if row['primary_concern'] != 'None':
                    concerns += f"\n* {row['primary_concern']}: {row['count']} entities"

            sections.append({
                'report_section': 'TOP_CONCERNS',
                'content': concerns,
                'priority': 3
            })

        # ----- HIGH Z-SCORE SUMMARY (computed, not classified) -----
        if anomaly_df is not None and len(anomaly_df) > 0 and 'z_score' in anomaly_df.columns:
            if 'window_id' in anomaly_df.columns:
                recent = anomaly_df.filter(pl.col('window_id') == latest_window)
            else:
                recent = anomaly_df

            # Count by z-score magnitude (PRISM computes, ORTHON interprets)
            high_zscore = recent.filter(pl.col('z_score').abs() > 3.0).height
            elevated_zscore = recent.filter(pl.col('z_score').abs() > 2.0).height

            anomaly_summary = f"""HIGH Z-SCORE SUMMARY
====================
z_score > 3.0: {high_zscore}
z_score > 2.0: {elevated_zscore}
"""

            if 'metric_name' in recent_anomalies.columns and len(recent_anomalies) > 0:
                metric_anomalies = recent_anomalies.group_by('metric_name').agg([
                    pl.count().alias('count'),
                    pl.col('z_score').mean().alias('avg_z') if 'z_score' in recent_anomalies.columns else pl.lit(0).alias('avg_z')
                ]).sort('count', descending=True)

                anomaly_summary += "\nMost Anomalous Metrics:\n"
                for row in metric_anomalies.head(5).iter_rows(named=True):
                    anomaly_summary += f"* {row['metric_name']}: {row['count']} anomalies"
                    if row.get('avg_z'):
                        anomaly_summary += f" (avg z={row['avg_z']:.1f})"
                    anomaly_summary += "\n"

            sections.append({
                'report_section': 'ANOMALY_SUMMARY',
                'content': anomaly_summary,
                'priority': 4
            })

        # ----- ENTITY RANKINGS -----
        if rankings_df is not None and len(rankings_df) > 0:
            worst = rankings_df.sort('avg_health').head(5)
            best = rankings_df.sort('avg_health', descending=True).head(5)

            rankings_text = "ENTITY RANKINGS\n===============\n\nWorst Performing:\n"
            for i, row in enumerate(worst.iter_rows(named=True)):
                rankings_text += f"{i+1}. {row['unit_id']}: Health {row['avg_health']:.1f}"
                if 'critical_events' in row:
                    rankings_text += f", Critical Events: {row['critical_events']}"
                rankings_text += "\n"

            rankings_text += "\nBest Performing:\n"
            for i, row in enumerate(best.iter_rows(named=True)):
                rankings_text += f"{i+1}. {row['unit_id']}: Health {row['avg_health']:.1f}\n"

            sections.append({
                'report_section': 'ENTITY_RANKINGS',
                'content': rankings_text,
                'priority': 5
            })

        # ----- RECOMMENDATIONS -----
        if self.config.include_recommendations:
            recommendations = "RECOMMENDATIONS\n===============\n"

            if n_critical > 0:
                recommendations += f"\n[IMMEDIATE] Inspect {n_critical} critical entities immediately"

            if n_high > 0:
                recommendations += f"\n[URGENT] Schedule maintenance for {n_high} high-risk entities within 1 week"

            if n_moderate > 0:
                recommendations += f"\n[MONITOR] Keep close watch on {n_moderate} moderate-risk entities"

            # Specific recommendations based on top concern
            if 'primary_concern' in latest_health.columns:
                top_concern = concern_counts['primary_concern'][0] if len(concern_counts) > 0 else None

                if top_concern == 'Chaotic dynamics':
                    recommendations += "\n[REVIEW] Multiple entities showing chaotic behavior - check operating conditions"
                elif top_concern == 'Energy imbalance':
                    recommendations += "\n[CHECK] Energy balance issues detected - inspect for leaks or sensor calibration"
                elif top_concern == 'Low determinism':
                    recommendations += "\n[INVESTIGATE] Unpredictable behavior patterns - review recent changes"
                elif top_concern == 'Approaching bifurcation':
                    recommendations += "\n[ALERT] Critical slowing down detected - high risk of regime change"

            recommendations += "\n\n[CONTINUE] Monitor all entities through PRISM dashboard"

            sections.append({
                'report_section': 'RECOMMENDATIONS',
                'content': recommendations,
                'priority': 6
            })

        # ----- TRENDS -----
        if 'window_id' in health_df.columns and health_df['window_id'].n_unique() > 1:
            health_trend = health_df.group_by('window_id').agg([
                pl.col('health_score').mean().alias('avg_health')
            ]).sort('window_id')

            first_health = health_trend['avg_health'][0]
            last_health = health_trend['avg_health'][-1]

            if last_health > first_health + 2:
                trend_direction = 'IMPROVING'
            elif last_health < first_health - 2:
                trend_direction = 'DEGRADING'
            else:
                trend_direction = 'STABLE'

            trends_text = f"""TRENDS
======
Fleet Health Trend: {trend_direction}
- Initial Average: {first_health:.1f}
- Current Average: {last_health:.1f}
- Change: {last_health - first_health:+.1f}"""

            sections.append({
                'report_section': 'TRENDS',
                'content': trends_text,
                'priority': 7
            })

        return sections


def run_summary_engine(
    health_path: Path,
    rankings_path: Optional[Path],
    anomaly_path: Optional[Path],
    config: SummaryConfig,
    output_path: Optional[Path] = None
) -> pl.DataFrame:
    """
    Generate executive summary report.

    Parameters
    ----------
    health_path : Path
        Path to health parquet
    rankings_path : Path, optional
        Path to rankings parquet
    anomaly_path : Path, optional
        Path to anomaly parquet
    config : SummaryConfig
        Engine configuration
    output_path : Path, optional
        Path to write output parquet

    Returns
    -------
    pl.DataFrame
        Summary report sections
    """
    engine = SummaryEngine(config)

    # Load data
    try:
        health_df = pl.read_parquet(health_path)
    except Exception as e:
        return pl.DataFrame({
            'report_section': ['ERROR'],
            'content': [f'Failed to load health data: {e}'],
            'priority': [0]
        })

    rankings_df = None
    if rankings_path:
        try:
            rankings_df = pl.read_parquet(rankings_path)
        except Exception:
            pass

    anomaly_df = None
    if anomaly_path:
        try:
            anomaly_df = pl.read_parquet(anomaly_path)
        except Exception:
            pass

    # Generate report sections
    sections = engine.generate_executive_summary(health_df, rankings_df, anomaly_df)

    df_out = pl.DataFrame(sections)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_out.write_parquet(output_path)

    return df_out


def generate_text_report(summary_df: pl.DataFrame) -> str:
    """Generate plain text report from summary DataFrame."""
    report = ""
    for row in summary_df.sort('priority').iter_rows(named=True):
        report += row['content'] + "\n\n" + "=" * 50 + "\n\n"
    return report


def generate_markdown_report(summary_df: pl.DataFrame) -> str:
    """Generate markdown report from summary DataFrame."""
    report = "# PRISM Health Report\n\n"

    for row in summary_df.sort('priority').iter_rows(named=True):
        section = row['report_section'].replace('_', ' ').title()
        content = row['content']

        lines = content.split('\n')
        report += f"## {section}\n\n"

        for line in lines[2:]:  # Skip header lines
            if line.strip():
                if line.startswith('*'):
                    report += f"- {line[1:].strip()}\n"
                elif line.startswith('['):
                    report += f"**{line}**\n"
                else:
                    report += f"{line}\n"

        report += "\n---\n\n"

    return report
