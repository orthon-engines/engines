"""
orchestrator.py - Ørthon Pipeline Orchestrator

Single entry point for comprehensive analysis.

Pipeline:
    raw → signal_typology → behavioral_geometry → phase_state → dynamical_systems → report

Stage names (research-facing):
    signal_typology:      What type of signal is this?
    behavioral_geometry:  How do signals relate to each other?
    phase_state:          How is the system evolving?
    dynamical_systems:    What type of dynamical system is this entity?

Usage:
    python -m prism.entry_points.orchestrator --input data.csv
    python -m prism.entry_points.orchestrator --input data.csv --stages signal_typology,behavioral_geometry
    python -m prism.entry_points.orchestrator --input data.csv --report html
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import polars as pl

# New pipeline modules
from prism.modules.signal_behavior import compute_all_metrics, CORE_ENGINES
from prism.modules.typology_classifier import compute_typology
from prism.modules.geometry import compute_geometry_features
from prism.modules.state import compute_state_features
from prism.modules.discovery import discover_cohorts, compute_cohort_summary

# Legacy aliases for backwards compatibility
from prism.modules.vector import compute_vector_features


# =============================================================================
# NEW PIPELINE STAGES
# =============================================================================

# Research-facing stage names
PIPELINE_STAGES = [
    'signal_typology',
    'behavioral_geometry',
    'phase_state',
    'dynamical_systems',
]

# Legacy stage aliases
STAGE_ALIASES = {
    'vector': 'signal_typology',
    'geometry': 'behavioral_geometry',
    'state': 'phase_state',
    'cohort': 'dynamical_systems',
}

# Output file names
STAGE_OUTPUT_FILES = {
    'signal_typology': 'signal_typology.parquet',
    'behavioral_geometry': 'behavioral_geometry.parquet',
    'phase_state': 'phase_state.parquet',
    'dynamical_systems': 'dynamical_systems.parquet',
}


@dataclass
class PipelineResult:
    """Results from a pipeline run."""
    outputs: Dict[str, Path]
    metadata: Dict[str, Any]
    started_at: datetime
    completed_at: datetime

    @property
    def duration_seconds(self) -> float:
        return (self.completed_at - self.started_at).total_seconds()

    def summary(self) -> str:
        lines = [
            "Ørthon Pipeline Results",
            "=" * 40,
            f"Duration: {self.duration_seconds:.1f}s",
            f"Input: {self.metadata.get('input', 'unknown')}",
            "",
            "Outputs:",
        ]
        for stage, path in self.outputs.items():
            if path.exists():
                size_kb = path.stat().st_size / 1024
                lines.append(f"  {stage}: {path.name} ({size_kb:.1f} KB)")
            else:
                lines.append(f"  {stage}: {path.name} (not found)")
        return "\n".join(lines)


def run(
    input_path: str | Path,
    output_dir: str | Path = "./results",
    entity_col: str = "entity_id",
    time_col: str = "timestamp",
    signal_col: str = "signal_id",
    value_col: str = "value",
    stages: Optional[List[str]] = None,
    report_format: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> PipelineResult:
    """
    Run comprehensive Ørthon pipeline.

    Pipeline: raw → signal_typology → behavioral_geometry → phase_state → dynamical_systems

    Args:
        input_path: Input data (csv, parquet, txt)
        output_dir: Directory for outputs
        entity_col: Entity identifier column
        time_col: Time/cycle column
        signal_col: Signal identifier column
        value_col: Value column
        stages: Stages to run. Default: all
                Options: ['signal_typology', 'behavioral_geometry', 'phase_state', 'dynamical_systems']
                Legacy aliases: ['vector', 'geometry', 'state', 'cohort']
        report_format: Generate report in this format ("html" or "parquet")
        config_overrides: Override config values

    Returns:
        PipelineResult with paths to all outputs
    """
    started_at = datetime.now()
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize stage names (support legacy aliases)
    all_stages = PIPELINE_STAGES
    if stages:
        stages = [STAGE_ALIASES.get(s, s) for s in stages]
        stages = [s for s in all_stages if s in stages]
    else:
        stages = all_stages

    outputs = {}

    # Load data
    print(f"Loading {input_path}...")
    df = _load_data(input_path)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Detect column types
    entity_col, time_col, signal_col, value_col = _detect_columns(
        df, entity_col, time_col, signal_col, value_col
    )
    print(f"  Entity: {entity_col}, Time: {time_col}, Signal: {signal_col}, Value: {value_col}")

    current_df = df

    # ==========================================================================
    # STAGE 1: SIGNAL TYPOLOGY
    # ==========================================================================
    if 'signal_typology' in stages:
        print("\n" + "=" * 60)
        print("  SIGNAL TYPOLOGY")
        print("  What type of signal is this?")
        print("=" * 60)

        typology_df = _compute_signal_typology(
            df=current_df,
            entity_col=entity_col,
            time_col=time_col,
            signal_col=signal_col,
            value_col=value_col,
        )
        typology_path = output_dir / STAGE_OUTPUT_FILES['signal_typology']
        typology_df.write_parquet(typology_path)
        outputs['signal_typology'] = typology_path
        current_df = typology_df
        print(f"  -> {typology_path} ({len(typology_df):,} rows)")

        # Print typology distribution
        if 'typology' in typology_df.columns:
            print("\n  Typology Distribution:")
            dist = typology_df.group_by('typology').len().sort('len', descending=True)
            for row in dist.iter_rows(named=True):
                print(f"    {row['typology']}: {row['len']:,}")

    # ==========================================================================
    # STAGE 2: BEHAVIORAL GEOMETRY
    # ==========================================================================
    if 'behavioral_geometry' in stages:
        print("\n" + "=" * 60)
        print("  BEHAVIORAL GEOMETRY")
        print("  How do signals relate to each other?")
        print("=" * 60)

        geometry_df = compute_geometry_features(
            df=current_df,
            entity_col=entity_col,
            time_col=time_col,
        )
        geometry_path = output_dir / STAGE_OUTPUT_FILES['behavioral_geometry']
        geometry_df.write_parquet(geometry_path)
        outputs['behavioral_geometry'] = geometry_path
        current_df = geometry_df
        print(f"  -> {geometry_path} ({len(geometry_df):,} rows)")

    # ==========================================================================
    # STAGE 3: PHASE STATE
    # ==========================================================================
    if 'phase_state' in stages:
        print("\n" + "=" * 60)
        print("  PHASE STATE")
        print("  How is the system evolving?")
        print("=" * 60)

        state_df = compute_state_features(
            df=current_df,
            entity_col=entity_col,
            time_col=time_col,
        )
        state_path = output_dir / STAGE_OUTPUT_FILES['phase_state']
        state_df.write_parquet(state_path)
        outputs['phase_state'] = state_path
        current_df = state_df
        print(f"  -> {state_path} ({len(state_df):,} rows)")

    # ==========================================================================
    # STAGE 4: DYNAMICAL SYSTEMS
    # ==========================================================================
    if 'dynamical_systems' in stages:
        print("\n" + "=" * 60)
        print("  DYNAMICAL SYSTEMS")
        print("  What type of dynamical system is this entity?")
        print("=" * 60)

        systems_df = discover_cohorts(
            df=current_df,
            entity_col=entity_col,
        )
        systems_path = output_dir / STAGE_OUTPUT_FILES['dynamical_systems']
        systems_df.write_parquet(systems_path)
        outputs['dynamical_systems'] = systems_path
        print(f"  -> {systems_path} ({len(systems_df):,} entities)")

        # Print system distribution
        if 'cohort' in systems_df.columns:
            print("\n  System Classification:")
            cohort_counts = systems_df.group_by('cohort').len().sort('cohort')
            for row in cohort_counts.iter_rows(named=True):
                print(f"    System {row['cohort']}: {row['len']} entities")

    # ==========================================================================
    # REPORT
    # ==========================================================================
    if report_format:
        print(f"\nGenerating {report_format} report...")
        report_path = _generate_report(
            outputs=outputs,
            output_dir=output_dir,
            format=report_format,
            entity_col=entity_col,
            input_path=input_path,
        )
        outputs['report'] = report_path
        print(f"  -> {report_path}")

    completed_at = datetime.now()

    result = PipelineResult(
        outputs=outputs,
        metadata={
            'input': str(input_path),
            'entity_col': entity_col,
            'time_col': time_col,
            'stages': stages,
        },
        started_at=started_at,
        completed_at=completed_at,
    )

    print("\n" + "=" * 60)
    print(f"  COMPLETE in {result.duration_seconds:.1f}s")
    print("  geometry leads — ørthon")
    print("=" * 60)

    return result


# =============================================================================
# STAGE IMPLEMENTATIONS
# =============================================================================

def _compute_signal_typology(
    df: pl.DataFrame,
    entity_col: str,
    time_col: str,
    signal_col: str,
    value_col: str,
    window_size: int = 252,
    stride: int = 21,
) -> pl.DataFrame:
    """
    Compute signal typology (combines characterization + engine computation + classification).

    This is the new unified first stage.
    """
    import numpy as np
    from prism.modules.signal_behavior import compute_all_metrics
    from prism.modules.typology_classifier import compute_typology

    results = []

    # Get unique (entity, signal) pairs
    pairs = df.select([entity_col, signal_col]).unique().to_dicts()
    print(f"  Processing {len(pairs)} (entity, signal) pairs...")

    for i, pair in enumerate(pairs):
        entity_id = pair[entity_col]
        signal_id = pair[signal_col]

        # Get signal data
        signal_df = df.filter(
            (pl.col(entity_col) == entity_id) & (pl.col(signal_col) == signal_id)
        ).sort(time_col)

        values = signal_df[value_col].to_numpy()
        times = signal_df[time_col].to_numpy()

        if len(values) < window_size:
            continue

        # Process windows
        prev_metrics = None

        for start in range(0, len(values) - window_size + 1, stride):
            window_values = values[start:start + window_size]
            window_start = times[start]
            window_end = times[start + window_size - 1]

            # Compute engine metrics
            metrics = compute_all_metrics(window_values)

            # Compute typology classification
            typology = compute_typology(metrics, prev_metrics)

            # Build row
            row = {
                entity_col: entity_id,
                signal_col: signal_id,
                'source_signal': signal_id,
                time_col: window_end,
                'window_start': window_start,
                'window_size': window_size,
                **metrics,
                **typology,
            }

            results.append(row)
            prev_metrics = metrics

        if (i + 1) % 50 == 0:
            print(f"    Processed {i + 1}/{len(pairs)} pairs")

    if not results:
        return pl.DataFrame()

    return pl.DataFrame(results)


# =============================================================================
# HELPERS
# =============================================================================

def _load_data(path: Path) -> pl.DataFrame:
    """Load data from various formats."""
    suffix = path.suffix.lower()
    if suffix == '.parquet':
        return pl.read_parquet(path)
    elif suffix == '.csv':
        return pl.read_csv(path)
    elif suffix == '.txt':
        return pl.read_csv(path, separator='\t')
    else:
        raise ValueError(f"Unsupported format: {suffix}")


def _detect_columns(
    df: pl.DataFrame,
    entity_col: str,
    time_col: str,
    signal_col: str,
    value_col: str,
) -> tuple:
    """Detect or validate column names."""
    cols = df.columns

    # Try to find entity column
    if entity_col not in cols:
        candidates = ['entity_id', 'unit_id', 'unit', 'id', 'machine_id', 'asset_id']
        for c in candidates:
            if c in cols:
                entity_col = c
                break

    # Try to find time column
    if time_col not in cols:
        candidates = ['timestamp', 'time', 'cycle', 't', 'datetime', 'date']
        for c in candidates:
            if c in cols:
                time_col = c
                break

    # Try to find signal column
    if signal_col not in cols:
        candidates = ['signal_id', 'signal', 'sensor', 'sensor_id', 'feature', 'variable']
        for c in candidates:
            if c in cols:
                signal_col = c
                break

    # Try to find value column
    if value_col not in cols:
        candidates = ['value', 'reading', 'measurement', 'y']
        for c in candidates:
            if c in cols:
                value_col = c
                break

    return entity_col, time_col, signal_col, value_col


def _generate_report(
    outputs: Dict[str, Path],
    output_dir: Path,
    format: str,
    entity_col: str,
    input_path: Path,
) -> Path:
    """Generate summary report."""
    if format == 'html':
        return _generate_html_report(outputs, output_dir, entity_col, input_path)
    elif format == 'parquet':
        return _generate_parquet_report(outputs, output_dir, entity_col)
    else:
        raise ValueError(f"Unknown format: {format}")


def _generate_html_report(
    outputs: Dict[str, Path],
    output_dir: Path,
    entity_col: str,
    input_path: Path,
) -> Path:
    """Generate HTML summary report."""
    report_path = output_dir / "report.html"

    # Gather stats from each output
    stats = {}
    for stage, path in outputs.items():
        if path.exists() and path.suffix == '.parquet':
            df = pl.read_parquet(path)
            stats[stage] = {
                'rows': len(df),
                'columns': len(df.columns),
                'col_names': df.columns[:20],
            }

    # Get system classification info
    systems_info = ""
    if 'dynamical_systems' in outputs and outputs['dynamical_systems'].exists():
        systems_df = pl.read_parquet(outputs['dynamical_systems'])
        if 'cohort' in systems_df.columns:
            systems_counts = systems_df.group_by('cohort').len().sort('cohort')
            systems_info = "<h2>Dynamical Systems</h2><table>"
            systems_info += "<tr><th>System</th><th>Entities</th></tr>"
            for row in systems_counts.iter_rows(named=True):
                systems_info += f"<tr><td>System {row['cohort']}</td><td>{row['len']}</td></tr>"
            systems_info += "</table>"

    # Get typology distribution
    typology_info = ""
    if 'signal_typology' in outputs and outputs['signal_typology'].exists():
        typology_df = pl.read_parquet(outputs['signal_typology'])
        if 'typology' in typology_df.columns:
            typology_counts = typology_df.group_by('typology').len().sort('len', descending=True)
            typology_info = "<h2>Signal Typology Distribution</h2><table>"
            typology_info += "<tr><th>Typology</th><th>Count</th></tr>"
            for row in typology_counts.iter_rows(named=True):
                typology_info += f"<tr><td>{row['typology']}</td><td>{row['len']:,}</td></tr>"
            typology_info += "</table>"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Ørthon Analysis Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #0d1117; color: #e6edf3; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #30363d; padding: 8px; text-align: left; }}
        th {{ background-color: #161b22; color: #58a6ff; }}
        tr:hover {{ background-color: #161b22; }}
        h1 {{ color: #e6edf3; }}
        h2 {{ color: #58a6ff; margin-top: 30px; }}
        .tagline {{ color: #8b949e; font-style: italic; }}
        .stage {{ background-color: #161b22; padding: 15px; margin: 10px 0; border-radius: 5px; border: 1px solid #30363d; }}
        .pipeline {{ display: flex; align-items: center; gap: 10px; margin: 20px 0; flex-wrap: wrap; }}
        .pipeline-stage {{ background: #161b22; border: 1px solid #30363d; padding: 10px 15px; border-radius: 6px; }}
        .pipeline-arrow {{ color: #58a6ff; font-size: 20px; }}
    </style>
</head>
<body>
    <h1>Ørthon Analysis Report</h1>
    <p class="tagline">geometry leads — ørthon</p>
    <p><em>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</em></p>
    <p><strong>Input:</strong> {input_path.name}</p>

    <h2>Pipeline</h2>
    <div class="pipeline">
        <div class="pipeline-stage">raw</div>
        <span class="pipeline-arrow">→</span>
        <div class="pipeline-stage">signal_typology</div>
        <span class="pipeline-arrow">→</span>
        <div class="pipeline-stage">behavioral_geometry</div>
        <span class="pipeline-arrow">→</span>
        <div class="pipeline-stage">phase_state</div>
        <span class="pipeline-arrow">→</span>
        <div class="pipeline-stage">dynamical_systems</div>
    </div>

    <h2>Pipeline Outputs</h2>
    <table>
        <tr><th>Stage</th><th>Rows</th><th>Columns</th><th>File</th></tr>
"""

    for stage, path in outputs.items():
        if stage in stats:
            s = stats[stage]
            html += f"<tr><td>{stage}</td><td>{s['rows']:,}</td><td>{s['columns']}</td><td>{path.name}</td></tr>\n"

    html += f"""
    </table>

    {typology_info}

    {systems_info}

    <h2>Column Details</h2>
"""

    for stage, s in stats.items():
        html += f"""
    <div class="stage">
        <h3>{stage}</h3>
        <p><strong>Columns:</strong> {', '.join(s['col_names'])}</p>
    </div>
"""

    html += """
</body>
</html>
"""

    report_path.write_text(html)
    return report_path


def _generate_parquet_report(
    outputs: Dict[str, Path],
    output_dir: Path,
    entity_col: str,
) -> Path:
    """Generate Parquet summary (combined features)."""
    report_path = output_dir / "summary.parquet"

    # Combine all outputs into one summary
    dfs = []
    for stage, path in outputs.items():
        if path.exists() and path.suffix == '.parquet':
            df = pl.read_parquet(path)
            # Add stage column
            df = df.with_columns(pl.lit(stage).alias('_stage'))
            dfs.append(df)

    if dfs:
        # This is a simple concat - in practice you'd want a proper join
        combined = pl.concat(dfs, how='diagonal')
        combined.write_parquet(report_path)

    return report_path


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ørthon Pipeline - Comprehensive behavioral geometry analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline:
    raw → signal_typology → behavioral_geometry → phase_state → dynamical_systems

Stages:
    signal_typology      What type of signal is this?
    behavioral_geometry  How do signals relate to each other?
    phase_state          How is the system evolving?
    dynamical_systems    What type of dynamical system is this entity?

Examples:
    python -m prism.entry_points.orchestrator --input data.csv
    python -m prism.entry_points.orchestrator --input data.parquet --stages signal_typology,behavioral_geometry
    python -m prism.entry_points.orchestrator --input data.csv --report html
    python -m prism.entry_points.orchestrator --input data.csv --output results/
"""
    )

    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input data file (csv, parquet, txt)")
    parser.add_argument("--output", "-o", type=str, default="./results",
                        help="Output directory (default: ./results)")
    parser.add_argument("--entity-col", type=str, default="entity_id",
                        help="Entity identifier column (default: entity_id)")
    parser.add_argument("--time-col", type=str, default="timestamp",
                        help="Time/cycle column (default: timestamp)")
    parser.add_argument("--signal-col", type=str, default="signal_id",
                        help="Signal identifier column (default: signal_id)")
    parser.add_argument("--value-col", type=str, default="value",
                        help="Value column (default: value)")
    parser.add_argument("--stages", type=str, default=None,
                        help="Comma-separated stages: signal_typology,behavioral_geometry,phase_state,dynamical_systems")
    parser.add_argument("--report", type=str, choices=["html", "parquet"],
                        help="Generate report in this format")

    args = parser.parse_args()

    stages = args.stages.split(",") if args.stages else None

    run(
        input_path=args.input,
        output_dir=args.output,
        entity_col=args.entity_col,
        time_col=args.time_col,
        signal_col=args.signal_col,
        value_col=args.value_col,
        stages=stages,
        report_format=args.report,
    )


if __name__ == "__main__":
    main()
