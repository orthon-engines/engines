"""
PRISM Report Utilities — Shared functions for all reports.

Handles:
- Domain config loading
- Signal name translation (PRISM → human readable)
- Consistent formatting
- Export to markdown/JSON
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import polars as pl
import yaml


def load_domain_config(domain: str) -> Dict[str, Any]:
    """
    Load domain YAML configuration.
    
    Searches:
        1. config/domains/{domain}.yaml (in data root)
        2. PRISM_CONFIG_PATH environment variable
    """
    import os
    from prism.db.parquet_store import get_data_root
    
    # Try data root first
    config_path = get_data_root() / 'config' / 'domains' / f'{domain}.yaml'
    
    if not config_path.exists():
        # Try environment variable
        env_path = os.environ.get('PRISM_CONFIG_PATH')
        if env_path:
            config_path = Path(env_path) / f'{domain}.yaml'
    
    if not config_path.exists():
        # Return minimal config if not found
        return {
            'domain': domain,
            'description': f'Domain: {domain}',
            'signals': {},
            'time': {'unit': 'units'},
        }
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def translate_signal_id(
    signal_id: str, 
    config: Dict[str, Any],
    include_description: bool = False,
) -> str:
    """
    Translate PRISM internal signal_id to human-readable name.
    
    Returns source name from YAML, or original if not found.
    """
    signals = config.get('signals', {})
    
    # Reverse lookup: find source name where prism_id matches
    for source_name, signal_config in signals.items():
        if signal_config.get('prism_id') == signal_id:
            if include_description:
                desc = signal_config.get('description', '')
                return f"{source_name} ({desc})" if desc else source_name
            return source_name
    
    # Not found in config, return original
    return signal_id


def translate_signal_column(
    df: pl.DataFrame,
    config: Dict[str, Any],
    signal_col: str = 'signal_id',
    output_col: str = 'signal_name',
) -> pl.DataFrame:
    """
    Add translated signal name column to DataFrame.
    """
    signals = config.get('signals', {})
    
    # Build mapping dict
    mapping = {}
    for source_name, signal_config in signals.items():
        prism_id = signal_config.get('prism_id')
        if prism_id:
            mapping[prism_id] = source_name
    
    return df.with_columns(
        pl.col(signal_col).replace(mapping).alias(output_col)
    )


def get_time_unit(config: Dict[str, Any]) -> str:
    """Get human-readable time unit from config."""
    return config.get('time', {}).get('unit', 'units')


def get_entity_description(config: Dict[str, Any]) -> str:
    """Get entity description from config."""
    return config.get('entity', {}).get('description', 'entity')


# =============================================================================
# Formatting Utilities
# =============================================================================

def format_number(n: float, decimals: int = 2) -> str:
    """Format number with commas and decimals."""
    if n is None:
        return "N/A"
    if abs(n) >= 1_000_000:
        return f"{n/1_000_000:,.{decimals}f}M"
    if abs(n) >= 1_000:
        return f"{n:,.0f}"
    return f"{n:,.{decimals}f}"


def format_percentage(n: float, decimals: int = 1) -> str:
    """Format as percentage."""
    if n is None:
        return "N/A"
    return f"{n * 100:.{decimals}f}%"


def format_table(
    headers: List[str],
    rows: List[List[str]],
    alignments: Optional[List[str]] = None,
) -> str:
    """
    Format data as ASCII table.
    
    alignments: list of 'l', 'r', 'c' for each column
    """
    if not rows:
        return ""
    
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    
    # Default alignments
    if alignments is None:
        alignments = ['l'] * len(headers)
    
    def align_cell(text: str, width: int, alignment: str) -> str:
        if alignment == 'r':
            return text.rjust(width)
        elif alignment == 'c':
            return text.center(width)
        return text.ljust(width)
    
    # Build table
    lines = []
    
    # Header
    header_line = " | ".join(
        align_cell(h, widths[i], alignments[i]) 
        for i, h in enumerate(headers)
    )
    lines.append(header_line)
    
    # Separator
    sep_line = "-+-".join("-" * w for w in widths)
    lines.append(sep_line)
    
    # Rows
    for row in rows:
        row_line = " | ".join(
            align_cell(str(cell), widths[i], alignments[i])
            for i, cell in enumerate(row)
        )
        lines.append(row_line)
    
    return "\n".join(lines)


# =============================================================================
# Report Output
# =============================================================================

class ReportBuilder:
    """
    Build reports with consistent formatting.
    
    Usage:
        report = ReportBuilder("Observations Report", domain="cmapss")
        report.add_section("Summary", summary_text)
        report.add_table("Signals", headers, rows)
        report.add_metric("Total Rows", 630000)
        
        print(report.to_text())
        report.save_markdown("report.md")
        report.save_json("report.json")
    """
    
    def __init__(self, title: str, domain: str = None):
        self.title = title
        self.domain = domain
        self.timestamp = datetime.now().isoformat()
        self.sections = []
        self.metrics = {}
    
    def add_section(self, name: str, content: str):
        """Add a text section."""
        self.sections.append({
            'type': 'text',
            'name': name,
            'content': content,
        })
    
    def add_table(
        self, 
        name: str, 
        headers: List[str], 
        rows: List[List[Any]],
        alignments: Optional[List[str]] = None,
    ):
        """Add a table section."""
        self.sections.append({
            'type': 'table',
            'name': name,
            'headers': headers,
            'rows': [[str(cell) for cell in row] for row in rows],
            'alignments': alignments,
        })
    
    def add_metric(self, name: str, value: Any, unit: str = None):
        """Add a key metric."""
        self.metrics[name] = {'value': value, 'unit': unit}
    
    def to_text(self) -> str:
        """Render as plain text."""
        lines = []
        
        # Header
        lines.append("=" * 70)
        lines.append(self.title.upper())
        if self.domain:
            lines.append(f"Domain: {self.domain}")
        lines.append(f"Generated: {self.timestamp}")
        lines.append("=" * 70)
        lines.append("")
        
        # Key metrics
        if self.metrics:
            lines.append("KEY METRICS")
            lines.append("-" * 40)
            for name, info in self.metrics.items():
                val = format_number(info['value']) if isinstance(info['value'], (int, float)) else str(info['value'])
                unit = f" {info['unit']}" if info['unit'] else ""
                lines.append(f"  {name}: {val}{unit}")
            lines.append("")
        
        # Sections
        for section in self.sections:
            lines.append(section['name'].upper())
            lines.append("-" * 40)
            
            if section['type'] == 'text':
                lines.append(section['content'])
            elif section['type'] == 'table':
                lines.append(format_table(
                    section['headers'],
                    section['rows'],
                    section.get('alignments'),
                ))
            
            lines.append("")
        
        return "\n".join(lines)
    
    def to_markdown(self) -> str:
        """Render as markdown."""
        lines = []
        
        # Header
        lines.append(f"# {self.title}")
        if self.domain:
            lines.append(f"\n**Domain:** {self.domain}")
        lines.append(f"\n**Generated:** {self.timestamp}")
        lines.append("")
        
        # Key metrics
        if self.metrics:
            lines.append("## Key Metrics")
            lines.append("")
            for name, info in self.metrics.items():
                val = format_number(info['value']) if isinstance(info['value'], (int, float)) else str(info['value'])
                unit = f" {info['unit']}" if info['unit'] else ""
                lines.append(f"- **{name}:** {val}{unit}")
            lines.append("")
        
        # Sections
        for section in self.sections:
            lines.append(f"## {section['name']}")
            lines.append("")
            
            if section['type'] == 'text':
                lines.append(section['content'])
            elif section['type'] == 'table':
                # Markdown table
                headers = section['headers']
                lines.append("| " + " | ".join(headers) + " |")
                lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for row in section['rows']:
                    lines.append("| " + " | ".join(row) + " |")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            'title': self.title,
            'domain': self.domain,
            'timestamp': self.timestamp,
            'metrics': self.metrics,
            'sections': self.sections,
        }
    
    def save_markdown(self, path: str):
        """Save as markdown file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(self.to_markdown())
    
    def save_json(self, path: str):
        """Save as JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
