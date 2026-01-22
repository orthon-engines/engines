---
title: Quickstart
description: Get up and running with Ørthon in 5 minutes
---

This guide walks you through running your first analysis with Ørthon.

## Prerequisites

- Python 3.9+
- Your data in CSV or Parquet format

## Installation

```bash
# Clone the repository
git clone https://github.com/yourname/prism-engines.git
cd prism-engines/diagnostic

# Install dependencies
pip install -r requirements.txt
```

## Prepare Your Data

Your data needs two key columns:

| Column | Description | Example |
|--------|-------------|---------|
| `entity_id` | Identifies each unit/system | `unit_001`, `bearing_A` |
| `timestamp` | Time index (cycle, seconds, etc.) | `1`, `2`, `3`... |

Everything else is treated as sensor columns.

```
entity_id,timestamp,temp,pressure,vibration
unit_001,1,489.2,1.23,0.045
unit_001,2,490.1,1.24,0.047
unit_001,3,491.3,1.25,0.052
...
```

## Run Your First Analysis

```bash
# Basic analysis
python -m prism.entry_points.analyze \
    --input your_data.csv \
    --entity-col entity_id \
    --time-col timestamp

# Output goes to ./output/
```

## What You Get

After analysis completes, you'll find:

```
output/
├── vector_features.parquet    # 51 metrics per sensor
├── geometry_features.parquet  # Pairwise relationships
├── state_analysis.parquet     # Regime + coherence tracking
└── summary_report.html        # Human-readable findings
```

## View Results

Open `summary_report.html` in a browser to see:

- **Cohort discovery** — Which units behave similarly
- **Degradation curves** — How coherence changes over time
- **Key features** — Which relationships are most predictive

## Next Steps

- [System Overview](/docs/overview/) — Understand the architecture
- [CLI Reference](/docs/cli-reference/) — All command options
- [C-MAPSS Example](/docs/cmapss/) — Walk through a real dataset
