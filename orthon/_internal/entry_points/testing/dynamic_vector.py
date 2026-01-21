"""
PRISM Dynamic Vector Runner
============================

Universal pipeline for computing dynamic vectors through coupling space.

THIS IS AN ORCHESTRATOR - NO COMPUTATION LOGIC HERE.
All math is delegated to canonical PRISM engines.

Layers:
    1. Observation  → Raw signal data
    2. Vector       → Behavioral metrics per signal (entropy, hurst, etc.)
    3. Geometry     → Pairwise coupling space (N signals → N*(N-1)/2 dimensions)
                      Engines: correlation, transfer_entropy, cointegration
    4. Dynamic State → Position in coupling space per window
    5. Dynamic Vector → Velocity/direction through coupling space

Output: Long-format parquet with dynamic vectors per entity per window.

Usage:
    python dynamic_vector.py --domain mimic --input data/vectors.parquet
    python dynamic_vector.py --domain turbofan --input data/vectors.parquet
    python dynamic_vector.py --domain cmapss --input data/vectors.parquet

The same orchestration applies regardless of domain. Only the input data changes.
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path
from datetime import datetime
from itertools import combinations
from typing import List, Tuple, Optional, Dict, Any

# =============================================================================
# ENGINE IMPORTS - All computation delegated to these
# =============================================================================

try:
    from orthon._internal.engines.state.transfer_entropy import compute_transfer_entropy
    HAS_TE = True
except ImportError:
    HAS_TE = False

try:
    from orthon._internal.engines.cointegration import compute_cointegration
    HAS_COINT = True
except ImportError:
    HAS_COINT = False

try:
    from orthon._internal.engines.correlation import compute_rolling_correlation
    HAS_CORR = True
except ImportError:
    HAS_CORR = False


# =============================================================================
# LAYER 3: GEOMETRY (Pairwise Coupling Space)
# =============================================================================

def compute_pairwise_geometry(
    series_a: np.ndarray,
    series_b: np.ndarray,
    pair_name: str,
) -> Dict[str, float]:
    """
    Compute pairwise geometry metrics by calling PRISM engines.

    This function is a DISPATCHER - no math here, just engine calls.

    Engines called:
        - correlation: Pearson correlation
        - transfer_entropy: Information flow A→B and B→A
        - cointegration: Long-run equilibrium test

    Args:
        series_a: Signal for signal A
        series_b: Signal for signal B
        pair_name: Name for this pair (e.g., "HR_BP")

    Returns:
        Dict with coupling metrics from all engines
    """
    results = {}

    # Correlation (fallback if engine not available)
    if HAS_CORR:
        corr_result = compute_rolling_correlation(series_a, series_b)
        results[f"corr_{pair_name}"] = corr_result.get("correlation", None)
    else:
        # Minimal fallback - just numpy correlation
        if len(series_a) >= 3 and np.std(series_a) > 0 and np.std(series_b) > 0:
            corr = np.corrcoef(series_a, series_b)[0, 1]
            results[f"corr_{pair_name}"] = float(corr) if not np.isnan(corr) else None
        else:
            results[f"corr_{pair_name}"] = None

    # Transfer Entropy
    if HAS_TE:
        te_result = compute_transfer_entropy(series_a, series_b)
        results[f"te_a2b_{pair_name}"] = te_result.get("te_a_to_b", None)
        results[f"te_b2a_{pair_name}"] = te_result.get("te_b_to_a", None)
        results[f"te_net_{pair_name}"] = te_result.get("te_net", None)

    # Cointegration
    if HAS_COINT:
        coint_result = compute_cointegration(series_a, series_b)
        results[f"coint_{pair_name}"] = 1.0 if coint_result.get("is_cointegrated", False) else 0.0
        results[f"coint_pval_{pair_name}"] = coint_result.get("pvalue", None)

    return results


def compute_geometry_for_window(
    window_data: Dict[str, np.ndarray],
    signals: List[str],
) -> Dict[str, float]:
    """
    Compute full pairwise geometry for a single window.

    Calls compute_pairwise_geometry for each signal pair.

    Args:
        window_data: Dict mapping signal_id -> numpy array of values
        signals: List of signal IDs to compare

    Returns:
        Dict with all pairwise metrics for this window
    """
    pairs = list(combinations(signals, 2))
    results = {}

    for ind_a, ind_b in pairs:
        if ind_a not in window_data or ind_b not in window_data:
            continue

        series_a = window_data[ind_a]
        series_b = window_data[ind_b]

        # Skip if insufficient data
        if len(series_a) < 3 or len(series_b) < 3:
            continue

        # Align lengths
        min_len = min(len(series_a), len(series_b))
        series_a = series_a[:min_len]
        series_b = series_b[:min_len]

        pair_name = f"{ind_a}_{ind_b}"
        pair_metrics = compute_pairwise_geometry(series_a, series_b, pair_name)
        results.update(pair_metrics)

    return results


def compute_rolling_geometry(
    df: pl.DataFrame,
    entity_col: str,
    signal_col: str,
    window_col: str,
    value_col: str,
    rolling_windows: int = 6,
) -> pl.DataFrame:
    """
    Compute rolling pairwise geometry by calling engines on each window.

    This is the ORCHESTRATOR - it prepares data and calls engines.
    """
    signals = sorted(df[signal_col].unique().to_list())
    n_pairs = len(list(combinations(signals, 2)))

    print(f"  Signals: {len(signals)} → {n_pairs} pairs")
    print(f"  Engines: correlation={HAS_CORR}, transfer_entropy={HAS_TE}, cointegration={HAS_COINT}")
    print(f"  Rolling window: {rolling_windows}")

    # Pivot to wide
    wide = df.pivot(
        index=[entity_col, window_col],
        on=signal_col,
        values=value_col,
        aggregate_function="mean"
    ).sort([entity_col, window_col])

    results = []

    for entity in wide[entity_col].unique().to_list():
        entity_data = wide.filter(pl.col(entity_col) == entity).sort(window_col)
        windows = entity_data[window_col].to_list()

        for i, window in enumerate(windows):
            if i < rolling_windows - 1:
                continue  # Need enough history

            # Get rolling window of data
            start_idx = i - rolling_windows + 1
            window_slice = entity_data.slice(start_idx, rolling_windows)

            # Build window_data dict for engine calls
            window_data = {}
            for ind in signals:
                if ind in window_slice.columns:
                    vals = window_slice[ind].drop_nulls().to_numpy()
                    if len(vals) >= 3:
                        window_data[ind] = vals

            # Call engines via orchestrator
            geometry_metrics = compute_geometry_for_window(window_data, signals)

            # Build result row
            row = {
                entity_col: entity,
                window_col: window,
            }
            row.update(geometry_metrics)
            results.append(row)

    return pl.DataFrame(results)


# =============================================================================
# LAYER 4: DYNAMIC STATE (Position in Coupling Space)
# =============================================================================

def compute_dynamic_state(
    geometry_df: pl.DataFrame,
    entity_col: str,
    window_col: str,
) -> pl.DataFrame:
    """
    Convert geometry (coupling values) to state metrics.

    Geometry columns come from engines:
        - corr_*: Correlation engine
        - te_*: Transfer entropy engine
        - coint_*: Cointegration engine

    For each entity at each window, compute:
    - State magnitude (L2 norm of correlation vector)
    - Mean correlation (average across all pairs)
    - Mean TE (information flow)
    - Cointegration fraction (% of pairs cointegrated)
    """
    # Find geometry columns by engine type
    corr_cols = [c for c in geometry_df.columns if c.startswith("corr_")]
    te_cols = [c for c in geometry_df.columns if c.startswith("te_") and not c.startswith("te_net")]
    te_net_cols = [c for c in geometry_df.columns if c.startswith("te_net_")]
    coint_cols = [c for c in geometry_df.columns if c.startswith("coint_") and not c.startswith("coint_pval")]

    all_metric_cols = corr_cols + te_net_cols + coint_cols

    if not all_metric_cols:
        raise ValueError("No geometry columns found (corr_*, te_*, coint_*)")

    print(f"  State metrics from: {len(corr_cols)} corr, {len(te_net_cols)} te_net, {len(coint_cols)} coint")

    results = []

    for row in geometry_df.iter_rows(named=True):
        entity = row[entity_col]
        window = row[window_col]

        # Correlation state
        corr_values = [row[c] for c in corr_cols if row[c] is not None]

        # Transfer entropy state
        te_net_values = [row[c] for c in te_net_cols if row[c] is not None]

        # Cointegration state
        coint_values = [row[c] for c in coint_cols if row[c] is not None]

        if len(corr_values) == 0:
            continue

        corr_array = np.array(corr_values)

        state_row = {
            entity_col: entity,
            window_col: window,
            # Correlation metrics
            "state_corr_magnitude": float(np.linalg.norm(corr_array)),
            "state_corr_mean": float(np.mean(corr_array)),
            "state_corr_min": float(np.min(corr_array)),
            "state_corr_max": float(np.max(corr_array)),
            "state_corr_std": float(np.std(corr_array)),
            "state_n_pairs": len(corr_values),
        }

        # Transfer entropy metrics
        if te_net_values:
            te_array = np.array(te_net_values)
            state_row["state_te_mean"] = float(np.mean(te_array))
            state_row["state_te_std"] = float(np.std(te_array))
            state_row["state_te_magnitude"] = float(np.linalg.norm(te_array))

        # Cointegration metrics
        if coint_values:
            coint_array = np.array(coint_values)
            state_row["state_coint_fraction"] = float(np.mean(coint_array))  # % cointegrated
            state_row["state_coint_count"] = int(np.sum(coint_array))

        # Preserve individual geometry values for vector computation
        for c in corr_cols:
            state_row[c] = row[c]
        for c in te_net_cols:
            state_row[c] = row[c]
        for c in coint_cols:
            state_row[c] = row[c]

        results.append(state_row)

    return pl.DataFrame(results)


# =============================================================================
# LAYER 5: DYNAMIC VECTOR (Velocity Through Coupling Space)
# =============================================================================

def compute_dynamic_vector(
    state_df: pl.DataFrame,
    entity_col: str,
    window_col: str,
) -> pl.DataFrame:
    """
    Compute velocity and direction through coupling space.

    For each entity between consecutive windows:
    - Velocity magnitude: how fast is coupling changing?
    - Velocity direction: which couplings are driving the change?
    - Acceleration: is the change speeding up?

    Tracks changes in:
    - Correlation (corr_*)
    - Transfer entropy (te_net_*)
    - Cointegration (coint_*)
    """
    # Find geometry columns
    corr_cols = [c for c in state_df.columns if c.startswith("corr_")]
    te_cols = [c for c in state_df.columns if c.startswith("te_net_")]
    coint_cols = [c for c in state_df.columns if c.startswith("coint_") and not c.startswith("coint_pval")]

    print(f"  Computing dynamic vectors: {len(corr_cols)} corr, {len(te_cols)} te, {len(coint_cols)} coint")

    results = []

    for entity in state_df[entity_col].unique().to_list():
        entity_data = state_df.filter(pl.col(entity_col) == entity).sort(window_col)

        if len(entity_data) < 2:
            continue

        rows = entity_data.to_dicts()
        prev_velocity_corr = None
        prev_velocity_te = None

        for i in range(1, len(rows)):
            prev = rows[i - 1]
            curr = rows[i]

            # Correlation velocity vector
            velocity_corr = []
            velocity_corr_components = {}
            for c in corr_cols:
                if prev[c] is not None and curr[c] is not None:
                    delta = curr[c] - prev[c]
                    pair_name = c.replace("corr_", "")
                    velocity_corr_components[f"v_corr_{pair_name}"] = delta
                    velocity_corr.append(delta)

            # TE velocity vector
            velocity_te = []
            velocity_te_components = {}
            for c in te_cols:
                if prev[c] is not None and curr[c] is not None:
                    delta = curr[c] - prev[c]
                    pair_name = c.replace("te_net_", "")
                    velocity_te_components[f"v_te_{pair_name}"] = delta
                    velocity_te.append(delta)

            # Cointegration change (binary)
            coint_changes = []
            for c in coint_cols:
                if prev[c] is not None and curr[c] is not None:
                    coint_changes.append(curr[c] - prev[c])

            if len(velocity_corr) == 0:
                continue

            velocity_corr_array = np.array(velocity_corr)
            velocity_corr_mag = float(np.linalg.norm(velocity_corr_array))

            # Direction: normalized velocity vector (unit vector)
            # +1 = all correlations increasing (coupling), -1 = all decreasing (decoupling)
            if velocity_corr_mag > 1e-10:
                direction = velocity_corr_array / velocity_corr_mag
                stable_direction = np.ones(len(velocity_corr_array)) / np.sqrt(len(velocity_corr_array))
                direction_cosine_corr = float(np.dot(direction, stable_direction))
            else:
                direction_cosine_corr = 0.0

            # Acceleration
            if prev_velocity_corr is not None:
                acceleration_corr = velocity_corr_mag - prev_velocity_corr
            else:
                acceleration_corr = 0.0
            prev_velocity_corr = velocity_corr_mag

            # TE metrics
            velocity_te_mag = 0.0
            direction_cosine_te = 0.0
            if velocity_te:
                velocity_te_array = np.array(velocity_te)
                velocity_te_mag = float(np.linalg.norm(velocity_te_array))
                if velocity_te_mag > 1e-10:
                    direction_te = velocity_te_array / velocity_te_mag
                    stable_te = np.ones(len(velocity_te_array)) / np.sqrt(len(velocity_te_array))
                    direction_cosine_te = float(np.dot(direction_te, stable_te))

            # Build output row
            vector_row = {
                entity_col: entity,
                window_col: curr[window_col],
                # Correlation dynamics
                "velocity_corr_magnitude": velocity_corr_mag,
                "velocity_corr_mean": float(np.mean(velocity_corr_array)),
                "direction_cosine_corr": direction_cosine_corr,
                "acceleration_corr": acceleration_corr,
                # TE dynamics
                "velocity_te_magnitude": velocity_te_mag,
                "direction_cosine_te": direction_cosine_te,
                # Cointegration dynamics
                "coint_change_sum": float(sum(coint_changes)) if coint_changes else 0.0,
                # State snapshots
                "state_corr_mean": curr.get("state_corr_mean", None),
                "state_te_mean": curr.get("state_te_mean", None),
                "state_coint_fraction": curr.get("state_coint_fraction", None),
            }

            # Add individual velocity components
            vector_row.update(velocity_corr_components)
            vector_row.update(velocity_te_components)

            results.append(vector_row)

    return pl.DataFrame(results)


# =============================================================================
# LONG FORMAT OUTPUT
# =============================================================================

def to_long_format(
    vector_df: pl.DataFrame,
    domain: str,
    entity_col: str,
    window_col: str,
) -> pl.DataFrame:
    """
    Convert wide dynamic vector dataframe to long format.

    Output schema:
        domain | entity_id | window_end | metric | value
    """
    metric_cols = [c for c in vector_df.columns if c not in [entity_col, window_col]]

    rows = []
    for row in vector_df.iter_rows(named=True):
        entity = row[entity_col]
        window = row[window_col]

        for metric in metric_cols:
            value = row[metric]
            if value is not None:
                rows.append({
                    "domain": domain,
                    "entity_id": str(entity),
                    "window_end": window,
                    "metric": metric,
                    "value": float(value),
                })

    return pl.DataFrame(rows)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(
    input_path: Path,
    output_path: Path,
    domain: str,
    entity_col: str = "entity_id",
    signal_col: str = "signal_id",
    window_col: str = "window_end",
    value_col: str = "value",
    rolling_window: int = 6,
    long_format: bool = True,
) -> pl.DataFrame:
    """
    Run the full dynamic vector pipeline.

    THIS IS AN ORCHESTRATOR - calls engines, no computation logic.

    Layers:
        Input (vectors) → Geometry (engines) → State → Dynamic Vector → Output
    """
    print("=" * 70)
    print(f"PRISM DYNAMIC VECTOR PIPELINE")
    print(f"Domain: {domain}")
    print("=" * 70)

    # Load input
    print(f"\n[1/5] Loading: {input_path}")
    df = pl.read_parquet(input_path)
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {df.columns}")

    # Detect column names if not standard
    if entity_col not in df.columns:
        for candidate in ["patient_id", "unit_id", "subject_id", "entity"]:
            if candidate in df.columns:
                entity_col = candidate
                break

    if signal_col not in df.columns:
        for candidate in ["vital", "sensor", "signal", "variable"]:
            if candidate in df.columns:
                signal_col = candidate
                break

    print(f"  Entity column: {entity_col}")
    print(f"  Signal column: {signal_col}")
    print(f"  Window column: {window_col}")
    print(f"  Value column: {value_col}")

    entities = df[entity_col].unique()
    signals = df[signal_col].unique()
    print(f"  Entities: {len(entities)}")
    print(f"  Signals: {len(signals)} → {signals.to_list()}")

    # Layer 3: Geometry (CALLS ENGINES)
    print(f"\n[2/5] Computing Geometry (calling engines: corr, TE, coint)")
    geometry_df = compute_rolling_geometry(
        df, entity_col, signal_col, window_col, value_col, rolling_window
    )
    print(f"  Geometry rows: {len(geometry_df):,}")

    # Layer 4: Dynamic State
    print(f"\n[3/5] Computing Dynamic State")
    state_df = compute_dynamic_state(geometry_df, entity_col, window_col)
    print(f"  State rows: {len(state_df):,}")

    # Layer 5: Dynamic Vector
    print(f"\n[4/5] Computing Dynamic Vector")
    vector_df = compute_dynamic_vector(state_df, entity_col, window_col)
    print(f"  Vector rows: {len(vector_df):,}")

    # Output
    print(f"\n[5/5] Writing Output")
    if long_format:
        output_df = to_long_format(vector_df, domain, entity_col, window_col)
        print(f"  Long format rows: {len(output_df):,}")
    else:
        output_df = vector_df.with_columns(pl.lit(domain).alias("domain"))

    output_df.write_parquet(output_path)
    print(f"  Saved: {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    if long_format:
        metrics = output_df["metric"].unique().to_list()
        print(f"Metrics computed: {len(metrics)}")
        for m in sorted(metrics):
            print(f"  - {m}")

    return output_df


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PRISM Dynamic Vector Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python dynamic_vector.py --domain mimic --input vitals.parquet
    python dynamic_vector.py --domain turbofan --input sensors.parquet
    python dynamic_vector.py --domain cmapss --input signals.parquet --wide
        """
    )

    parser.add_argument("--domain", required=True, help="Domain name (mimic, turbofan, cmapss)")
    parser.add_argument("--input", required=True, help="Input parquet with signal vectors")
    parser.add_argument("--output", help="Output parquet path (default: dynamic_vectors.parquet)")
    parser.add_argument("--entity-col", default="entity_id", help="Entity column name")
    parser.add_argument("--signal-col", default="signal_id", help="Signal column name")
    parser.add_argument("--window-col", default="window_end", help="Window column name")
    parser.add_argument("--value-col", default="value", help="Value column name")
    parser.add_argument("--rolling-window", type=int, default=6, help="Rolling window size for correlation")
    parser.add_argument("--wide", action="store_true", help="Output wide format instead of long")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return

    output_path = Path(args.output) if args.output else Path(f"dynamic_vectors_{args.domain}.parquet")

    run_pipeline(
        input_path=input_path,
        output_path=output_path,
        domain=args.domain,
        entity_col=args.entity_col,
        signal_col=args.signal_col,
        window_col=args.window_col,
        value_col=args.value_col,
        rolling_window=args.rolling_window,
        long_format=not args.wide,
    )


if __name__ == "__main__":
    main()
