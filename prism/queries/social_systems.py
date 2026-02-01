"""
SOCIAL SYSTEMS ANALYSIS QUERIES
================================
Financial markets as social systems - crisis geometry and fragility analysis.

Dataset: Fama-French 10 Industry Portfolios (2000-2015)
Analysis Date: 2026-01-30
Version: PRISM v1.0 / ORTHON v1.0

KEY FINDINGS:
1. Three Crisis Geometries (not two)
2. The Fragility Paradox (chaos=healthy, order=panic)
3. Collapse/Recovery Asymmetry (2.5x faster collapse)
"""

import polars as pl
from pathlib import Path
from datetime import datetime, timedelta


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("/Users/jasonrudder/prism/data/fama_french")

# Date mapping for Fama-French (I=0 is Jan 2000)
START_DATE = datetime(2000, 1, 1)
END_DATE = datetime(2015, 12, 31)
DAYS_PER_OBS = (END_DATE - START_DATE).days / 767


def i_to_date(i: int) -> datetime:
    """Convert index I to datetime."""
    return START_DATE + timedelta(days=int(i * DAYS_PER_OBS))


def date_to_i(d: datetime) -> int:
    """Convert datetime to index I."""
    return int((d - START_DATE).days / DAYS_PER_OBS)


# =============================================================================
# 1. BASIC REPORTS
# =============================================================================

def basic_statistics():
    """Run basic statistical reports on observations."""
    obs = pl.read_parquet(DATA_DIR / "observations.parquet")
    stats = pl.read_parquet(DATA_DIR / "statistics.parquet")
    corr = pl.read_parquet(DATA_DIR / "correlation.parquet")

    print("OBSERVATIONS SUMMARY")
    print(f"Total rows: {len(obs):,}")
    print(f"Signals: {obs['signal_id'].n_unique()}")
    print(f"Units: {obs['unit_id'].n_unique()}")

    print("\nSTATISTICS")
    print(stats)

    print("\nCORRELATION MATRIX")
    print(corr)

    return obs, stats, corr


# =============================================================================
# 2. STABILITY ANALYSIS
# =============================================================================

def find_most_stable_geometry():
    """Find the most stable signal and pair in the dataset."""
    dynamics = pl.read_parquet(DATA_DIR / "dynamics.parquet")
    primitives = pl.read_parquet(DATA_DIR / "primitives.parquet")
    geometry = pl.read_parquet(DATA_DIR / "geometry.parquet")

    # Signal stability score
    stability = dynamics.select([
        "signal_id", "determinism", "laminarity", "recurrence_rate"
    ]).join(
        primitives.select(["signal_id", "basin_stability", "hurst"]),
        on="signal_id"
    )

    # Normalize and weight
    stability = stability.with_columns([
        ((pl.col("determinism") - pl.col("determinism").min()) /
         (pl.col("determinism").max() - pl.col("determinism").min())).alias("det_norm"),
        ((pl.col("basin_stability") - pl.col("basin_stability").min()) /
         (pl.col("basin_stability").max() - pl.col("basin_stability").min())).alias("basin_norm"),
    ])

    stability = stability.with_columns(
        (pl.col("det_norm") * 0.5 + pl.col("basin_norm") * 0.5).alias("stability_score")
    )

    # Pair stability (cointegration + correlation)
    geometry = geometry.with_columns([
        (1 - pl.col("coint_pvalue").clip(0, 1)).alias("coint_confidence"),
    ])

    geometry = geometry.with_columns(
        (pl.col("coint_confidence") * 0.4 + pl.col("correlation_abs") * 0.3 +
         pl.col("normalized_mi") * 0.3).alias("pair_stability")
    )

    most_stable_signal = stability.sort("stability_score", descending=True).head(1)
    most_stable_pair = geometry.sort("pair_stability", descending=True).head(1)

    print(f"MOST STABLE SIGNAL: {most_stable_signal['signal_id'][0]}")
    print(f"MOST STABLE PAIR: {most_stable_pair['signal_a'][0]} <-> {most_stable_pair['signal_b'][0]}")

    return stability, geometry


# =============================================================================
# 3. CRISIS TIMELINE
# =============================================================================

def fragility_timeline():
    """Analyze stability metrics through time, mapping to historical crises."""
    manifold = pl.read_parquet(DATA_DIR / "manifold.parquet")
    physics = pl.read_parquet(DATA_DIR / "physics.parquet")

    # Add dates
    manifold = manifold.with_columns(
        pl.col("I").map_elements(lambda x: i_to_date(x).strftime("%Y-%m-%d"),
                                  return_dtype=pl.String).alias("date")
    )

    # Key crisis dates
    crises = {
        "Dot-com Peak": datetime(2000, 3, 10),
        "9/11": datetime(2001, 9, 11),
        "GFC Lehman": datetime(2008, 9, 15),
        "Flash Crash": datetime(2010, 5, 6),
        "China 2015": datetime(2015, 8, 24),
    }

    print("CRISIS EARLY WARNING ANALYSIS")
    print("-" * 60)

    for name, event_date in crises.items():
        event_i = date_to_i(event_date)

        # Find peak velocity in 1 year before event
        pre_start = max(0, event_i - int(365 / DAYS_PER_OBS))
        pre_window = manifold.filter(
            (pl.col("I") >= pre_start) & (pl.col("I") < event_i)
        )

        if pre_window.height > 0:
            max_vel_row = pre_window.sort("manifold_velocity", descending=True).head(1)
            max_vel_i = max_vel_row["I"][0]
            max_vel = max_vel_row["manifold_velocity"][0]

            days_before = int((event_i - max_vel_i) * DAYS_PER_OBS)

            print(f"\n{name} ({event_date.strftime('%Y-%m-%d')}):")
            print(f"  Peak velocity: {max_vel:.1f}")
            print(f"  Early warning: {days_before} days before")

    return manifold, physics


# =============================================================================
# 4. CRISIS GEOMETRY ANALYSIS
# =============================================================================

def crisis_geometry_analysis():
    """
    Analyze pre-crisis geometry for historical events.

    Reports computed metrics (coherence, entropy, effective_dim) for each event.
    PRISM computes these values; interpretation is left to the analyst.

    THREE OBSERVED GEOMETRY PATTERNS:
    1. High coherence + Low entropy (ordered collapse, e.g., GFC)
    2. Low coherence + High entropy (disordered, e.g., Dot-com)
    3. Normal coherence/entropy (external shock, e.g., 9/11)
    """
    physics = pl.read_parquet(DATA_DIR / "physics.parquet")

    events = {
        "GFC 2008": datetime(2008, 9, 15),
        "9/11": datetime(2001, 9, 11),
        "Dot-com": datetime(2000, 3, 10),
        "Flash Crash": datetime(2010, 5, 6),
        "EU Debt": datetime(2011, 8, 5),
        "China 2015": datetime(2015, 8, 24),
    }

    print("CRISIS GEOMETRY ANALYSIS")
    print("=" * 70)
    print(f"{'Event':<15} {'Coherence':>10} {'Entropy':>10} {'Eff.Dim':>10}")
    print("-" * 70)

    for name, event_date in events.items():
        event_i = date_to_i(event_date)

        # Pre-event window (30-7 days before)
        pre_start = event_i - int(30 / DAYS_PER_OBS)
        pre_end = event_i - int(7 / DAYS_PER_OBS)

        pre_data = physics.filter((pl.col("I") >= pre_start) & (pl.col("I") <= pre_end))

        if pre_data.height > 0:
            ent = pre_data["eigenvalue_entropy"].mean()
            coh = pre_data["coherence"].mean()
            dim = pre_data["effective_dim"].mean()

            print(f"{name:<15} {coh:>10.4f} {ent:>10.4f} {dim:>10.2f}")
        else:
            print(f"{name:<15} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

    print("\nNote: PRISM computes values. Pattern interpretation is analyst's domain.")


# =============================================================================
# 5. COLLAPSE/RECOVERY ASYMMETRY
# =============================================================================

def deformation_asymmetry():
    """
    Analyze the speed asymmetry between collapse and recovery.

    KEY FINDING: Collapse is 2.5x faster than recovery.
    - Compression kurtosis: 125.7 (extreme fat tails)
    - Diversification kurtosis: 40.5 (less extreme)
    """
    physics = pl.read_parquet(DATA_DIR / "physics.parquet")
    manifold = pl.read_parquet(DATA_DIR / "manifold.parquet")

    physics = physics.sort("I")
    physics = physics.with_columns([
        (pl.col("coherence") - pl.col("coherence").shift(1)).alias("d_coherence"),
    ])

    # Compression (toward panic) vs Diversification (toward health)
    compression = physics.filter(pl.col("d_coherence") > 0)
    diversification = physics.filter(pl.col("d_coherence") < 0)

    print("DEFORMATION ASYMMETRY")
    print("=" * 60)

    print(f"\nCOMPRESSION (collapse):")
    print(f"  Mean speed: +{compression['d_coherence'].mean():.6f}")
    print(f"  Max speed:  +{compression['d_coherence'].max():.6f}")
    print(f"  Kurtosis:   {compression['d_coherence'].kurtosis():.1f}")

    print(f"\nDIVERSIFICATION (recovery):")
    print(f"  Mean speed: {diversification['d_coherence'].mean():.6f}")
    print(f"  Max speed:  {diversification['d_coherence'].min():.6f}")
    print(f"  Kurtosis:   {diversification['d_coherence'].kurtosis():.1f}")

    # Asymmetry ratios
    max_comp = abs(compression['d_coherence'].max())
    max_div = abs(diversification['d_coherence'].min())

    print(f"\nASYMMETRY:")
    print(f"  Max speed ratio: {max_comp/max_div:.1f}x (collapse faster)")

    return physics, manifold


# =============================================================================
# 6. THE FRAGILITY PARADOX
# =============================================================================

def fragility_paradox():
    """
    The Fragility Paradox: "Normal" is chaos, "Crisis" is order.

    NORMAL MARKET (healthy but fragile):
    - High entropy (0.50-0.70)
    - Low coherence (0.50-0.65)
    - High effective dimension (2.5-3.0)
    - Positive Lyapunov (chaotic)

    CRISIS MARKET (unhealthy but stable):
    - Low entropy (0.02-0.15)
    - High coherence (0.90-0.99)
    - Low effective dimension (1.0-1.2)
    - Zero Lyapunov (deterministic)
    """
    physics = pl.read_parquet(DATA_DIR / "physics.parquet")

    # Define state thresholds
    normal = physics.filter(
        (pl.col("coherence") < 0.70) & (pl.col("eigenvalue_entropy") > 0.40)
    )

    panic = physics.filter(
        (pl.col("coherence") > 0.90) & (pl.col("eigenvalue_entropy") < 0.20)
    )

    print("THE FRAGILITY PARADOX")
    print("=" * 60)

    print(f"\n'NORMAL' STATE (n={normal.height} periods):")
    print(f"  Mean coherence: {normal['coherence'].mean():.4f}")
    print(f"  Mean entropy:   {normal['eigenvalue_entropy'].mean():.4f}")
    print(f"  Mean eff. dim:  {normal['effective_dim'].mean():.2f}")
    print(f"  → CHAOTIC but DIVERSE")

    print(f"\n'PANIC' STATE (n={panic.height} periods):")
    print(f"  Mean coherence: {panic['coherence'].mean():.4f}")
    print(f"  Mean entropy:   {panic['eigenvalue_entropy'].mean():.4f}")
    print(f"  Mean eff. dim:  {panic['effective_dim'].mean():.2f}")
    print(f"  → ORDERED but UNIFORM")

    print(f"\nPARADOX: Chaos is healthy. Order is pathological.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SOCIAL SYSTEMS ANALYSIS - PRISM v1.0")
    print("=" * 70)

    # Run all analyses
    basic_statistics()
    print("\n")

    find_most_stable_geometry()
    print("\n")

    fragility_timeline()
    print("\n")

    crisis_geometry_analysis()
    print("\n")

    deformation_asymmetry()
    print("\n")

    fragility_paradox()
