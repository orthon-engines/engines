"""
PRISM Dynamical Systems
=======================

Three-layer hierarchical cohort discovery for dynamical system classification.
This is the CLASSIFICATION layer - it answers "what type of dynamical system is this entity?"

Layers:
1. SIGNAL TYPOLOGY → Signal Types (behavioral fingerprints)
2. BEHAVIORAL GEOMETRY → Structural Groups (correlation patterns)
3. PHASE STATE → Temporal Cohorts WITHIN structural groups

Key insight: Temporal cohorts must be discovered within structural groups,
otherwise structural differences confound trajectory differences.

Input: signal_typology.parquet, behavioral_geometry.parquet, phase_state.parquet
Output: dynamical_systems.parquet

Pipeline:
    raw → signal_typology → behavioral_geometry → phase_state → dynamical_systems

Usage:
    python -m prism.entry_points.dynamical_systems
    python -m prism.entry_points.dynamical_systems --data-path /path/to/data
"""

import argparse
import polars as pl
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import json
import os
import warnings

warnings.filterwarnings('ignore')

DATA_PATH = Path(os.environ.get("ORTHON_DATA_PATH", "data"))


def load_layer_data(data_path: Path) -> dict:
    """Load vector, geometry, state layers."""
    data = {}
    for name in ['vector', 'geometry', 'state']:
        path = data_path / f'{name}.parquet'
        if path.exists():
            data[name] = pl.read_parquet(path)
        else:
            data[name] = None
    return data


# =============================================================================
# STAGE 1: SIGNAL TYPES
# =============================================================================

def discover_signal_types(vector: pl.DataFrame, max_types: int = 10) -> dict:
    """
    Discover signal types by behavioral fingerprint.

    Clusters signals based on their statistical profiles across all windows.
    Signals with similar behavioral patterns = same type.

    Returns:
        {
            'n_types': int,
            'confidence': float (silhouette score),
            'mapping': {signal_id: type_id},
            'type_members': {type_id: [signal_ids]}
        }
    """
    if vector is None:
        return {'n_types': 0, 'confidence': 0, 'mapping': {}, 'type_members': {}}

    signal_col = 'source_signal' if 'source_signal' in vector.columns else 'signal_id'
    signals = sorted(vector[signal_col].unique().to_list())
    metrics = sorted(vector['engine'].unique().to_list())

    # Build fingerprint matrix: mean value per metric per signal
    fingerprints = []
    valid_signals = []

    for sig in signals:
        sig_data = vector.filter(pl.col(signal_col) == sig)
        fp = []
        for metric in metrics:
            val = sig_data.filter(pl.col('engine') == metric)['value'].mean()
            fp.append(val if val is not None else 0.0)

        if not all(np.isnan(v) or v == 0 for v in fp):
            fingerprints.append(fp)
            valid_signals.append(sig)

    if len(fingerprints) < 2:
        return {
            'n_types': 1,
            'confidence': 0,
            'mapping': {s: 'type_0' for s in signals},
            'type_members': {'type_0': signals}
        }

    X = np.nan_to_num(np.array(fingerprints))
    X_scaled = StandardScaler().fit_transform(X)

    # Find optimal k
    best_k, best_score = 1, -1
    for k in range(2, min(max_types + 1, len(valid_signals))):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        if len(set(labels)) > 1:
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_k, best_score = k, score

    # Final clustering
    if best_k > 1:
        labels = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X_scaled)
    else:
        labels = [0] * len(valid_signals)

    # Build result
    mapping = {sig: f"type_{l}" for sig, l in zip(valid_signals, labels)}
    type_members = {}
    for t in range(best_k):
        type_members[f"type_{t}"] = [s for s, l in zip(valid_signals, labels) if l == t]

    return {
        'n_types': best_k,
        'confidence': best_score,
        'mapping': mapping,
        'type_members': type_members
    }


# =============================================================================
# STAGE 2: STRUCTURAL GROUPS
# =============================================================================

def discover_structural_groups(geometry: pl.DataFrame, max_groups: int = 10) -> dict:
    """
    Discover structural groups by correlation pattern similarity.

    Entities with similar pairwise relationship structures = same group.
    This answers: "What type of entity is this?"

    Returns:
        {
            'n_groups': int,
            'confidence': float,
            'mapping': {entity_id: group_id},
            'group_members': {group_id: [entity_ids]}
        }
    """
    if geometry is None:
        return {'n_groups': 0, 'confidence': 0, 'mapping': {}, 'group_members': {}}

    entities = sorted(geometry['entity_id'].unique().to_list())

    # Build correlation fingerprint per entity
    if 'signal_a' in geometry.columns:
        # Long format
        pairs = [(r['signal_a'], r['signal_b'])
                 for r in geometry.select(['signal_a', 'signal_b']).unique().iter_rows(named=True)]

        fingerprints = []
        for entity in entities:
            entity_data = geometry.filter(pl.col('entity_id') == entity)
            fp = []
            for sig_a, sig_b in pairs:
                pair_data = entity_data.filter(
                    (pl.col('signal_a') == sig_a) & (pl.col('signal_b') == sig_b)
                )
                corr_col = 'pearson' if 'pearson' in pair_data.columns else 'value'
                if len(pair_data) > 0:
                    val = pair_data[corr_col].mean()
                    fp.append(float(val) if val is not None else 0.0)
                else:
                    fp.append(0.0)
            fingerprints.append(fp)
    else:
        # Wide format - only numeric columns
        numeric_types = [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64]
        corr_cols = [c for c in geometry.columns
                    if c not in ['entity_id', 'timestamp', 'window', 'mode_id', 'computed_at', 'signal_ids']
                    and geometry[c].dtype in numeric_types]
        agg = geometry.group_by('entity_id').agg([pl.col(c).mean().alias(c) for c in corr_cols])
        fingerprints = []
        for entity in entities:
            row = agg.filter(pl.col('entity_id') == entity)
            fp = []
            for c in corr_cols:
                if len(row) > 0:
                    val = row[c].to_list()[0]
                    fp.append(val if val is not None else 0.0)
                else:
                    fp.append(0.0)
            fingerprints.append(fp)

    X = np.array(fingerprints, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if X.std() == 0 or len(entities) < 2:
        return {
            'n_groups': 1,
            'confidence': 0,
            'mapping': {e: 'group_0' for e in entities},
            'group_members': {'group_0': entities}
        }

    X_scaled = StandardScaler().fit_transform(X)

    # Find optimal k
    best_k, best_score = 1, -1
    for k in range(2, min(max_groups + 1, len(entities))):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        if len(set(labels)) > 1:
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_k, best_score = k, score

    # Threshold for meaningful separation
    if best_score < 0.1:
        return {
            'n_groups': 1,
            'confidence': best_score,
            'mapping': {e: 'group_0' for e in entities},
            'group_members': {'group_0': entities}
        }

    labels = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X_scaled)

    mapping = {e: f"group_{l}" for e, l in zip(entities, labels)}
    group_members = {}
    for g in range(best_k):
        group_members[f"group_{g}"] = [e for e, l in zip(entities, labels) if l == g]

    return {
        'n_groups': best_k,
        'confidence': best_score,
        'mapping': mapping,
        'group_members': group_members
    }


# =============================================================================
# STAGE 3: TEMPORAL COHORTS (Within Structural Groups)
# =============================================================================

def discover_temporal_cohorts(state: pl.DataFrame, structural_groups: dict,
                              max_cohorts_per_group: int = 5) -> dict:
    """
    Discover temporal cohorts WITHIN each structural group.

    Key insight: Cluster by degradation trajectory, not by structural similarity.
    This is the confirmatory step that requires all layers.

    Returns:
        {
            'n_cohorts': int,
            'mapping': {entity_id: cohort_id},
            'cohorts_by_group': {group_id: [cohort_ids]},
            'cohort_profiles': {cohort_id: {trajectory, slope, members}}
        }
    """
    if state is None or not structural_groups.get('group_members'):
        return {'n_cohorts': 0, 'mapping': {}, 'cohorts_by_group': {}, 'cohort_profiles': {}}

    result = {
        'n_cohorts': 0,
        'mapping': {},
        'cohorts_by_group': {},
        'cohort_profiles': {}
    }

    global_cohort_id = 0

    for group_name, group_members in structural_groups['group_members'].items():
        if len(group_members) < 2:
            # Single entity = single cohort
            cohort_name = f"cohort_{global_cohort_id}"
            for entity in group_members:
                result['mapping'][entity] = cohort_name
            result['cohorts_by_group'][group_name] = [cohort_name]
            result['cohort_profiles'][cohort_name] = {
                'trajectory': 'unknown',
                'slope': 0.0,
                'members': group_members
            }
            global_cohort_id += 1
            continue

        # Build trajectory fingerprint for each entity
        fingerprints = []
        valid_entities = []

        for entity in group_members:
            entity_data = state.filter(pl.col('entity_id') == entity).sort('timestamp')
            if len(entity_data) < 3:
                continue

            features = []
            # Try multiple possible column sets (different state schemas)
            trajectory_cols = [
                # Primary: trajectory dynamics columns
                'speed', 'curvature', 'mean_velocity', 'mean_acceleration',
                # Alternative: coherence-based columns
                'coherence', 'healthy_distance', 'degradation_score', 'coherence_velocity',
            ]
            available_cols = [c for c in trajectory_cols if c in entity_data.columns]

            for col in available_cols[:4]:  # Use up to 4 columns
                values = entity_data[col].drop_nulls().to_numpy()
                if len(values) < 2:
                    features.extend([0, 0, 0, 0, 0])
                    continue

                features.append(float(values[0]))                    # Start
                features.append(float(values[-1]))                   # End
                features.append(float(np.mean(values)))              # Mean
                features.append(float(values[-1] - values[0]))       # Change
                features.append(float(np.polyfit(range(len(values)), values, 1)[0]))  # Slope

            if features:
                fingerprints.append(features)
                valid_entities.append(entity)

        if len(fingerprints) < 2:
            cohort_name = f"cohort_{global_cohort_id}"
            for entity in group_members:
                result['mapping'][entity] = cohort_name
            result['cohorts_by_group'][group_name] = [cohort_name]
            result['cohort_profiles'][cohort_name] = {
                'trajectory': 'unknown',
                'slope': 0.0,
                'members': group_members
            }
            global_cohort_id += 1
            continue

        X = np.nan_to_num(np.array(fingerprints))
        if X.std() == 0:
            cohort_name = f"cohort_{global_cohort_id}"
            for entity in valid_entities:
                result['mapping'][entity] = cohort_name
            result['cohorts_by_group'][group_name] = [cohort_name]
            result['cohort_profiles'][cohort_name] = {
                'trajectory': 'stable',
                'slope': 0.0,
                'members': valid_entities
            }
            global_cohort_id += 1
            continue

        X_scaled = StandardScaler().fit_transform(X)

        # Find optimal k
        best_k, best_score = 1, -1
        for k in range(2, min(max_cohorts_per_group + 1, len(valid_entities))):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            if len(set(labels)) > 1:
                score = silhouette_score(X_scaled, labels)
                if score > best_score:
                    best_k, best_score = k, score

        # Threshold for meaningful separation
        if best_score < 0.15:
            cohort_name = f"cohort_{global_cohort_id}"
            for entity in valid_entities:
                result['mapping'][entity] = cohort_name
            result['cohorts_by_group'][group_name] = [cohort_name]

            avg_slope = np.mean([fp[4] if len(fp) > 4 else 0 for fp in fingerprints])
            trajectory = _classify_trajectory(avg_slope)
            result['cohort_profiles'][cohort_name] = {
                'trajectory': trajectory,
                'slope': avg_slope,
                'members': valid_entities
            }
            global_cohort_id += 1
            continue

        # Cluster
        labels = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X_scaled)

        group_cohorts = []
        for c in range(best_k):
            cohort_name = f"cohort_{global_cohort_id}"
            members = [e for e, l in zip(valid_entities, labels) if l == c]

            for entity in members:
                result['mapping'][entity] = cohort_name

            member_fps = [fingerprints[valid_entities.index(m)] for m in members]
            avg_slope = np.mean([fp[4] if len(fp) > 4 else 0 for fp in member_fps])
            trajectory = _classify_trajectory(avg_slope)

            result['cohort_profiles'][cohort_name] = {
                'trajectory': trajectory,
                'slope': avg_slope,
                'members': members
            }

            group_cohorts.append(cohort_name)
            global_cohort_id += 1

        result['cohorts_by_group'][group_name] = group_cohorts

    result['n_cohorts'] = global_cohort_id
    return result


def _classify_trajectory(slope: float) -> str:
    """
    Classify trajectory based on slope.

    Note: Positive slope in state space = faster movement = accelerating degradation.
    This is the OPPOSITE of what you might expect - higher speed means worse health.
    """
    if slope > 1.0:
        return 'catastrophic'  # Rapid acceleration toward failure
    elif slope > 0.1:
        return 'failing'       # Clear degradation acceleration
    elif slope > 0.01:
        return 'degrading'     # Mild degradation
    elif slope < -0.01:
        return 'recovering'    # Unusual - decelerating (rare)
    else:
        return 'stable'        # Normal operation


# =============================================================================
# OUTPUT
# =============================================================================

def build_cohorts_output(signal_types: dict, structural_groups: dict,
                         temporal_cohorts: dict) -> pl.DataFrame:
    """Build cohorts.parquet output."""

    # Entity-level assignments
    entity_rows = []
    all_entities = set(structural_groups.get('mapping', {}).keys()) | \
                   set(temporal_cohorts.get('mapping', {}).keys())

    for entity in sorted(all_entities):
        entity_rows.append({
            'entity_id': entity,
            'structural_group': structural_groups.get('mapping', {}).get(entity, 'unknown'),
            'temporal_cohort': temporal_cohorts.get('mapping', {}).get(entity, 'unknown'),
        })

    entities_df = pl.DataFrame(entity_rows) if entity_rows else pl.DataFrame()

    # Signal-level assignments
    signal_rows = []
    for sig, stype in signal_types.get('mapping', {}).items():
        signal_rows.append({
            'signal_id': sig,
            'signal_type': stype
        })

    signals_df = pl.DataFrame(signal_rows) if signal_rows else pl.DataFrame()

    return entities_df, signals_df


def main():
    parser = argparse.ArgumentParser(description="Ørthon Cohort Discovery")
    parser.add_argument('--data-path', type=str, default=str(DATA_PATH),
                        help='Path to data directory')
    args = parser.parse_args()

    data_path = Path(args.data_path)

    print("="*70)
    print("ØRTHON COHORT DISCOVERY")
    print("="*70)
    print(f"\nData path: {data_path}")

    # Load data
    print("\nLoading layer data...")
    data = load_layer_data(data_path)
    for name, df in data.items():
        print(f"  {name}: {len(df):,} rows" if df is not None else f"  {name}: NOT FOUND")

    # Stage 1: Signal Types
    print("\n" + "-"*70)
    print("STAGE 1: Signal Type Discovery")
    print("-"*70)
    signal_types = discover_signal_types(data['vector'])
    print(f"  Types found: {signal_types['n_types']} (silhouette={signal_types['confidence']:.3f})")
    for t, members in signal_types.get('type_members', {}).items():
        print(f"    {t}: {members}")

    # Stage 2: Structural Groups
    print("\n" + "-"*70)
    print("STAGE 2: Structural Group Discovery")
    print("-"*70)
    structural_groups = discover_structural_groups(data['geometry'])
    print(f"  Groups found: {structural_groups['n_groups']} (silhouette={structural_groups['confidence']:.3f})")
    for g, members in structural_groups.get('group_members', {}).items():
        print(f"    {g}: {members}")

    # Stage 3: Temporal Cohorts
    print("\n" + "-"*70)
    print("STAGE 3: Temporal Cohort Discovery (Within Groups)")
    print("-"*70)
    temporal_cohorts = discover_temporal_cohorts(data['state'], structural_groups)
    print(f"  Cohorts found: {temporal_cohorts['n_cohorts']}")
    for cohort, profile in temporal_cohorts.get('cohort_profiles', {}).items():
        print(f"    {cohort}: {profile['members']} ({profile['trajectory']}, slope={profile['slope']:.4f})")

    # Build output
    entities_df, signals_df = build_cohorts_output(signal_types, structural_groups, temporal_cohorts)

    # Save
    entities_df.write_parquet(data_path / 'cohorts.parquet')
    print(f"\nSaved: {data_path}/cohorts.parquet ({len(entities_df)} entities)")

    if len(signals_df) > 0:
        signals_df.write_parquet(data_path / 'signal_types.parquet')
        print(f"Saved: {data_path}/signal_types.parquet ({len(signals_df)} signals)")

    # Save metadata
    metadata = {
        'signal_types': {
            'n_types': signal_types['n_types'],
            'confidence': signal_types['confidence'],
            'type_members': signal_types.get('type_members', {})
        },
        'structural_groups': {
            'n_groups': structural_groups['n_groups'],
            'confidence': structural_groups['confidence'],
            'group_members': structural_groups.get('group_members', {})
        },
        'temporal_cohorts': {
            'n_cohorts': temporal_cohorts['n_cohorts'],
            'cohorts_by_group': temporal_cohorts.get('cohorts_by_group', {}),
            'cohort_profiles': {
                k: {kk: vv for kk, vv in v.items() if kk != 'members'}
                for k, v in temporal_cohorts.get('cohort_profiles', {}).items()
            }
        }
    }

    with open(data_path / 'cohorts_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Saved: {data_path}/cohorts_metadata.json")

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
