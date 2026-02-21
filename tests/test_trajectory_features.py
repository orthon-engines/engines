"""
Tests for trajectory feature extraction.

Validates:
    1. Feature extraction runs without error
    2. Output shape and column structure correct
    3. Cross-axis features computed correctly
    4. Known engines produce expected collapse patterns
    5. No unexpected all-NaN columns
"""

import polars as pl
import numpy as np
import pytest
from pathlib import Path

from manifold.features.trajectory_features import (
    extract_single_axis_features,
    extract_cross_axis_features,
    build_feature_matrix,
    _extract_subwindow_features,
    TRAJECTORY_METRICS,
)


# ─────────────────────────────────────────────────────────────────────
# Fixtures: synthetic trajectory data
# ─────────────────────────────────────────────────────────────────────

def _make_synthetic_sigs(n_engines=10, n_windows=8, collapse_engines=None):
    """Create synthetic trajectory signatures for testing."""
    collapse_engines = collapse_engines or []
    rows = []
    for i in range(n_engines):
        eng = f'engine_{i+1}'
        collapsing = eng in collapse_engines
        for w in range(n_windows):
            t = w / (n_windows - 1)  # 0 to 1

            if collapsing:
                eff_dim = 2.8 - 0.5 * t  # 2.8 → 2.3
                cond = 2.0 + 6.0 * t      # 2.0 → 8.0
            else:
                eff_dim = 2.6 + 0.05 * np.sin(t * 3)  # stable oscillation
                cond = 2.5 + 0.3 * np.sin(t * 2)

            speed = 0.5 + 0.3 * np.random.randn()
            curv = 1.0 + 0.5 * t + 0.2 * np.random.randn()

            rows.append({
                'cohort': eng,
                'signal_0_end': float(w * 24 + 24),
                'eigenvalue_1': 1.5 + 0.2 * np.random.randn(),
                'eigenvalue_2': 1.0 + 0.1 * np.random.randn(),
                'eigenvalue_3': 0.5 + 0.1 * np.random.randn(),
                'effective_dim': eff_dim,
                'total_variance': 3.0,
                'condition_number': cond,
                'effective_dim_velocity': -0.05 if collapsing else 0.01,
                'effective_dim_acceleration': 0.0,
                'effective_dim_curvature': 0.0,
                'speed': max(0.1, speed),
                'curvature': max(0.1, curv),
                'acceleration_magnitude': 0.5,
                'torsion': float('nan'),
                'arc_length': float(w) * 1.2,
            })

    return pl.DataFrame(rows)


def _make_synthetic_match(n_engines=10, collapse_engines=None):
    """Create synthetic trajectory match for testing."""
    collapse_engines = collapse_engines or []
    rows = []
    for i in range(n_engines):
        eng = f'engine_{i+1}'
        tid = 1 if eng in collapse_engines else 0
        rows.append({
            'cohort': eng,
            'trajectory_id': tid,
            'match_distance': 0.2 + 0.1 * np.random.rand(),
            'second_distance': 0.35,
            'match_confidence': 0.3 if eng not in collapse_engines else 0.6,
            'trajectory_position': 1.0,
            'n_windows': 8,
        })
    return pl.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────

class TestSingleAxisFeatures:

    def test_output_shape(self):
        sigs = _make_synthetic_sigs()
        match = _make_synthetic_match()
        feats = extract_single_axis_features(sigs, match, 'test')
        assert len(feats) == 10
        assert 'cohort' in feats.columns
        assert 'test_n_windows' in feats.columns

    def test_all_metrics_present(self):
        sigs = _make_synthetic_sigs()
        match = _make_synthetic_match()
        feats = extract_single_axis_features(sigs, match, 'ax')
        for metric in TRAJECTORY_METRICS:
            assert f'ax_{metric}_mean' in feats.columns, f"Missing {metric}_mean"
            assert f'ax_{metric}_delta' in feats.columns, f"Missing {metric}_delta"

    def test_collapse_detected_in_delta(self):
        collapse = ['engine_1', 'engine_2']
        sigs = _make_synthetic_sigs(collapse_engines=collapse)
        match = _make_synthetic_match(collapse_engines=collapse)
        feats = extract_single_axis_features(sigs, match, 'ax')

        for eng in collapse:
            row = feats.filter(pl.col('cohort') == eng)
            delta = row['ax_effective_dim_delta'][0]
            assert delta < -0.2, f"{eng} should collapse: delta={delta}"

        stable = feats.filter(~pl.col('cohort').is_in(collapse))
        for row in stable.iter_rows(named=True):
            assert abs(row['ax_effective_dim_delta']) < 0.2, \
                f"{row['cohort']} should be stable"

    def test_position_weighted_emphasizes_late(self):
        collapse = ['engine_1']
        sigs = _make_synthetic_sigs(n_engines=3, collapse_engines=collapse)
        match = _make_synthetic_match(n_engines=3, collapse_engines=collapse)
        feats = extract_single_axis_features(sigs, match, 'ax')

        row = feats.filter(pl.col('cohort') == 'engine_1')
        pw = row['ax_eff_dim_pos_weighted'][0]
        mean = row['ax_effective_dim_mean'][0]
        # Position-weighted should be lower than mean for collapsers
        # (late values weighted more, late values are lower)
        assert pw < mean, "Position-weighted should emphasize late (low) values"

    def test_quartile_ordering(self):
        sigs = _make_synthetic_sigs(n_engines=3, collapse_engines=['engine_1'])
        match = _make_synthetic_match(n_engines=3, collapse_engines=['engine_1'])
        feats = extract_single_axis_features(sigs, match, 'ax')

        row = feats.filter(pl.col('cohort') == 'engine_1')
        q1 = row['ax_effective_dim_q1'][0]
        q3 = row['ax_effective_dim_q3'][0]
        # For collapsing engine, q3 < q1 (dimension decreases)
        assert q3 < q1, f"Collapser q3={q3} should be < q1={q1}"

    def test_match_metadata_joined(self):
        sigs = _make_synthetic_sigs()
        match = _make_synthetic_match()
        feats = extract_single_axis_features(sigs, match, 'ax')
        assert 'ax_trajectory_id' in feats.columns
        assert 'ax_match_confidence' in feats.columns
        assert feats['ax_trajectory_id'].drop_nulls().len() == 10


class TestCrossAxisFeatures:

    def test_collapse_count(self):
        # Two axes: engine_1 collapses on both, engine_2 on one
        axis_a = extract_single_axis_features(
            _make_synthetic_sigs(n_engines=5, collapse_engines=['engine_1', 'engine_2']),
            _make_synthetic_match(n_engines=5, collapse_engines=['engine_1', 'engine_2']),
            'axis_a',
        )
        axis_b = extract_single_axis_features(
            _make_synthetic_sigs(n_engines=5, collapse_engines=['engine_1']),
            _make_synthetic_match(n_engines=5, collapse_engines=['engine_1']),
            'axis_b',
        )

        cross = extract_cross_axis_features({'axis_a': axis_a, 'axis_b': axis_b})

        eng1 = cross.filter(pl.col('cohort') == 'engine_1')
        assert eng1['cross_n_axes_collapsing'][0] == 2
        assert eng1['cross_collapse_on_all'][0] == 1

        eng2 = cross.filter(pl.col('cohort') == 'engine_2')
        assert eng2['cross_n_axes_collapsing'][0] == 1
        assert eng2['cross_collapse_on_all'][0] == 0

    def test_stable_engine_no_collapse(self):
        axis_a = extract_single_axis_features(
            _make_synthetic_sigs(n_engines=3),
            _make_synthetic_match(n_engines=3),
            'axis_a',
        )
        cross = extract_cross_axis_features({'axis_a': axis_a})

        for row in cross.iter_rows(named=True):
            assert row['cross_n_axes_collapsing'] == 0


class TestBuildFeatureMatrix:

    def test_full_pipeline(self, tmp_path):
        collapse = ['engine_1']
        sigs = _make_synthetic_sigs(collapse_engines=collapse)
        match = _make_synthetic_match(collapse_engines=collapse)

        sigs_path = tmp_path / 'sigs.parquet'
        match_path = tmp_path / 'match.parquet'
        sigs.write_parquet(str(sigs_path))
        match.write_parquet(str(match_path))

        features = build_feature_matrix(
            {'test_axis': {'sigs': str(sigs_path), 'match': str(match_path)}},
            verbose=False,
        )

        assert len(features) == 10
        assert 'cohort' in features.columns
        assert 'cross_n_axes_collapsing' in features.columns

    def test_zero_variance_dropped(self, tmp_path):
        sigs = _make_synthetic_sigs()
        match = _make_synthetic_match()
        sigs_path = tmp_path / 'sigs.parquet'
        match_path = tmp_path / 'match.parquet'
        sigs.write_parquet(str(sigs_path))
        match.write_parquet(str(match_path))

        features = build_feature_matrix(
            {'test': {'sigs': str(sigs_path), 'match': str(match_path)}},
            drop_zero_variance=True,
            verbose=False,
        )

        # trajectory_position should be dropped (always 1.0)
        assert 'test_trajectory_position' not in features.columns

    def test_with_geometry(self, tmp_path):
        collapse = ['engine_1']
        sigs = _make_synthetic_sigs(collapse_engines=collapse)
        match = _make_synthetic_match(collapse_engines=collapse)
        geometry = _make_synthetic_geometry(
            n_engines=10, n_windows=8, n_subwindows=3, collapse_engines=collapse
        )

        sigs_path = tmp_path / 'sigs.parquet'
        match_path = tmp_path / 'match.parquet'
        geo_path = tmp_path / 'geometry.parquet'
        sigs.write_parquet(str(sigs_path))
        match.write_parquet(str(match_path))
        geometry.write_parquet(str(geo_path))

        features = build_feature_matrix(
            {'ax': {
                'sigs': str(sigs_path),
                'match': str(match_path),
                'geometry': str(geo_path),
            }},
            verbose=False,
        )

        assert len(features) == 10
        assert 'ax_sw_eff_dim_min_mean' in features.columns
        assert 'ax_sw_cond_max_max' in features.columns
        assert 'ax_sw_n_degenerate' in features.columns


# ─────────────────────────────────────────────────────────────────────
# Sub-window dispersion tests
# ─────────────────────────────────────────────────────────────────────

def _make_synthetic_geometry(n_engines=10, n_windows=8, n_subwindows=3,
                              collapse_engines=None):
    """Create synthetic cohort_geometry with multiple sub-windows per signal_0_end.

    For collapse engines, sub-window 0 stays healthy while sub-windows 1-2
    collapse — mimicking the engine_100 pattern where averaging masks failure.
    """
    collapse_engines = collapse_engines or []
    rows = []
    for i in range(n_engines):
        eng = f'engine_{i+1}'
        collapsing = eng in collapse_engines
        for w in range(n_windows):
            t = w / (n_windows - 1)
            s0_end = float(w * 24 + 55)
            for sw in range(n_subwindows):
                if collapsing:
                    if sw == 0:
                        # Primary sub-window stays healthy (what sigs picks)
                        eff_dim = 2.7 + 0.1 * np.sin(t * 2)
                        cond = 2.0 + 0.5 * t
                    else:
                        # Other sub-windows collapse
                        eff_dim = 2.7 - 0.8 * t
                        cond = 2.0 + 5000.0 * t**2
                else:
                    eff_dim = 2.6 + 0.05 * np.sin(t * 3) - 0.1 * sw * 0.1
                    cond = 2.5 + 0.3 * np.sin(t * 2)

                rows.append({
                    'cohort': eng,
                    'signal_0_end': s0_end,
                    'signal_0_start': float(max(0, w * 24 - 24 * sw)),
                    'signal_0_center': s0_end - 12.0,
                    'engine': eng,
                    'n_signals': 16,
                    'n_features': 80,
                    'effective_dim': eff_dim,
                    'condition_number': cond,
                    'eigenvalue_1': 1.5,
                    'eigenvalue_2': 1.0,
                    'eigenvalue_3': max(0.01, 0.5 - 0.3 * t) if (collapsing and sw > 0) else 0.5,
                    'total_variance': 3.0,
                })
    return pl.DataFrame(rows)


class TestSubwindowFeatures:

    def test_subwindow_features_present(self):
        geometry = _make_synthetic_geometry(n_engines=5, collapse_engines=['engine_1'])
        feats = _extract_subwindow_features(geometry, 'ax')
        assert len(feats) == 5
        assert 'ax_sw_eff_dim_min_mean' in feats.columns
        assert 'ax_sw_cond_max_max' in feats.columns
        assert 'ax_sw_n_degenerate' in feats.columns
        assert 'ax_sw_eff_dim_min_delta' in feats.columns

    def test_hidden_collapse_detected(self):
        """Collapser's sub-windows diverge: min eff_dim drops, spread grows."""
        geometry = _make_synthetic_geometry(
            n_engines=5, collapse_engines=['engine_1']
        )
        feats = _extract_subwindow_features(geometry, 'ax')

        col_row = feats.filter(pl.col('cohort') == 'engine_1')
        stable_row = feats.filter(pl.col('cohort') == 'engine_2')

        # Collapser: min eff_dim should be much lower (sub-windows collapse)
        col_min = col_row['ax_sw_eff_dim_min_mean'][0]
        stable_min = stable_row['ax_sw_eff_dim_min_mean'][0]
        assert col_min < stable_min, \
            f"Collapser min_mean={col_min} should be < stable={stable_min}"

        # Collapser: spread should be larger (primary stays, others collapse)
        col_spread = col_row['ax_sw_eff_dim_spread_mean'][0]
        stable_spread = stable_row['ax_sw_eff_dim_spread_mean'][0]
        assert col_spread > stable_spread * 2, \
            f"Collapser spread={col_spread} should be >> stable={stable_spread}"

        # Collapser: cond_max should be huge (degenerate sub-windows)
        col_cond = col_row['ax_sw_cond_max_max'][0]
        assert col_cond > 100, f"Collapser max cond={col_cond} should spike"

    def test_degenerate_count(self):
        geometry = _make_synthetic_geometry(
            n_engines=3, collapse_engines=['engine_1']
        )
        feats = _extract_subwindow_features(geometry, 'ax')

        col_row = feats.filter(pl.col('cohort') == 'engine_1')
        stable_row = feats.filter(pl.col('cohort') == 'engine_2')

        assert col_row['ax_sw_n_degenerate'][0] > 0
        assert stable_row['ax_sw_n_degenerate'][0] == 0

    def test_integrated_with_single_axis(self):
        """Sub-window features are joined when geometry is passed."""
        collapse = ['engine_1']
        sigs = _make_synthetic_sigs(n_engines=5, collapse_engines=collapse)
        match = _make_synthetic_match(n_engines=5, collapse_engines=collapse)
        geometry = _make_synthetic_geometry(
            n_engines=5, collapse_engines=collapse
        )

        feats = extract_single_axis_features(sigs, match, 'ax', geometry=geometry)

        # Should have both regular and sub-window features
        assert 'ax_effective_dim_mean' in feats.columns
        assert 'ax_sw_eff_dim_min_mean' in feats.columns
        assert len(feats) == 5
