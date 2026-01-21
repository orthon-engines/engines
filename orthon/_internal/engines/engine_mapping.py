"""
UNIFIED ENGINE MAPPING - Single Source of Truth
================================================

This replaces the separate ENGINE_THRESHOLDS and WEIGHT_RULES
with a single, explicit mapping table.

Location: prism/engines/characterize.py
Replace the existing ENGINE_THRESHOLDS and WEIGHT_RULES (~lines 45-75)

"""

# =============================================================================
# SINGLE SOURCE OF TRUTH: ENGINE MAPPING
# =============================================================================

ENGINE_MAPPING = {
    # =========================================================================
    # ALWAYS VALID ENGINES - Run regardless of characterization
    # =========================================================================
    'hurst': {
        'always_valid': True,
        'description': 'Memory/persistence analysis',
        'metrics': [
            'hurst_exponent',
            'hurst_persistence', 
            'hurst_r_squared',
        ],
        'weight_axis': 'ax_memory',
        'weight_scale': 1.5,  # Up to 1.5x weight when ax_memory is high
    },
    
    'entropy': {
        'always_valid': True,
        'description': 'Complexity/predictability measures',
        'metrics': [
            'sample_entropy',
            'permutation_entropy',
        ],
        'weight_axis': 'ax_complexity',
        'weight_scale': 1.3,
    },
    
    'rqa': {
        'always_valid': True,
        'exclude_if': {'is_step_function': True},  # Skip for step functions (future)
        'description': 'Recurrence quantification analysis',
        'metrics': [
            'rqa_recurrence_rate',
            'rqa_determinism',
            'rqa_laminarity',
            'rqa_avg_diagonal',
            'rqa_max_diagonal',
            'rqa_avg_vertical',
            'rqa_entropy',
        ],
        'weight_axis': 'ax_determinism',
        'weight_scale': 1.3,
    },
    
    'realized_vol': {
        'always_valid': True,
        'description': 'Realized volatility and distribution statistics',
        'metrics': [
            'realized_vol',
            'realized_vol_daily',
            'up_vol',
            'down_vol',
            'vol_skew',
            'max_drawdown',
            'max_drawdown_duration',
            'avg_drawdown',
            'time_underwater',
            'skewness',
            'kurtosis',
            'mean_change',        # Renamed from mean_return (domain-neutral)
            'signal_to_noise',    # Renamed from sharpe_approx (domain-neutral)
        ],
        'weight_axis': 'ax_volatility',
        'weight_scale': 1.3,
    },

    'hilbert': {
        'always_valid': True,
        'description': 'Hilbert transform analysis',
        'metrics': [
            'hilbert_amp_mean',
            'hilbert_amp_std',
            'hilbert_amp_cv',
            'hilbert_phase_mean',
            'hilbert_phase_std',
            'hilbert_inst_freq_mean',
            'hilbert_inst_freq_std',
        ],
        'weight_axis': None,
        'weight_scale': 1.0,
    },
    
    # =========================================================================
    # CONDITIONALLY VALID ENGINES - Only run when axis thresholds met
    # =========================================================================
    'spectral': {
        # Lowered threshold from 0.3 to 0.2 for better coverage.
        # Spectral analysis can reveal structure even in weakly periodic series.
        'always_valid': False,
        'requires': {'ax_periodicity': ('>', 0.2)},
        'description': 'Power spectral density analysis',
        'metrics': [
            'spectral_dominant_freq',
            'spectral_dominant_period',
            'spectral_entropy',
            'spectral_n_peaks',
            'spectral_total_power',
            'spectral_centroid',
            'spectral_spread',
        ],
        'weight_axis': 'ax_periodicity',
        'weight_scale': 2.0,  # Strong weight boost for periodic series
    },

    'wavelet': {
        # Lowered threshold from 0.3 to 0.2 for better coverage.
        'always_valid': False,
        'requires': {'ax_periodicity': ('>', 0.2)},
        'description': 'Multi-scale wavelet decomposition',
        'metrics': [
            'wavelet_dominant_scale',
            'wavelet_scale_entropy',
            'wavelet_short_energy',
            'wavelet_mid_energy',
            'wavelet_long_energy',
        ],
        'weight_axis': 'ax_periodicity',
        'weight_scale': 1.5,
    },
    
    'hjorth': {
        'always_valid': False,
        'requires': {'ax_periodicity': ('>', 0.4)},
        'description': 'Hjorth parameters (EEG-derived)',
        'metrics': [
            'hjorth_activity',
            'hjorth_mobility',
            'hjorth_complexity',
        ],
        'weight_axis': 'ax_periodicity',
        'weight_scale': 1.3,
    },
    
    'garch': {
        # Changed from conditional to always-valid to prevent "Engine Ghosting"
        # where ACF-based volatility check disagrees with full GARCH fitting.
        # Weight scales down when ax_volatility is low, so irrelevant results
        # have minimal impact on downstream geometry.
        'always_valid': True,
        'description': 'Volatility clustering analysis',
        'metrics': [
            'garch_omega',
            'garch_alpha',
            'garch_beta',
            'garch_persistence',
            'garch_unconditional_vol',
            'garch_ll',
        ],
        'weight_axis': 'ax_volatility',
        'weight_scale': 2.0,  # Increased scale: 1x at ax_vol=0, 2x at ax_vol=1
    },
    
    'lyapunov': {
        # Lowered threshold from 0.3 to 0.2 to prevent ghosting on borderline cases.
        # Critical for dynamical systems validation (chaos detection).
        'always_valid': False,
        'requires': {'ax_complexity': ('>', 0.2)},
        'description': 'Chaos/sensitivity analysis',
        'metrics': [
            'lyapunov_exponent',
            'lyapunov_is_chaotic',
        ],
        'weight_axis': 'ax_determinism',
        'weight_scale': 1.5,
    },
    
    'changes': {
        # Renamed from 'returns' for domain neutrality
        'always_valid': False,
        'requires': {'ax_stationarity': ('<', 0.7)},  # Only for NON-stationary
        'description': 'Period-over-period change metrics (for trending series)',
        'metrics': [
            'change_mean',
            'change_std',
            'change_skew',
            'change_kurt',
            'change_snr',        # Renamed from return_sharpe (signal-to-noise)
            'change_max_decline',  # Renamed from return_max_dd
            'change_pos_pct',
            'change_autocorr',
        ],
        'weight_axis': None,
        'weight_scale': 1.0,
    },

    # =========================================================================
    # DISCONTINUITY ENGINES - Only run when has_discontinuities = True
    # =========================================================================
    'heaviside': {
        'always_valid': False,
        'requires': {'has_discontinuities': ('==', True)},
        'description': 'Persistent level shifts (steps that stay)',
        'metrics': [
            'heaviside_n_steps',
            'heaviside_mean_magnitude',
            'heaviside_max_magnitude',
            'heaviside_net_displacement',
            'heaviside_up_ratio',
            'heaviside_mean_interval',
            'heaviside_mean_sharpness',
            'heaviside_mean_persistence',
        ],
        'weight_axis': None,
        'weight_scale': 1.0,
    },

    'dirac': {
        'always_valid': False,
        'requires': {'has_discontinuities': ('==', True)},
        'description': 'Reverting impulses (shocks that decay)',
        'metrics': [
            'dirac_n_impulses',
            'dirac_mean_magnitude',
            'dirac_max_magnitude',
            'dirac_total_energy',
            'dirac_mean_decay_rate',
            'dirac_mean_half_life',
            'dirac_up_ratio',
            'dirac_mean_reversion',
            'dirac_mean_interval',
        ],
        'weight_axis': None,
        'weight_scale': 1.0,
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_valid_engines(char_result: dict) -> list:
    """
    Determine which engines are valid given characterization result.

    Args:
        char_result: Dict with 6 axes + has_discontinuities:
            - ax_stationarity, ax_memory, ax_periodicity
            - ax_complexity, ax_determinism, ax_volatility
            - has_discontinuities (bool)

    Returns:
        Sorted list of valid engine names
    """
    valid = []

    for engine_name, config in ENGINE_MAPPING.items():
        # Check always_valid
        if config.get('always_valid', False):
            # Check exclude_if conditions
            exclude_if = config.get('exclude_if', {})
            should_exclude = False

            for key, val in exclude_if.items():
                if char_result.get(key) == val:
                    should_exclude = True
                    break

            if not should_exclude:
                valid.append(engine_name)
            continue

        # Check requirements for conditional engines
        requires = config.get('requires', {})
        is_valid = True

        for field_name, (op, threshold) in requires.items():
            field_value = char_result.get(field_name, 0.5 if field_name.startswith('ax_') else False)

            if op == '==' and field_value != threshold:
                is_valid = False
                break
            elif op == '!=' and field_value == threshold:
                is_valid = False
                break
            elif op == '>' and field_value <= threshold:
                is_valid = False
                break
            elif op == '<' and field_value >= threshold:
                is_valid = False
                break
            elif op == '>=' and field_value < threshold:
                is_valid = False
                break
            elif op == '<=' and field_value > threshold:
                is_valid = False
                break

        if is_valid:
            valid.append(engine_name)

    return sorted(valid)


def get_metric_weights(axes: dict, valid_engines: list) -> dict:
    """
    Compute per-metric weights based on characterization axes.
    
    Args:
        axes: Dict with characterization axes
        valid_engines: List of valid engine names
    
    Returns:
        Dict mapping metric_name -> weight (float)
    """
    weights = {}
    
    for engine_name, config in ENGINE_MAPPING.items():
        # Skip if engine not valid
        if engine_name not in valid_engines:
            for metric in config.get('metrics', []):
                weights[metric] = 0.0
            continue
        
        weight_axis = config.get('weight_axis')
        weight_scale = config.get('weight_scale', 1.0)
        
        if weight_axis is None:
            # No dynamic weighting
            weight = 1.0
        else:
            axis_value = axes.get(weight_axis, 0.5)
            # Weight = 1.0 + axis_value * (scale - 1.0)
            # If axis is 0, weight = 1.0
            # If axis is 1, weight = scale
            weight = 1.0 + axis_value * (weight_scale - 1.0)
        
        for metric in config.get('metrics', []):
            weights[metric] = round(weight, 3)
    
    return weights


def get_all_metrics() -> list:
    """Return list of all possible metrics across all engines."""
    metrics = []
    for config in ENGINE_MAPPING.values():
        metrics.extend(config.get('metrics', []))
    return sorted(metrics)


def get_engine_for_metric(metric_name: str) -> str:
    """Return the engine that owns a given metric, or None."""
    for engine_name, config in ENGINE_MAPPING.items():
        if metric_name in config.get('metrics', []):
            return engine_name
    return None


def get_engine_info(engine_name: str) -> dict:
    """Get full configuration for an engine."""
    return ENGINE_MAPPING.get(engine_name, {})


# =============================================================================
# UPDATED _get_valid_engines and _compute_weights in Characterizer class
# =============================================================================

"""
Replace the existing methods in Characterizer class with:

    def _get_valid_engines(self, axes: Dict[str, float]) -> List[str]:
        '''Determine which engines are valid for this characterization.'''
        return get_valid_engines(axes)

    def _compute_weights(
        self,
        axes: Dict[str, float],
        valid_engines: List[str],
    ) -> Dict[str, float]:
        '''Compute per-metric weights based on axes.'''
        return get_metric_weights(axes, valid_engines)
"""


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    # Test with sample characterization result
    test_char = {
        'ax_stationarity': 0.3,  # Non-stationary → returns engine valid
        'ax_memory': 0.7,        # High memory → hurst gets boosted weight
        'ax_periodicity': 0.5,   # Moderate periodicity → spectral/wavelet valid
        'ax_complexity': 0.4,    # Moderate complexity → lyapunov valid
        'ax_determinism': 0.6,
        'ax_volatility': 0.5,    # Moderate volatility → garch valid
        'has_discontinuities': False,
    }

    print("=" * 60)
    print("ENGINE MAPPING TEST")
    print("=" * 60)
    print(f"\nTest characterization: {test_char}")

    valid = get_valid_engines(test_char)
    print(f"\nValid engines ({len(valid)}): {valid}")

    weights = get_metric_weights(test_char, valid)

    print(f"\nMetric weights (sample):")
    for metric in ['hurst_exponent', 'spectral_dominant_freq', 'garch_persistence', 'stat_mean']:
        print(f"  {metric}: {weights.get(metric, 'N/A')}")

    print(f"\nTotal metrics: {len(get_all_metrics())}")

    # Test with discontinuities
    print("\n" + "=" * 60)
    print("TEST WITH DISCONTINUITIES")
    print("=" * 60)

    test_char_disc = test_char.copy()
    test_char_disc['has_discontinuities'] = True

    valid_disc = get_valid_engines(test_char_disc)
    print(f"\nValid engines ({len(valid_disc)}): {valid_disc}")

    # Show difference
    new_engines = set(valid_disc) - set(valid)
    print(f"\nAdditional engines from discontinuities: {sorted(new_engines)}")
