"""
Signal Typology Layer
=====================

Pure orchestrator for signal typology analysis.

Produces BOTH:
    - SignalVector: Numerical measurements (DATA)
    - SignalTypology: Classification (INFORMATION)

RULE: NO computation in this file.
      If you see `np.` or `scipy.`, it belongs in an engine.

This layer:
    1. Calls engines to compute measurements
    2. Aggregates into SignalVector
    3. Classifies into SignalTypology
    4. Returns SignalTypologyOutput
"""

from datetime import datetime
from typing import Optional, Dict, List
import numpy as np

# Models
from ..models import (
    SignalVector,
    SignalTypology,
    SignalTypologyOutput,
    MemoryAxis,
    InformationAxis,
    RecurrenceAxis,
    VolatilityAxis,
    FrequencyAxis,
    DynamicsAxis,
    MomentumAxis,
    WaveletAxis,
    DerivativesAxis,
    DiscontinuityData,
)
from ..models.enums import (
    MemoryClass,
    InformationClass,
    RecurrenceClass,
    VolatilityClass,
    FrequencyClass,
    DynamicsClass,
    EnergyClass,
    ACFDecayType,
    TransitionType,
)

# Engines - Memory
from ..engines.memory.hurst_dfa import compute as compute_hurst_dfa
from ..engines.memory.acf_decay import compute as compute_acf_decay
from ..engines.memory.spectral_slope import compute as compute_spectral_slope

# Engines - Information
from ..engines.information.permutation_entropy import compute as compute_permutation_entropy
from ..engines.information.sample_entropy import compute as compute_sample_entropy
from ..engines.information.entropy_rate import compute as compute_entropy_rate

# Engines - Recurrence
from ..engines.recurrence.rqa import compute as compute_rqa

# Engines - Volatility
from ..engines.volatility.garch import compute as compute_garch
from ..engines.volatility.realized_vol import compute as compute_realized_vol
from ..engines.volatility.bipower_variation import compute as compute_bipower
from ..engines.volatility.hilbert_amplitude import compute as compute_hilbert_amplitude

# Engines - Frequency
from ..engines.frequency.spectral import compute as compute_spectral
from ..engines.frequency.wavelet import compute as compute_wavelet

# Engines - Dynamics
from ..engines.dynamics.lyapunov import compute as compute_lyapunov
from ..engines.dynamics.embedding import compute as compute_embedding

# Engines - Physics (momentum/energy)
from ..engines.physics.derivatives import compute as compute_derivatives
from ..engines.physics import kinetic_energy_dict as compute_kinetic
from ..engines.physics import potential_energy_dict as compute_potential
from ..engines.physics import hamiltonian_dict as compute_hamiltonian

# Engines - Discontinuity
from ..engines.discontinuity.structural import compute as compute_structural_discontinuity

# Archetypes
from ..archetypes import (
    match_archetype,
    compute_fingerprint,
    compute_boundary_proximity,
    diagnose_differential,
    generate_summary,
    generate_alerts,
)


class SignalTypologyLayer:
    """
    Pure orchestrator for signal typology analysis.

    Calls engines, aggregates measurements, classifies behavior.
    Contains ZERO computation - all analysis in engines/.
    """

    def __init__(
        self,
        entity_id: str = "unknown",
        signal_id: str = "unknown",
    ):
        """
        Initialize layer.

        Args:
            entity_id: Entity identifier
            signal_id: Signal identifier
        """
        self.entity_id = entity_id
        self.signal_id = signal_id
        self._previous_fingerprint: Optional[np.ndarray] = None
        self._previous_extras: Optional[Dict] = None

    def analyze(
        self,
        series: np.ndarray,
        window_start: Optional[datetime] = None,
        window_end: Optional[datetime] = None,
    ) -> SignalTypologyOutput:
        """
        Perform complete signal typology analysis.

        Args:
            series: 1D numpy array of observations
            window_start: Window start time
            window_end: Window end time

        Returns:
            SignalTypologyOutput with vector + typology
        """
        n_obs = len(series)
        now = datetime.now()
        window_start = window_start or now
        window_end = window_end or now

        # =================================================================
        # MEASURE ALL AXES (call engines)
        # =================================================================

        memory = self._measure_memory(series)
        information = self._measure_information(series)
        recurrence = self._measure_recurrence(series)
        volatility = self._measure_volatility(series)
        frequency = self._measure_frequency(series)
        dynamics = self._measure_dynamics(series)
        momentum = self._measure_momentum(series)
        wavelet = self._measure_wavelet(series)
        derivatives = self._measure_derivatives(series)
        discontinuity = self._measure_discontinuity(series)

        # =================================================================
        # BUILD SignalVector (numerical DATA)
        # =================================================================

        vector = SignalVector(
            entity_id=self.entity_id,
            signal_id=self.signal_id,
            timestamp=window_end,
            # Memory
            hurst_exponent=memory.hurst_exponent,
            hurst_confidence=memory.hurst_confidence,
            acf_decay_type=memory.acf_decay_type.value,
            acf_half_life=memory.acf_half_life,
            spectral_slope=memory.spectral_slope,
            spectral_slope_r2=memory.spectral_slope_r2,
            # Information
            entropy_permutation=information.entropy_permutation,
            entropy_sample=information.entropy_sample,
            entropy_rate=information.entropy_rate,
            # Recurrence
            rqa_determinism=recurrence.determinism,
            rqa_laminarity=recurrence.laminarity,
            rqa_entropy=recurrence.entropy,
            rqa_recurrence_rate=recurrence.recurrence_rate,
            rqa_trapping_time=recurrence.trapping_time,
            rqa_max_diagonal=recurrence.max_diagonal,
            rqa_avg_diagonal=recurrence.avg_diagonal,
            # Volatility
            garch_alpha=volatility.garch_alpha,
            garch_beta=volatility.garch_beta,
            garch_persistence=volatility.garch_persistence,
            garch_omega=volatility.garch_omega,
            garch_unconditional=volatility.garch_unconditional,
            realized_vol=volatility.realized_vol,
            bipower_variation=volatility.bipower_variation,
            jump_component=volatility.jump_component,
            hilbert_amplitude_mean=volatility.hilbert_amplitude_mean,
            hilbert_amplitude_std=volatility.hilbert_amplitude_std,
            # Frequency
            spectral_centroid=frequency.spectral_centroid,
            spectral_bandwidth=frequency.spectral_bandwidth,
            spectral_low_high_ratio=frequency.spectral_low_high_ratio,
            spectral_rolloff=frequency.spectral_rolloff,
            # Wavelet
            wavelet_dominant_scale=wavelet.dominant_scale,
            wavelet_scale_entropy=wavelet.scale_entropy,
            wavelet_energy_ratio=wavelet.energy_ratio_low_high,
            wavelet_energy_by_scale=wavelet.energy_by_scale,
            # Dynamics
            lyapunov_exponent=dynamics.lyapunov_exponent,
            lyapunov_confidence=dynamics.lyapunov_confidence,
            embedding_dimension=dynamics.embedding_dimension,
            correlation_dimension=dynamics.correlation_dimension,
            # Momentum
            d1_mean=momentum.d1_mean,
            d1_std=momentum.d1_std,
            d2_mean=momentum.d2_mean,
            d2_std=momentum.d2_std,
            momentum_strength=momentum.momentum_strength,
            kinetic_energy_mean=momentum.kinetic_energy_mean,
            potential_energy_mean=momentum.potential_energy_mean,
            hamiltonian_mean=momentum.hamiltonian_mean,
            hamiltonian_trend=momentum.hamiltonian_trend,
            energy_conserved=momentum.energy_conserved,
            # Discontinuity
            dirac_detected=discontinuity.dirac_detected,
            dirac_count=discontinuity.dirac_count,
            dirac_max_magnitude=discontinuity.dirac_max_magnitude,
            dirac_mean_half_life=discontinuity.dirac_mean_half_life,
            heaviside_detected=discontinuity.heaviside_detected,
            heaviside_count=discontinuity.heaviside_count,
            heaviside_max_magnitude=discontinuity.heaviside_max_magnitude,
            heaviside_mean_magnitude=discontinuity.heaviside_mean_magnitude,
            discontinuity_mean_interval=discontinuity.mean_interval,
            discontinuity_accelerating=discontinuity.is_accelerating,
        )

        # =================================================================
        # CLASSIFY (compute typology)
        # =================================================================

        # Compute fingerprint
        fingerprint = compute_fingerprint(
            hurst=memory.hurst_exponent,
            entropy=information.entropy_permutation,
            determinism=recurrence.determinism,
            persistence=volatility.garch_persistence,
            centroid=frequency.spectral_centroid,
            lyapunov=dynamics.lyapunov_exponent,
            hamiltonian_trend=momentum.hamiltonian_trend,
            include_energy=True,
        )

        # Match archetype
        primary, primary_score, secondary, secondary_score = match_archetype(fingerprint)
        boundary = compute_boundary_proximity(fingerprint, primary)

        # Differential diagnosis (if we have previous window)
        if self._previous_fingerprint is not None:
            current_extras = {
                'wavelet': wavelet.scale_entropy,
                'derivatives': derivatives.sign_change_rate,
            }
            diff_result = diagnose_differential(
                self._previous_fingerprint,
                fingerprint,
                self._previous_extras,
                current_extras,
            )
            transition = diff_result['transition_type']
            axes_moving = diff_result['axes_moving']
            axes_stable = diff_result['axes_stable']
            diagnosis = diff_result['diagnosis']
        else:
            transition = TransitionType.NONE
            axes_moving = []
            axes_stable = ['memory', 'information', 'recurrence',
                          'volatility', 'frequency', 'dynamics', 'energy']
            diagnosis = ""

        # Store for next window
        self._previous_fingerprint = fingerprint
        self._previous_extras = {
            'wavelet': wavelet.scale_entropy,
            'derivatives': derivatives.sign_change_rate,
        }

        # Calculate archetype distance
        from ..archetypes.library import ARCHETYPES
        archetype_obj = ARCHETYPES.get(primary)
        primary_distance = archetype_obj.distance(fingerprint) if archetype_obj else 1.0
        secondary_obj = ARCHETYPES.get(secondary)
        secondary_distance = secondary_obj.distance(fingerprint) if secondary_obj else 1.0

        # Generate summary and alerts
        typology_result = {
            'archetype': primary,
            'archetype_distance': primary_distance,
            'secondary_archetype': secondary,
            'secondary_distance': secondary_distance,
            'boundary_proximity': boundary,
            'confidence': primary_score,
        }
        diff_for_summary = {
            'transition_type': transition,
            'axes_moving': axes_moving,
            'axes_stable': axes_stable,
            'diagnosis': diagnosis,
            'alert_level': 'info',
        } if transition != TransitionType.NONE else None

        summary = generate_summary(typology_result, diff_for_summary)
        alerts = generate_alerts(typology_result, diff_for_summary, vector.to_dict())

        # =================================================================
        # BUILD SignalTypology (INFORMATION)
        # =================================================================

        typology = SignalTypology(
            entity_id=self.entity_id,
            signal_id=self.signal_id,
            window_start=window_start,
            window_end=window_end,
            n_observations=n_obs,
            # Classifications
            memory_class=memory.memory_class,
            information_class=information.information_class,
            recurrence_class=recurrence.recurrence_class,
            volatility_class=volatility.volatility_class,
            frequency_class=frequency.frequency_class,
            dynamics_class=dynamics.dynamics_class,
            energy_class=momentum.energy_class,
            # Archetype
            archetype=primary,
            archetype_distance=primary_distance,
            secondary_archetype=secondary,
            secondary_distance=secondary_distance,
            boundary_proximity=boundary,
            fingerprint=fingerprint,
            # Transition
            regime_transition=transition,
            axes_moving=axes_moving,
            axes_stable=axes_stable,
            transition_diagnosis=diagnosis,
            # Human-readable
            summary=summary,
            alerts=alerts,
            confidence=primary_score,
        )

        return SignalTypologyOutput(vector=vector, typology=typology)

    # =====================================================================
    # AXIS MEASUREMENT METHODS (call engines, no computation)
    # =====================================================================

    def _measure_memory(self, series: np.ndarray) -> MemoryAxis:
        """Measure memory axis using engines."""
        hurst = compute_hurst_dfa(series)
        acf = compute_acf_decay(series)
        spectral = compute_spectral_slope(series)

        # Classify
        h = hurst['hurst_exponent']
        if h < 0.45:
            memory_class = MemoryClass.ANTI_PERSISTENT
        elif h > 0.55:
            memory_class = MemoryClass.PERSISTENT
        else:
            memory_class = MemoryClass.RANDOM

        return MemoryAxis(
            hurst_exponent=hurst['hurst_exponent'],
            hurst_confidence=hurst['confidence'],
            hurst_method=hurst.get('method', 'dfa'),
            acf_decay_type=ACFDecayType(acf['decay_type']),
            acf_half_life=acf['half_life'],
            spectral_slope=spectral['slope'],
            spectral_slope_r2=spectral['r_squared'],
            memory_class=memory_class,
        )

    def _measure_information(self, series: np.ndarray) -> InformationAxis:
        """Measure information axis using engines."""
        perm = compute_permutation_entropy(series)
        samp = compute_sample_entropy(series)
        rate = compute_entropy_rate(series)

        # Classify
        entropy = perm['entropy']
        if entropy < 0.4:
            info_class = InformationClass.LOW
        elif entropy > 0.7:
            info_class = InformationClass.HIGH
        else:
            info_class = InformationClass.MODERATE

        return InformationAxis(
            entropy_permutation=perm['entropy'],
            entropy_sample=samp['entropy'],
            entropy_rate=rate['entropy_rate'],
            information_class=info_class,
        )

    def _measure_recurrence(self, series: np.ndarray) -> RecurrenceAxis:
        """Measure recurrence axis using engines."""
        rqa = compute_rqa(series)

        # Classify
        det = rqa['determinism']
        if det > 0.7:
            rec_class = RecurrenceClass.DETERMINISTIC
        elif det < 0.4:
            rec_class = RecurrenceClass.STOCHASTIC
        else:
            rec_class = RecurrenceClass.TRANSITIONAL

        return RecurrenceAxis(
            determinism=rqa['determinism'],
            laminarity=rqa['laminarity'],
            entropy=rqa['entropy'],
            recurrence_rate=rqa['recurrence_rate'],
            trapping_time=rqa['trapping_time'],
            max_diagonal=rqa['max_diagonal'],
            avg_diagonal=rqa['avg_diagonal'],
            recurrence_class=rec_class,
        )

    def _measure_volatility(self, series: np.ndarray) -> VolatilityAxis:
        """Measure volatility axis using engines."""
        garch = compute_garch(series)
        realized = compute_realized_vol(series)
        bipower = compute_bipower(series)
        hilbert = compute_hilbert_amplitude(series)

        # Classify
        persistence = garch['alpha'] + garch['beta']
        if persistence >= 0.99:
            vol_class = VolatilityClass.INTEGRATED
        elif persistence >= 0.85:
            vol_class = VolatilityClass.PERSISTENT
        else:
            vol_class = VolatilityClass.DISSIPATING

        return VolatilityAxis(
            garch_alpha=garch['alpha'],
            garch_beta=garch['beta'],
            garch_omega=garch['omega'],
            garch_persistence=persistence,
            garch_unconditional=garch.get('unconditional', 0.0),
            realized_vol=realized['realized_vol'],
            bipower_variation=bipower['bipower'],
            jump_component=bipower.get('jump_component', 0.0),
            jump_ratio=bipower.get('jump_ratio', 0.0),
            hilbert_amplitude_mean=hilbert['amplitude_mean'],
            hilbert_amplitude_std=hilbert['amplitude_std'],
            volatility_class=vol_class,
        )

    def _measure_frequency(self, series: np.ndarray) -> FrequencyAxis:
        """Measure frequency axis using engines."""
        spectral = compute_spectral(series)

        # Classify
        bandwidth = spectral['bandwidth']
        low_high = spectral.get('low_high_ratio', 1.0)
        if bandwidth < 0.1:
            freq_class = FrequencyClass.NARROWBAND
        elif low_high > 5.0:
            freq_class = FrequencyClass.ONE_OVER_F
        else:
            freq_class = FrequencyClass.BROADBAND

        return FrequencyAxis(
            spectral_centroid=spectral['centroid'],
            spectral_bandwidth=bandwidth,
            spectral_low_high_ratio=low_high,
            spectral_rolloff=spectral.get('rolloff', 0.0),
            frequency_class=freq_class,
        )

    def _measure_dynamics(self, series: np.ndarray) -> DynamicsAxis:
        """Measure dynamics axis using engines."""
        lyap = compute_lyapunov(series)
        embed = compute_embedding(series)

        # Classify
        le = lyap['lyapunov_exponent']
        if le < -0.05:
            dyn_class = DynamicsClass.STABLE
        elif le > 0.05:
            dyn_class = DynamicsClass.CHAOTIC
        else:
            dyn_class = DynamicsClass.EDGE_OF_CHAOS

        return DynamicsAxis(
            lyapunov_exponent=le,
            lyapunov_confidence=lyap.get('confidence', 0.0),
            embedding_dimension=embed['embedding_dimension'],
            correlation_dimension=embed.get('correlation_dim', 0.0),
            dynamics_class=dyn_class,
        )

    def _measure_momentum(self, series: np.ndarray) -> MomentumAxis:
        """Measure momentum/energy axis using engines."""
        deriv = compute_derivatives(series)
        kinetic = compute_kinetic(series)
        potential = compute_potential(series)
        hamiltonian = compute_hamiltonian(series)

        # Classify
        if hamiltonian['conserved']:
            energy_class = EnergyClass.CONSERVATIVE
        elif hamiltonian['trend'] > 0:
            energy_class = EnergyClass.DRIVEN
        else:
            energy_class = EnergyClass.DISSIPATIVE

        return MomentumAxis(
            d1_mean=deriv['d1_mean'],
            d1_std=deriv['d1_std'],
            d2_mean=deriv['d2_mean'],
            d2_std=deriv['d2_std'],
            momentum_strength=deriv['momentum_strength'],
            acceleration_regime=deriv['acceleration_regime'],
            kinetic_energy_mean=kinetic['mean'],
            potential_energy_mean=potential['mean'],
            hamiltonian_mean=hamiltonian['mean'],
            hamiltonian_trend=hamiltonian['trend'],
            energy_conserved=hamiltonian['conserved'],
            kinetic_ratio=hamiltonian.get('kinetic_ratio', 0.5),
            angular_momentum_mean=0.0,  # Computed in angular_momentum engine if needed
            is_periodic=False,
            energy_class=energy_class,
        )

    def _measure_wavelet(self, series: np.ndarray) -> WaveletAxis:
        """Measure wavelet multi-scale using engines."""
        wav = compute_wavelet(series)

        from ..models.enums import WaveletClass

        # Classify
        ratio = wav.get('energy_ratio_low_high', 1.0)
        if ratio > 2.0:
            wav_class = WaveletClass.LOW_FREQUENCY_DOMINANT
        elif ratio < 0.5:
            wav_class = WaveletClass.HIGH_FREQUENCY_DOMINANT
        else:
            wav_class = WaveletClass.BALANCED

        return WaveletAxis(
            energy_by_scale=wav.get('energy_by_scale', []),
            dominant_scale=wav.get('dominant_scale', 0),
            scale_entropy=wav.get('scale_entropy', 0.0),
            energy_ratio_low_high=ratio,
            detail_mean=wav.get('detail_mean', 0.0),
            detail_std=wav.get('detail_std', 0.0),
            detail_kurtosis=wav.get('detail_kurtosis', 0.0),
            approx_slope=wav.get('approx_slope', 0.0),
            approx_curvature=wav.get('approx_curvature', 0.0),
            scale_shift_detected=wav.get('scale_shift_detected', False),
            scale_shift_direction=wav.get('scale_shift_direction', 0),
            wavelet_class=wav_class,
        )

    def _measure_derivatives(self, series: np.ndarray) -> DerivativesAxis:
        """Measure derivatives/motion using engines."""
        deriv = compute_derivatives(series)

        from ..models.enums import DerivativesClass

        # Classify
        d1_std = deriv['d1_std']
        d3_std = deriv.get('d3_std', 0.0)
        smoothness = deriv.get('smoothness', 1.0)

        if d1_std < 0.1 and smoothness > 0.8:
            deriv_class = DerivativesClass.STATIONARY
        elif d3_std < 0.1:
            deriv_class = DerivativesClass.SMOOTH_MOTION
        else:
            deriv_class = DerivativesClass.JERKY_MOTION

        return DerivativesAxis(
            d1_mean=deriv['d1_mean'],
            d1_std=deriv['d1_std'],
            d1_max=deriv.get('d1_max', 0.0),
            d2_mean=deriv['d2_mean'],
            d2_std=deriv['d2_std'],
            d2_max=deriv.get('d2_max', 0.0),
            d3_mean=deriv.get('d3_mean', 0.0),
            d3_std=d3_std,
            momentum_strength=deriv['momentum_strength'],
            acceleration_regime=deriv['acceleration_regime'],
            smoothness=smoothness,
            d1_sign_changes=deriv.get('d1_sign_changes', 0),
            d2_sign_changes=deriv.get('d2_sign_changes', 0),
            sign_change_rate=deriv.get('sign_change_rate', 0.0),
            derivatives_class=deriv_class,
        )

    def _measure_discontinuity(self, series: np.ndarray) -> DiscontinuityData:
        """Measure structural discontinuities using engines."""
        result = compute_structural_discontinuity(series)

        dirac = result['dirac']
        heaviside = result['heaviside']

        return DiscontinuityData(
            dirac_detected=dirac['detected'],
            dirac_count=dirac['count'],
            dirac_max_magnitude=dirac.get('max_magnitude', 0.0),
            dirac_mean_magnitude=dirac.get('mean_magnitude', 0.0),
            dirac_mean_half_life=dirac.get('mean_half_life', 0.0),
            dirac_up_ratio=dirac.get('up_ratio', 0.5),
            heaviside_detected=heaviside['detected'],
            heaviside_count=heaviside['count'],
            heaviside_max_magnitude=heaviside.get('max_magnitude', 0.0),
            heaviside_mean_magnitude=heaviside.get('mean_magnitude', 0.0),
            heaviside_up_ratio=heaviside.get('up_ratio', 0.5),
            total_count=result['total_count'],
            mean_interval=result['mean_interval'],
            interval_cv=result.get('interval_cv', 0.0),
            dominant_period=result.get('dominant_period', 0.0),
            is_accelerating=result['is_accelerating'],
        )


# Convenience function for single-call usage
def analyze_signal(
    series: np.ndarray,
    entity_id: str = "unknown",
    signal_id: str = "unknown",
    window_start: Optional[datetime] = None,
    window_end: Optional[datetime] = None,
) -> SignalTypologyOutput:
    """
    Convenience function for single signal analysis.

    Args:
        series: 1D numpy array
        entity_id: Entity identifier
        signal_id: Signal identifier
        window_start: Window start time
        window_end: Window end time

    Returns:
        SignalTypologyOutput with vector + typology
    """
    layer = SignalTypologyLayer(entity_id=entity_id, signal_id=signal_id)
    return layer.analyze(series, window_start, window_end)
