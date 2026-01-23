"""
Ørthon Signal Typology: Data Models
===================================

Signal Typology classifies time series through measurement across six orthogonal 
axes combined with structural discontinuity detection, producing a multi-dimensional 
fingerprint that characterizes signal behavior and identifies regime transitions.

The Six Orthogonal Axes:
    1. Memory        - Long-range dependence, persistence
    2. Information   - Complexity, randomness, predictability  
    3. Recurrence    - Pattern repetition, deterministic structure
    4. Volatility    - Amplitude dynamics, variance persistence
    5. Frequency     - Spectral character, dominant timescales
    6. Dynamics      - Stability, chaos, attractor structure

Plus: Structural Discontinuity Detection (Dirac impulse + Heaviside step)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Tuple
import numpy as np


# =============================================================================
# ENUMS: Classification Categories
# =============================================================================

class MemoryClass(Enum):
    """Axis 1: Memory structure classification"""
    ANTI_PERSISTENT = "anti_persistent"   # H < 0.45: mean-reverting
    RANDOM = "random"                      # 0.45 <= H <= 0.55: no memory
    PERSISTENT = "persistent"              # H > 0.55: trending


class InformationClass(Enum):
    """Axis 2: Information/entropy classification"""
    LOW = "low"           # Highly structured, predictable
    MODERATE = "moderate" # Mixed structure
    HIGH = "high"         # High randomness, complex


class RecurrenceClass(Enum):
    """Axis 3: Recurrence structure classification"""
    DETERMINISTIC = "deterministic"   # DET > 0.7: strong pattern repetition
    TRANSITIONAL = "transitional"     # 0.4 <= DET <= 0.7: mixed
    STOCHASTIC = "stochastic"         # DET < 0.4: weak/no patterns


class VolatilityClass(Enum):
    """Axis 4: Volatility persistence classification"""
    DISSIPATING = "dissipating"   # α+β < 0.85: shocks fade quickly
    PERSISTENT = "persistent"      # 0.85 <= α+β < 0.99: shocks linger
    INTEGRATED = "integrated"      # α+β >= 0.99: shocks permanent


class FrequencyClass(Enum):
    """Axis 5: Spectral character classification"""
    NARROWBAND = "narrowband"     # Dominant frequency, low bandwidth
    BROADBAND = "broadband"       # No dominant frequency, high bandwidth
    ONE_OVER_F = "one_over_f"     # 1/f noise, power-law spectrum


class DynamicsClass(Enum):
    """Axis 6: Dynamical stability classification"""
    STABLE = "stable"             # λ < -0.05: converging to attractor
    EDGE_OF_CHAOS = "edge_of_chaos"  # -0.05 <= λ <= 0.05: critical
    CHAOTIC = "chaotic"           # λ > 0.05: sensitive dependence


class ACFDecayType(Enum):
    """Autocorrelation decay classification"""
    EXPONENTIAL = "exponential"   # Short-range dependence
    POWER_LAW = "power_law"       # Long-range dependence


class TransitionType(Enum):
    """Regime transition status"""
    NONE = "none"                 # Stable in current regime
    APPROACHING = "approaching"   # Moving toward boundary
    IN_PROGRESS = "in_progress"   # Currently crossing boundary
    COMPLETED = "completed"       # Recently changed regime


# =============================================================================
# AXIS MEASUREMENT DATACLASSES
# =============================================================================

@dataclass
class MemoryAxis:
    """Axis 1: Memory structure measurements"""
    hurst_exponent: float = 0.5              # 0-1
    hurst_method: str = "dfa"                # 'dfa' | 'rs' | 'wavelet'
    hurst_confidence: float = 0.0            # R² of power-law fit
    acf_decay_type: ACFDecayType = ACFDecayType.EXPONENTIAL
    acf_half_life: float = 0.0               # Lags to 50% decay
    spectral_slope: float = 0.0              # β in S(f) ~ f^-β
    spectral_slope_r2: float = 0.0           # Fit quality
    memory_class: MemoryClass = MemoryClass.RANDOM
    
    def to_vector(self) -> np.ndarray:
        """Normalize to [0,1] for fingerprint"""
        return np.array([
            self.hurst_exponent,  # Already 0-1
            1.0 if self.acf_decay_type == ACFDecayType.POWER_LAW else 0.0,
            np.clip((self.spectral_slope + 2) / 4, 0, 1)  # Normalize -2 to 2 → 0 to 1
        ])


@dataclass
class InformationAxis:
    """Axis 2: Information/entropy measurements"""
    entropy_permutation: float = 0.0         # 0-1 normalized
    entropy_sample: float = 0.0              # Sample entropy
    entropy_rate: float = 0.0                # Δentropy/Δt (change rate)
    information_class: InformationClass = InformationClass.MODERATE
    
    def to_vector(self) -> np.ndarray:
        """Normalize to [0,1] for fingerprint"""
        return np.array([
            np.clip(self.entropy_permutation, 0, 1),
            np.clip(self.entropy_sample / 3.0, 0, 1),  # Typical range 0-3
            np.clip((self.entropy_rate + 0.5) / 1.0, 0, 1)  # Center around 0
        ])


@dataclass  
class RecurrenceAxis:
    """Axis 3: Recurrence Quantification Analysis measurements"""
    determinism: float = 0.0                 # 0-1: % diagonal structures
    laminarity: float = 0.0                  # 0-1: % vertical structures
    entropy_diagonal: float = 0.0            # Diagonal line length entropy
    recurrence_rate: float = 0.0             # % of recurrent points
    trapping_time: float = 0.0               # Avg vertical line length
    max_diagonal: int = 0                    # Longest diagonal (predictability horizon)
    avg_diagonal: float = 0.0                # Average diagonal length
    recurrence_class: RecurrenceClass = RecurrenceClass.TRANSITIONAL
    
    def to_vector(self) -> np.ndarray:
        """Normalize to [0,1] for fingerprint"""
        return np.array([
            self.determinism,
            self.laminarity,
            np.clip(self.entropy_diagonal / 5.0, 0, 1),  # Typical range 0-5
            self.recurrence_rate
        ])


@dataclass
class VolatilityAxis:
    """Axis 4: Volatility/amplitude dynamics measurements"""
    garch_alpha: float = 0.0                 # ARCH effect
    garch_beta: float = 0.0                  # GARCH effect
    garch_persistence: float = 0.0           # α + β
    garch_omega: float = 0.0                 # Constant term
    garch_unconditional: float = 0.0         # Long-run variance
    hilbert_amplitude_mean: float = 0.0      # Mean envelope
    hilbert_amplitude_std: float = 0.0       # Envelope variability
    volatility_class: VolatilityClass = VolatilityClass.PERSISTENT
    
    def to_vector(self) -> np.ndarray:
        """Normalize to [0,1] for fingerprint"""
        return np.array([
            np.clip(self.garch_persistence, 0, 1),
            np.clip(self.garch_alpha / 0.3, 0, 1),  # High α rare
            np.clip(self.hilbert_amplitude_std / self.hilbert_amplitude_mean 
                    if self.hilbert_amplitude_mean > 0 else 0, 0, 2) / 2
        ])


@dataclass
class FrequencyAxis:
    """Axis 5: Spectral characteristics measurements"""
    spectral_centroid: float = 0.0           # Dominant frequency
    spectral_bandwidth: float = 0.0          # Frequency spread
    spectral_low_high_ratio: float = 0.0     # Energy distribution
    spectral_rolloff: float = 0.0            # 85% energy frequency
    frequency_class: FrequencyClass = FrequencyClass.BROADBAND
    
    def to_vector(self) -> np.ndarray:
        """Normalize to [0,1] for fingerprint"""
        # Centroid normalized by Nyquist (0.5)
        return np.array([
            np.clip(self.spectral_centroid / 0.5, 0, 1),
            np.clip(self.spectral_bandwidth / 0.25, 0, 1),
            np.clip(self.spectral_low_high_ratio / 10, 0, 1)
        ])


@dataclass
class DynamicsAxis:
    """Axis 6: Dynamical systems measurements"""
    lyapunov_exponent: float = 0.0           # Can be negative (stable)
    lyapunov_confidence: float = 0.0         # Estimation quality
    embedding_dimension: int = 1             # Complexity of attractor
    correlation_dimension: float = 0.0       # Fractal dimension
    dynamics_class: DynamicsClass = DynamicsClass.STABLE
    
    def to_vector(self) -> np.ndarray:
        """Normalize to [0,1] for fingerprint"""
        return np.array([
            np.clip((self.lyapunov_exponent + 0.5) / 1.0, 0, 1),  # -0.5 to 0.5 → 0 to 1
            np.clip(self.embedding_dimension / 10, 0, 1),
            np.clip(self.correlation_dimension / 5, 0, 1)
        ])


# =============================================================================
# STRUCTURAL DISCONTINUITY
# =============================================================================

@dataclass
class DiracDiscontinuity:
    """Impulse (transient) discontinuity detection - Dirac δ"""
    detected: bool = False
    count: int = 0                           # Number of impulses
    max_magnitude: float = 0.0               # Largest impulse (σ units)
    mean_magnitude: float = 0.0              # Average impulse size
    mean_half_life: float = 0.0              # Average decay rate (periods)
    up_ratio: float = 0.5                    # Fraction of positive impulses
    locations: List[int] = field(default_factory=list)  # Indices


@dataclass
class HeavisideDiscontinuity:
    """Step (permanent) discontinuity detection - Heaviside H"""
    detected: bool = False
    count: int = 0                           # Number of steps
    max_magnitude: float = 0.0               # Largest step (σ units)
    mean_magnitude: float = 0.0              # Average step size
    up_ratio: float = 0.5                    # Fraction of positive steps
    locations: List[int] = field(default_factory=list)  # Indices


@dataclass
class StructuralDiscontinuity:
    """Combined structural discontinuity analysis"""
    dirac: DiracDiscontinuity = field(default_factory=DiracDiscontinuity)
    heaviside: HeavisideDiscontinuity = field(default_factory=HeavisideDiscontinuity)
    
    # Structural metrics
    mean_interval: float = 0.0               # Avg time between any discontinuity
    interval_cv: float = 0.0                 # Coefficient of variation (regularity)
    dominant_period: float = 0.0             # Characteristic frequency
    is_accelerating: bool = False            # Are breaks getting more frequent?
    
    @property
    def any_detected(self) -> bool:
        return self.dirac.detected or self.heaviside.detected
    
    @property
    def total_count(self) -> int:
        return self.dirac.count + self.heaviside.count


# =============================================================================
# ARCHETYPE DEFINITIONS
# =============================================================================

@dataclass
class Archetype:
    """Behavioral archetype definition"""
    name: str = "Unknown"
    description: str = ""
    
    # Expected ranges for each axis (min, max)
    memory_range: Tuple[float, float] = (0.0, 1.0)
    information_range: Tuple[float, float] = (0.0, 1.0)
    recurrence_range: Tuple[float, float] = (0.0, 1.0)
    volatility_range: Tuple[float, float] = (0.0, 1.0)
    frequency_range: Tuple[float, float] = (0.0, 1.0)
    dynamics_range: Tuple[float, float] = (0.0, 1.0)
    
    # Expected discontinuity patterns
    expects_dirac: bool = False
    expects_heaviside: bool = False
    
    def centroid(self) -> np.ndarray:
        """Return center point of archetype region"""
        return np.array([
            (self.memory_range[0] + self.memory_range[1]) / 2,
            (self.information_range[0] + self.information_range[1]) / 2,
            (self.recurrence_range[0] + self.recurrence_range[1]) / 2,
            (self.volatility_range[0] + self.volatility_range[1]) / 2,
            (self.frequency_range[0] + self.frequency_range[1]) / 2,
            (self.dynamics_range[0] + self.dynamics_range[1]) / 2,
        ])
    
    def contains(self, fingerprint: np.ndarray) -> bool:
        """Check if fingerprint falls within archetype bounds"""
        ranges = [
            self.memory_range, self.information_range, self.recurrence_range,
            self.volatility_range, self.frequency_range, self.dynamics_range
        ]
        for i, (low, high) in enumerate(ranges):
            if fingerprint[i] < low or fingerprint[i] > high:
                return False
        return True
    
    def distance_to(self, fingerprint: np.ndarray) -> float:
        """Euclidean distance from fingerprint to archetype centroid"""
        return np.linalg.norm(fingerprint - self.centroid())


# =============================================================================
# PRIMARY OUTPUT: SIGNAL TYPOLOGY
# =============================================================================

@dataclass
class SignalTypology:
    """
    Complete signal typology output.
    
    This is the primary output of the Signal Typology layer, containing:
    - All six axis measurements
    - Structural discontinuity detection
    - Archetype classification
    - Regime transition status
    - Human-readable summary
    """
    
    # === IDENTIFICATION ===
    entity_id: str = "unknown"
    signal_id: str = "unknown"
    window_start: datetime = field(default_factory=datetime.now)
    window_end: datetime = field(default_factory=datetime.now)
    n_observations: int = 0
    
    # === THE SIX ORTHOGONAL AXES ===
    memory: MemoryAxis = field(default_factory=MemoryAxis)
    information: InformationAxis = field(default_factory=InformationAxis)
    recurrence: RecurrenceAxis = field(default_factory=RecurrenceAxis)
    volatility: VolatilityAxis = field(default_factory=VolatilityAxis)
    frequency: FrequencyAxis = field(default_factory=FrequencyAxis)
    dynamics: DynamicsAxis = field(default_factory=DynamicsAxis)
    
    # === STRUCTURAL DISCONTINUITY ===
    discontinuity: StructuralDiscontinuity = field(default_factory=StructuralDiscontinuity)
    
    # === COMPOSITE CLASSIFICATION ===
    archetype: str = "Unknown"                    # Primary archetype name
    archetype_distance: float = 0.0               # Distance to nearest archetype
    secondary_archetype: str = "Unknown"          # Second nearest
    secondary_distance: float = 0.0               # Distance to second
    boundary_proximity: float = 1.0               # 0 = at boundary, 1 = far from boundary
    
    # === FINGERPRINT ===
    fingerprint: np.ndarray = field(default_factory=lambda: np.zeros(6))
    
    # === CHANGE DETECTION ===
    regime_transition: TransitionType = TransitionType.NONE
    axes_moving: List[str] = field(default_factory=list)
    axes_stable: List[str] = field(default_factory=list)
    
    # === HUMAN-READABLE ===
    summary: str = ""
    confidence: float = 0.0
    alerts: List[str] = field(default_factory=list)
    
    def compute_fingerprint(self) -> np.ndarray:
        """Compute 6D fingerprint from axis measurements"""
        # Use primary metric from each axis for the core fingerprint
        self.fingerprint = np.array([
            self.memory.hurst_exponent,
            self.information.entropy_permutation,
            self.recurrence.determinism,
            self.volatility.garch_persistence,
            np.clip(self.frequency.spectral_bandwidth / 0.25, 0, 1),
            np.clip((self.dynamics.lyapunov_exponent + 0.5) / 1.0, 0, 1)
        ])
        return self.fingerprint
    
    def to_dict(self) -> Dict:
        """Export to dictionary for serialization"""
        return {
            'entity_id': self.entity_id,
            'signal_id': self.signal_id,
            'window_start': self.window_start.isoformat(),
            'window_end': self.window_end.isoformat(),
            'n_observations': self.n_observations,
            
            # Axis classifications
            'memory_class': self.memory.memory_class.value,
            'information_class': self.information.information_class.value,
            'recurrence_class': self.recurrence.recurrence_class.value,
            'volatility_class': self.volatility.volatility_class.value,
            'frequency_class': self.frequency.frequency_class.value,
            'dynamics_class': self.dynamics.dynamics_class.value,
            
            # Key metrics
            'hurst_exponent': self.memory.hurst_exponent,
            'entropy_permutation': self.information.entropy_permutation,
            'determinism': self.recurrence.determinism,
            'garch_persistence': self.volatility.garch_persistence,
            'spectral_centroid': self.frequency.spectral_centroid,
            'lyapunov_exponent': self.dynamics.lyapunov_exponent,
            
            # Discontinuity
            'dirac_detected': self.discontinuity.dirac.detected,
            'dirac_count': self.discontinuity.dirac.count,
            'heaviside_detected': self.discontinuity.heaviside.detected,
            'heaviside_count': self.discontinuity.heaviside.count,
            
            # Classification
            'archetype': self.archetype,
            'archetype_distance': self.archetype_distance,
            'secondary_archetype': self.secondary_archetype,
            'boundary_proximity': self.boundary_proximity,
            'fingerprint': self.fingerprint.tolist(),
            
            # Transition
            'regime_transition': self.regime_transition.value,
            'axes_moving': self.axes_moving,
            'axes_stable': self.axes_stable,
            
            # Summary
            'summary': self.summary,
            'confidence': self.confidence,
            'alerts': self.alerts
        }
