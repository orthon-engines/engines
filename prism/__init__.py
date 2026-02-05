"""
PRISM - Pure Mathematical Signal Analysis Primitives

MIT Licensed computation engine providing domain-agnostic signal analysis functions.
Every function takes numpy arrays and returns numbers or arrays. No file I/O,
no configuration, no orchestration - just math.

Usage:
    import prism

    # Spectral analysis
    psd_freqs, psd_vals = prism.power_spectral_density(signal, sample_rate=1000)
    dom_freq = prism.dominant_frequency(signal, sample_rate=1000)
    spec_flat = prism.spectral_flatness(signal)

    # Statistics
    kurt = prism.kurtosis(signal)
    skew = prism.skewness(signal)
    crest = prism.crest_factor(signal)

    # Complexity
    perm_ent = prism.permutation_entropy(signal, order=3)
    samp_ent = prism.sample_entropy(signal, m=2, r=0.2)

    # Memory and correlation
    hurst = prism.hurst_exponent(signal)
    acf = prism.autocorrelation(signal, max_lag=100)

    # Geometry and eigenstructure
    cov_matrix = prism.covariance_matrix(multivariate_signal)
    eigenvals, eigenvecs = prism.eigendecomposition(cov_matrix)
    eff_dim = prism.effective_dimension(eigenvals)

    # Dynamics
    lyap = prism.lyapunov_exponent(signal)

    # Normalization
    normalized, params = prism.zscore_normalize(signal)
    robust_norm, params = prism.robust_normalize(signal)
"""

# Spectral analysis primitives
from .spectral import (
    power_spectral_density,
    dominant_frequency,
    spectral_flatness,
    spectral_entropy,
    fundamental_frequency,
    total_harmonic_distortion,
    signal_to_noise_ratio,
    phase_coherence,
    laplace_transform,
)

# Temporal analysis primitives
from .temporal import (
    autocorrelation,
    autocorrelation_decay,
    trend_fit,
    turning_points,
    zero_crossings,
    mean_crossings,
    peak_detection,
    envelope_extraction,
)

# Statistical primitives
from .statistics import (
    mean,
    variance,
    standard_deviation,
    skewness,
    kurtosis,
    crest_factor,
    rms,
    peak_to_peak,
    coefficient_of_variation,
    median,
    iqr,
    mad,
)

# Complexity and entropy primitives
from .complexity import (
    permutation_entropy,
    sample_entropy,
    approximate_entropy,
    multiscale_entropy,
    lempel_ziv_complexity,
    fractal_dimension,
)

# Memory and long-range dependence
from .memory import (
    hurst_exponent,
    detrended_fluctuation_analysis,
    rescaled_range,
    long_range_correlation,
    variance_growth,
)

# Stationarity testing
from .stationarity import (
    kpss_test,
    augmented_dickey_fuller,
    variance_ratio_test,
    runs_test,
    is_stationary,
)

# Geometry and linear algebra
from .geometry import (
    covariance_matrix,
    correlation_matrix,
    eigendecomposition,
    effective_dimension,
    participation_ratio,
    condition_number,
    matrix_rank,
    alignment_metric,
    matrix_entropy,
)

# Similarity and distance measures
from .similarity import (
    cosine_similarity,
    euclidean_distance,
    manhattan_distance,
    correlation_coefficient,
    spearman_correlation,
    mutual_information,
    cross_correlation,
    lag_at_max_correlation,
    dynamic_time_warping,
    coherence,
    earth_movers_distance,
)

# Dynamical systems analysis
from .dynamics import (
    lyapunov_exponent,
    largest_lyapunov_exponent,
    attractor_reconstruction,
    embedding_dimension,
    optimal_delay,
    recurrence_analysis,
    poincare_map,
)

# Normalization and preprocessing
from .normalization import (
    zscore_normalize,
    robust_normalize,
    mad_normalize,
    minmax_normalize,
    quantile_normalize,
    inverse_normalize,
    normalize,
    recommend_method,
)

# Information theory
from .information import (
    transfer_entropy,
    conditional_entropy,
    joint_entropy,
    granger_causality,
    phase_coupling,
    normalized_transfer_entropy,
    information_flow,
)

# Numerical derivatives
from .derivatives import (
    first_derivative,
    second_derivative,
    gradient,
    laplacian,
    finite_difference,
    velocity,
    acceleration,
    jerk,
    curvature,
    smoothed_derivative,
    integral,
)

__version__ = "1.0.0"
__author__ = "Jason Rudder"
__license__ = "MIT"

__all__ = [
    # Spectral
    'power_spectral_density', 'dominant_frequency', 'spectral_flatness',
    'spectral_entropy', 'fundamental_frequency', 'total_harmonic_distortion',
    'signal_to_noise_ratio', 'phase_coherence', 'laplace_transform',

    # Temporal
    'autocorrelation', 'autocorrelation_decay', 'trend_fit',
    'turning_points', 'zero_crossings', 'mean_crossings',
    'peak_detection', 'envelope_extraction',

    # Statistics
    'mean', 'variance', 'standard_deviation', 'skewness', 'kurtosis',
    'crest_factor', 'rms', 'peak_to_peak', 'coefficient_of_variation',
    'median', 'iqr', 'mad',

    # Complexity
    'permutation_entropy', 'sample_entropy', 'approximate_entropy',
    'multiscale_entropy', 'lempel_ziv_complexity', 'fractal_dimension',

    # Memory
    'hurst_exponent', 'detrended_fluctuation_analysis', 'rescaled_range',
    'long_range_correlation', 'variance_growth',

    # Stationarity
    'kpss_test', 'augmented_dickey_fuller', 'variance_ratio_test',
    'runs_test', 'is_stationary',

    # Geometry
    'covariance_matrix', 'correlation_matrix', 'eigendecomposition',
    'effective_dimension', 'participation_ratio', 'condition_number',
    'matrix_rank', 'alignment_metric', 'matrix_entropy',

    # Similarity
    'cosine_similarity', 'euclidean_distance', 'manhattan_distance',
    'correlation_coefficient', 'spearman_correlation', 'mutual_information',
    'cross_correlation', 'lag_at_max_correlation', 'dynamic_time_warping',
    'coherence', 'earth_movers_distance',

    # Dynamics
    'lyapunov_exponent', 'largest_lyapunov_exponent', 'attractor_reconstruction',
    'embedding_dimension', 'optimal_delay', 'recurrence_analysis', 'poincare_map',

    # Normalization
    'zscore_normalize', 'robust_normalize', 'mad_normalize', 'minmax_normalize',
    'quantile_normalize', 'inverse_normalize', 'normalize', 'recommend_method',

    # Information
    'transfer_entropy', 'conditional_entropy', 'joint_entropy',
    'granger_causality', 'phase_coupling', 'normalized_transfer_entropy',
    'information_flow',

    # Derivatives
    'first_derivative', 'second_derivative', 'gradient', 'laplacian',
    'finite_difference', 'velocity', 'acceleration', 'jerk', 'curvature',
    'smoothed_derivative', 'integral',
]
