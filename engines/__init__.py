"""
ENGINES — Dynamical systems computation engines.

Part of the Orthon Engines platform. Domain-agnostic signal analysis functions.
Every function takes numpy arrays and returns numbers or arrays. No file I/O,
no configuration, no orchestration — just math.

Usage:
    import engines

    # Spectral analysis
    dom_freq = engines.dominant_frequency(signal, sample_rate=1000)
    spec_ent = engines.spectral_entropy(signal)

    # Statistics
    kurt = engines.kurtosis(signal)
    skew = engines.skewness(signal)
    crest = engines.crest_factor(signal)

    # Complexity
    perm_ent = engines.permutation_entropy(signal, order=3)
    samp_ent = engines.sample_entropy(signal, m=2)

    # Memory and correlation
    hurst = engines.hurst_exponent(signal)
    acf = engines.autocorrelation(signal)

    # Geometry and eigenstructure
    cov_matrix = engines.covariance_matrix(multivariate_signal)
    eigenvals, eigenvecs = engines.eigendecomposition(cov_matrix)
    eff_dim = engines.effective_dimension(eigenvals)

    # Dynamics
    lyap = engines.lyapunov_exponent(signal)

    # Normalization
    normalized, params = engines.zscore_normalize(signal)
    robust_norm, params = engines.robust_normalize(signal)
"""

# Spectral analysis primitives
from .primitives.individual.spectral import (
    fft,
    psd,
    dominant_frequency,
    spectral_centroid,
    spectral_bandwidth,
    spectral_entropy,
    wavelet_coeffs,
)

# Temporal analysis primitives
from .primitives.individual.temporal import (
    autocorrelation,
    autocorrelation_decay,
    trend_fit,
    rate_of_change,
    turning_points,
    zero_crossings,
    mean_crossings,
    peak_detection,
    envelope_extraction,
    moving_average,
    detrend,
    segment_signal,
)

# Statistical primitives
from .primitives.individual.statistics import (
    mean,
    std,
    variance,
    min_max,
    percentiles,
    skewness,
    kurtosis,
    rms,
    peak_to_peak,
    crest_factor,
)

# Complexity and entropy primitives
from .primitives.individual.complexity import (
    permutation_entropy,
    sample_entropy,
    approximate_entropy,
    multiscale_entropy,
    lempel_ziv_complexity,
    fractal_dimension,
    entropy_rate,
)

# Memory and long-range dependence
from .primitives.individual.memory import (
    hurst_exponent,
    detrended_fluctuation_analysis,
    rescaled_range,
    long_range_correlation,
    variance_growth,
)

# Stationarity testing
from .primitives.individual.stationarity import (
    stationarity_test,
    trend,
    changepoints,
    mann_kendall_test,
)

# Geometry and linear algebra
from .primitives.individual.geometry import (
    covariance_matrix,
    correlation_matrix,
    eigendecomposition,
    effective_dimension,
    participation_ratio,
    condition_number,
    matrix_rank,
    alignment_metric,
    eigenvalue_spread,
    matrix_entropy,
    geometric_mean_eigenvalue,
    svd_decomposition,
    explained_variance_ratio,
    cumulative_variance_ratio,
)

# Similarity and distance measures
from .primitives.individual.similarity import (
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
from .primitives.individual.dynamics import (
    lyapunov_exponent,
    largest_lyapunov_exponent,
    attractor_reconstruction,
    embedding_dimension,
    optimal_delay,
    recurrence_analysis,
    poincare_map,
)

# Normalization and preprocessing
from .primitives.individual.normalization import (
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
from .primitives.individual.information import (
    transfer_entropy,
    conditional_entropy,
    joint_entropy,
    granger_causality,
    phase_coupling,
    normalized_transfer_entropy,
    information_flow,
)

# Numerical derivatives
from .primitives.individual.derivatives import (
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

# Correlation primitives
from .primitives.individual.correlation import (
    partial_autocorrelation,
)

# Fractal primitives
from .primitives.individual.fractal import (
    dfa,
    hurst_r2,
)

# Hilbert transform primitives
from .primitives.individual.hilbert import (
    hilbert_transform,
    envelope,
    instantaneous_amplitude,
    instantaneous_frequency,
    instantaneous_phase,
)

# Calculus primitives
from .primitives.individual.calculus import (
    derivative,
)

__version__ = "1.0.0"
__author__ = "Jason Rudder"
__license__ = "PolyForm-Noncommercial-1.0.0"

__all__ = [
    # Spectral
    'fft', 'psd', 'dominant_frequency', 'spectral_centroid',
    'spectral_bandwidth', 'spectral_entropy', 'wavelet_coeffs',

    # Temporal
    'autocorrelation', 'autocorrelation_decay', 'trend_fit',
    'rate_of_change', 'turning_points', 'zero_crossings', 'mean_crossings',
    'peak_detection', 'envelope_extraction', 'moving_average', 'detrend',
    'segment_signal',

    # Statistics
    'mean', 'std', 'variance', 'min_max', 'percentiles',
    'skewness', 'kurtosis', 'rms', 'peak_to_peak', 'crest_factor',

    # Complexity
    'permutation_entropy', 'sample_entropy', 'approximate_entropy',
    'multiscale_entropy', 'lempel_ziv_complexity', 'fractal_dimension',
    'entropy_rate',

    # Memory
    'hurst_exponent', 'detrended_fluctuation_analysis', 'rescaled_range',
    'long_range_correlation', 'variance_growth',

    # Stationarity
    'stationarity_test', 'trend', 'changepoints', 'mann_kendall_test',

    # Geometry
    'covariance_matrix', 'correlation_matrix', 'eigendecomposition',
    'effective_dimension', 'participation_ratio', 'condition_number',
    'matrix_rank', 'alignment_metric', 'eigenvalue_spread', 'matrix_entropy',
    'geometric_mean_eigenvalue', 'svd_decomposition', 'explained_variance_ratio',
    'cumulative_variance_ratio',

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

    # Correlation
    'partial_autocorrelation',

    # Fractal
    'dfa', 'hurst_r2',

    # Hilbert
    'hilbert_transform', 'envelope', 'instantaneous_amplitude',
    'instantaneous_frequency', 'instantaneous_phase',

    # Calculus
    'derivative',
]
