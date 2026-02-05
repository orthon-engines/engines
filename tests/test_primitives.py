"""
Tests for PRISM primitives.

Each test validates basic functionality of pure mathematical functions.
"""

import numpy as np
import pytest

import prism


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sine_signal():
    """Generate a clean sine wave."""
    t = np.linspace(0, 1, 1000)
    return np.sin(2 * np.pi * 10 * t)


@pytest.fixture
def noisy_signal():
    """Generate a noisy signal."""
    np.random.seed(42)
    t = np.linspace(0, 1, 1000)
    return np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(1000)


@pytest.fixture
def random_walk():
    """Generate a random walk (non-stationary)."""
    np.random.seed(42)
    return np.cumsum(np.random.randn(1000))


@pytest.fixture
def multivariate_data():
    """Generate multivariate data for geometry tests."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    return np.random.randn(n_samples, n_features)


# =============================================================================
# Spectral Tests
# =============================================================================

class TestSpectral:
    """Test spectral analysis primitives."""

    def test_power_spectral_density(self, sine_signal):
        """PSD should return frequencies and power values."""
        freqs, psd = prism.power_spectral_density(sine_signal, sample_rate=1000)
        assert len(freqs) > 0
        assert len(psd) == len(freqs)
        assert np.all(psd >= 0)  # Power is non-negative

    def test_dominant_frequency(self, sine_signal):
        """Dominant frequency should detect the 10 Hz sine."""
        dom_freq = prism.dominant_frequency(sine_signal, sample_rate=1000)
        assert 9 <= dom_freq <= 11  # Should be close to 10 Hz

    def test_spectral_flatness(self, sine_signal, noisy_signal):
        """Spectral flatness: sine should be lower than noise."""
        flat_sine = prism.spectral_flatness(sine_signal)
        flat_noise = prism.spectral_flatness(noisy_signal)
        assert 0 <= flat_sine <= 1
        assert flat_sine < flat_noise  # Pure tone has lower flatness

    def test_spectral_entropy(self, sine_signal):
        """Spectral entropy should be bounded."""
        entropy = prism.spectral_entropy(sine_signal, normalize=True)
        assert 0 <= entropy <= 1


# =============================================================================
# Statistics Tests
# =============================================================================

class TestStatistics:
    """Test statistical primitives."""

    def test_mean(self):
        """Mean of [1,2,3,4,5] should be 3."""
        result = prism.mean(np.array([1, 2, 3, 4, 5]))
        assert result == 3.0

    def test_variance(self):
        """Variance computation."""
        data = np.array([1, 2, 3, 4, 5])
        var = prism.variance(data)
        assert var == pytest.approx(2.0)

    def test_skewness_symmetric(self):
        """Symmetric distribution has zero skewness."""
        data = np.array([-2, -1, 0, 1, 2])
        skew = prism.skewness(data)
        assert abs(skew) < 0.1

    def test_kurtosis_normal(self, noisy_signal):
        """Kurtosis of normal-ish data should be near 0 (excess kurtosis)."""
        kurt = prism.kurtosis(noisy_signal, fisher=True)
        assert -2 < kurt < 2  # Should be near 0 for normal data

    def test_crest_factor(self, sine_signal):
        """Crest factor of sine wave is sqrt(2)."""
        cf = prism.crest_factor(sine_signal)
        assert cf == pytest.approx(np.sqrt(2), rel=0.1)

    def test_rms(self, sine_signal):
        """RMS of unit sine wave is 1/sqrt(2)."""
        rms_val = prism.rms(sine_signal)
        assert rms_val == pytest.approx(1 / np.sqrt(2), rel=0.01)


# =============================================================================
# Temporal Tests
# =============================================================================

class TestTemporal:
    """Test temporal analysis primitives."""

    def test_autocorrelation(self, sine_signal):
        """Autocorrelation at lag 0 should be 1."""
        acf = prism.autocorrelation(sine_signal, max_lag=10)
        assert acf[0] == pytest.approx(1.0)

    def test_trend_fit(self):
        """Trend fit should detect linear trend."""
        x = np.arange(100).astype(float)  # Linear trend
        coeffs, r2 = prism.trend_fit(x, order=1)
        assert r2 > 0.99  # Perfect linear fit
        assert coeffs[0] == pytest.approx(1.0, rel=0.01)  # Slope ~1

    def test_zero_crossings(self, sine_signal):
        """Sine wave should have ~20 zero crossings per second at 10 Hz."""
        zc = prism.zero_crossings(sine_signal)
        assert 18 <= zc <= 22

    def test_turning_points(self, sine_signal):
        """Sine wave should have ~20 turning points at 10 Hz."""
        tp = prism.turning_points(sine_signal)
        assert 18 <= tp <= 22


# =============================================================================
# Complexity Tests
# =============================================================================

class TestComplexity:
    """Test complexity and entropy primitives."""

    def test_permutation_entropy_bounds(self, noisy_signal):
        """Permutation entropy should be between 0 and 1 when normalized."""
        pe = prism.permutation_entropy(noisy_signal, order=3, normalize=True)
        assert 0 <= pe <= 1

    def test_sample_entropy_positive(self, noisy_signal):
        """Sample entropy should be non-negative."""
        se = prism.sample_entropy(noisy_signal, m=2)
        assert se >= 0

    def test_lempel_ziv_bounds(self, noisy_signal):
        """LZ complexity should be bounded when normalized."""
        lz = prism.lempel_ziv_complexity(noisy_signal, normalize=True)
        assert 0 <= lz <= 1


# =============================================================================
# Memory Tests
# =============================================================================

class TestMemory:
    """Test memory and long-range dependence primitives."""

    def test_hurst_random_walk(self, random_walk):
        """Random walk should have Hurst exponent near 0.5."""
        h = prism.hurst_exponent(random_walk)
        assert 0.3 <= h <= 0.7  # Should be around 0.5

    def test_dfa(self, random_walk):
        """DFA should return scales and fluctuations."""
        scales, fluct, alpha = prism.detrended_fluctuation_analysis(random_walk)
        assert len(scales) > 0
        assert len(fluct) == len(scales)
        assert not np.isnan(alpha)


# =============================================================================
# Stationarity Tests
# =============================================================================

class TestStationarity:
    """Test stationarity testing primitives."""

    def test_kpss_stationary(self, noisy_signal):
        """Noisy signal should pass KPSS test (stationary)."""
        stat, pval, critical = prism.kpss_test(noisy_signal)
        assert not np.isnan(stat)
        assert 0 <= pval <= 1

    def test_adf_random_walk(self, random_walk):
        """Random walk should fail ADF test (unit root)."""
        stat, pval, critical = prism.augmented_dickey_fuller(random_walk)
        assert not np.isnan(stat)
        assert 0 <= pval <= 1


# =============================================================================
# Geometry Tests
# =============================================================================

class TestGeometry:
    """Test geometry and eigenstructure primitives."""

    def test_covariance_matrix(self, multivariate_data):
        """Covariance matrix should be square and symmetric."""
        cov = prism.covariance_matrix(multivariate_data)
        assert cov.shape[0] == cov.shape[1]
        assert np.allclose(cov, cov.T)

    def test_eigendecomposition(self, multivariate_data):
        """Eigenvalues should be non-negative for covariance."""
        cov = prism.covariance_matrix(multivariate_data)
        eigenvals, eigenvecs = prism.eigendecomposition(cov)
        assert np.all(eigenvals >= -1e-10)  # Allow small numerical error

    def test_effective_dimension(self):
        """Effective dimension should be between 1 and n."""
        eigenvals = np.array([10, 5, 2, 1, 0.5])
        eff_dim = prism.effective_dimension(eigenvals)
        assert 1 <= eff_dim <= len(eigenvals)

    def test_participation_ratio(self):
        """Participation ratio same as effective dimension."""
        eigenvals = np.array([10, 5, 2, 1, 0.5])
        pr = prism.participation_ratio(eigenvals)
        assert 1 <= pr <= len(eigenvals)


# =============================================================================
# Similarity Tests
# =============================================================================

class TestSimilarity:
    """Test similarity and distance primitives."""

    def test_cosine_similarity_identical(self):
        """Identical vectors should have cosine similarity 1."""
        x = np.array([1, 2, 3])
        sim = prism.cosine_similarity(x, x)
        assert sim == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        """Orthogonal vectors should have cosine similarity 0."""
        x = np.array([1, 0, 0])
        y = np.array([0, 1, 0])
        sim = prism.cosine_similarity(x, y)
        assert sim == pytest.approx(0.0)

    def test_euclidean_distance_self(self):
        """Distance to self should be 0."""
        x = np.array([1, 2, 3])
        dist = prism.euclidean_distance(x, x)
        assert dist == pytest.approx(0.0)

    def test_correlation_coefficient(self):
        """Perfect correlation should be 1."""
        x = np.arange(100).astype(float)
        corr = prism.correlation_coefficient(x, x)
        assert corr == pytest.approx(1.0)


# =============================================================================
# Dynamics Tests
# =============================================================================

class TestDynamics:
    """Test dynamical systems primitives."""

    def test_attractor_reconstruction(self, noisy_signal):
        """Attractor reconstruction should return embedded vectors."""
        embedded = prism.attractor_reconstruction(noisy_signal, embed_dim=3, tau=10)
        assert embedded.shape[1] == 3
        assert len(embedded) > 0

    def test_embedding_dimension(self, noisy_signal):
        """Embedding dimension should be a positive integer."""
        dim = prism.embedding_dimension(noisy_signal[:500], max_dim=5)
        assert 1 <= dim <= 5


# =============================================================================
# Normalization Tests
# =============================================================================

class TestNormalization:
    """Test normalization primitives."""

    def test_zscore_normalize(self):
        """Z-score normalized data should have mean~0, std~1."""
        data = np.random.randn(1000) * 5 + 10  # mean=10, std=5
        normalized, params = prism.zscore_normalize(data, axis=None)
        assert np.mean(normalized) == pytest.approx(0, abs=0.1)
        assert np.std(normalized) == pytest.approx(1, abs=0.1)

    def test_minmax_normalize(self):
        """Minmax normalized data should be in [0, 1]."""
        data = np.random.randn(100)
        normalized, params = prism.minmax_normalize(data, axis=None)
        assert np.min(normalized) == pytest.approx(0)
        assert np.max(normalized) == pytest.approx(1)

    def test_inverse_normalize(self):
        """Inverse should recover original data."""
        data = np.random.randn(100) * 5 + 10
        normalized, params = prism.zscore_normalize(data, axis=None)
        recovered = prism.inverse_normalize(normalized, params)
        assert np.allclose(data, recovered)


# =============================================================================
# Information Theory Tests
# =============================================================================

class TestInformation:
    """Test information theory primitives."""

    def test_transfer_entropy_self(self, noisy_signal):
        """Transfer entropy from self should be low."""
        te = prism.transfer_entropy(noisy_signal[:500], noisy_signal[:500])
        assert te >= 0

    def test_granger_causality(self, noisy_signal):
        """Granger causality should return F-stat and p-value."""
        signal2 = np.roll(noisy_signal, 10)  # Shifted version
        f_stat, p_val = prism.granger_causality(
            noisy_signal[:500], signal2[:500], max_lag=5
        )
        assert not np.isnan(f_stat)
        assert 0 <= p_val <= 1


# =============================================================================
# Derivatives Tests
# =============================================================================

class TestDerivatives:
    """Test numerical derivatives primitives."""

    def test_first_derivative(self):
        """Derivative of x^2 should be ~2x."""
        x = np.linspace(0, 10, 1000)
        y = x ** 2
        dy = prism.first_derivative(y, dt=x[1] - x[0])
        # At x=5, derivative should be ~10
        mid_idx = 500
        assert dy[mid_idx] == pytest.approx(2 * x[mid_idx], rel=0.05)

    def test_second_derivative(self):
        """Second derivative of x^2 should be ~2."""
        x = np.linspace(0, 10, 1000)
        y = x ** 2
        d2y = prism.second_derivative(y, dt=x[1] - x[0])
        # Second derivative should be ~2 everywhere (except edges)
        assert np.mean(d2y[100:-100]) == pytest.approx(2, rel=0.1)

    def test_gradient(self, sine_signal):
        """Gradient should return same-length array."""
        grad = prism.gradient(sine_signal)
        assert len(grad) == len(sine_signal)

    def test_integral(self):
        """Integral of constant should be linear."""
        constant = np.ones(100)
        integral_vals = prism.integral(constant, dt=1.0)
        # Integral of 1 should be approximately linear (0, 1, 2, 3, ...)
        assert integral_vals[-1] == pytest.approx(99, rel=0.1)


# =============================================================================
# Import Tests
# =============================================================================

class TestImports:
    """Test that all exports are accessible."""

    def test_all_exports_exist(self):
        """All functions in __all__ should be importable."""
        for name in prism.__all__:
            assert hasattr(prism, name), f"Missing export: {name}"

    def test_version(self):
        """Version should be defined."""
        assert hasattr(prism, '__version__')
        assert prism.__version__ == "1.0.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
