"""
Critical Slowing Down Engine

Detects early warning signals of approaching bifurcation (B-tipping).

Near a critical transition, systems exhibit CRITICAL SLOWING DOWN:
the rate of recovery from perturbations decreases as the system
approaches the tipping point. (Scheffer 2009)

Detectable signatures:
1. Increased autocorrelation (memory increases)
2. Increased variance (fluctuations grow)
3. Decreased recovery rate (return to equilibrium slower)
4. Increased skewness (asymmetric fluctuations)

IMPORTANT: These signals apply to B-TIPPING (bifurcation-induced).
R-TIPPING (rate-induced) may show NO warning!

Outputs:
    - autocorrelation_lag1: AR(1) coefficient
    - variance: Signal variance
    - variance_trend: Slope of rolling variance
    - autocorr_trend: Slope of rolling autocorrelation
    - recovery_rate: Estimated recovery rate (1/λ)
    - csd_score: Composite critical slowing down score
    - csd_detected: Boolean flag
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy.stats import linregress


def compute(
    values: np.ndarray,
    window_size: int = 50,
    step: int = 10,
    detrend: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute critical slowing down indicators.

    Args:
        values: 1D time series
        window_size: Rolling window for local statistics
        step: Step size for rolling computation
        detrend: Remove linear trend before analysis

    Returns:
        Dict with CSD indicators and detection flag
    """
    values = np.asarray(values).flatten()
    n = len(values)

    if n < window_size * 2:
        return _empty_result("Insufficient data")

    # Remove NaN
    valid_mask = ~np.isnan(values)
    if valid_mask.sum() < window_size * 2:
        return _empty_result("Too many NaN values")

    values = values[valid_mask]
    n = len(values)

    # Optionally detrend (important for non-stationary data)
    if detrend:
        t = np.arange(n)
        slope, intercept, _, _, _ = linregress(t, values)
        values = values - (slope * t + intercept)

    # Compute rolling statistics
    n_windows = (n - window_size) // step + 1

    autocorr_series = []
    variance_series = []
    skewness_series = []

    for i in range(n_windows):
        start = i * step
        end = start + window_size
        window = values[start:end]

        # Autocorrelation at lag 1
        ac1 = _autocorr_lag1(window)
        autocorr_series.append(ac1)

        # Variance
        var = np.var(window)
        variance_series.append(var)

        # Skewness
        skew = _skewness(window)
        skewness_series.append(skew)

    autocorr_series = np.array(autocorr_series)
    variance_series = np.array(variance_series)
    skewness_series = np.array(skewness_series)

    # Compute trends
    t = np.arange(len(autocorr_series))

    # Autocorrelation trend (should increase before tipping)
    ac_slope, ac_intercept, ac_r, _, _ = linregress(t, autocorr_series)

    # Variance trend (should increase before tipping)
    var_slope, var_intercept, var_r, _, _ = linregress(t, variance_series)

    # Current values (end of series)
    current_autocorr = autocorr_series[-1]
    current_variance = variance_series[-1]
    current_skewness = skewness_series[-1]

    # Baseline values (start of series)
    baseline_variance = variance_series[0]

    # Recovery rate estimate (from autocorrelation)
    # AR(1): x_t = ρ*x_{t-1} + ε
    # Recovery rate ≈ -ln(ρ) for |ρ| < 1
    if 0 < current_autocorr < 1:
        recovery_rate = -np.log(current_autocorr)
    else:
        recovery_rate = np.nan

    # Compute CSD score (composite indicator)
    # Higher score = more signs of critical slowing down
    csd_score = _compute_csd_score(
        autocorr_trend=ac_slope,
        variance_trend=var_slope,
        current_autocorr=current_autocorr,
        variance_ratio=current_variance / baseline_variance if baseline_variance > 0 else 1.0,
    )

    # Detection threshold
    csd_detected = (
        csd_score > 0.6 and           # High composite score
        current_autocorr > 0.7 and    # High memory
        var_slope > 0                  # Variance increasing
    )

    return {
        # Current state
        'autocorrelation_lag1': float(current_autocorr),
        'variance': float(current_variance),
        'skewness': float(current_skewness),
        'recovery_rate': float(recovery_rate) if np.isfinite(recovery_rate) else None,

        # Trends
        'autocorr_trend': float(ac_slope),
        'autocorr_trend_r2': float(ac_r ** 2),
        'variance_trend': float(var_slope),
        'variance_trend_r2': float(var_r ** 2),
        'variance_ratio': float(current_variance / baseline_variance) if baseline_variance > 0 else None,

        # Composite
        'csd_score': float(csd_score),
        'csd_detected': csd_detected,

        # Series (for plotting)
        'autocorr_series': autocorr_series.tolist(),
        'variance_series': variance_series.tolist(),
    }


def _empty_result(reason: str) -> Dict[str, Any]:
    """Return empty result."""
    return {
        'autocorrelation_lag1': np.nan,
        'variance': np.nan,
        'skewness': np.nan,
        'recovery_rate': None,
        'autocorr_trend': np.nan,
        'autocorr_trend_r2': np.nan,
        'variance_trend': np.nan,
        'variance_trend_r2': np.nan,
        'variance_ratio': None,
        'csd_score': 0.0,
        'csd_detected': False,
        'reason': reason,
    }


def _autocorr_lag1(x: np.ndarray) -> float:
    """Compute autocorrelation at lag 1."""
    x = x - np.mean(x)
    n = len(x)
    if n < 2:
        return np.nan

    numerator = np.sum(x[:-1] * x[1:])
    denominator = np.sum(x ** 2)

    if denominator < 1e-10:
        return np.nan

    return numerator / denominator


def _skewness(x: np.ndarray) -> float:
    """Compute skewness."""
    x = x - np.mean(x)
    n = len(x)
    if n < 3:
        return np.nan

    std = np.std(x)
    if std < 1e-10:
        return 0.0

    return np.mean((x / std) ** 3)


def _compute_csd_score(
    autocorr_trend: float,
    variance_trend: float,
    current_autocorr: float,
    variance_ratio: float,
) -> float:
    """
    Compute composite critical slowing down score.

    Combines multiple indicators into 0-1 score.
    Higher = more evidence of approaching bifurcation.
    """
    score = 0.0
    weights_sum = 0.0

    # Autocorrelation level (weight: 0.3)
    # High autocorr = slow dynamics
    if np.isfinite(current_autocorr):
        ac_score = min(1.0, max(0.0, current_autocorr))
        score += 0.3 * ac_score
        weights_sum += 0.3

    # Autocorrelation trend (weight: 0.25)
    # Positive trend = slowing down
    if np.isfinite(autocorr_trend):
        ac_trend_score = min(1.0, max(0.0, autocorr_trend * 10))  # Scale
        score += 0.25 * ac_trend_score
        weights_sum += 0.25

    # Variance trend (weight: 0.25)
    # Positive trend = growing fluctuations
    if np.isfinite(variance_trend):
        var_trend_score = min(1.0, max(0.0, variance_trend * 100))  # Scale
        score += 0.25 * var_trend_score
        weights_sum += 0.25

    # Variance ratio (weight: 0.2)
    # High ratio = variance has grown
    if variance_ratio is not None and np.isfinite(variance_ratio):
        var_ratio_score = min(1.0, max(0.0, (variance_ratio - 1) / 2))
        score += 0.2 * var_ratio_score
        weights_sum += 0.2

    if weights_sum > 0:
        return score / weights_sum
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# ALTERNATIVE: KENDALL TAU TREND TEST
# ═══════════════════════════════════════════════════════════════════════════════

def kendall_tau_trend(
    values: np.ndarray,
    window_size: int = 50,
    step: int = 10,
) -> Dict[str, Any]:
    """
    Use Kendall tau for robust trend detection in CSD indicators.

    Kendall tau is more robust to outliers than linear regression.
    Used in Dakos et al. (2012) for CSD detection.
    """
    from scipy.stats import kendalltau

    values = np.asarray(values).flatten()
    n = len(values)

    if n < window_size * 2:
        return {'tau_autocorr': np.nan, 'tau_variance': np.nan}

    # Compute rolling statistics
    n_windows = (n - window_size) // step + 1

    autocorr_series = []
    variance_series = []

    for i in range(n_windows):
        start = i * step
        end = start + window_size
        window = values[start:end]

        ac1 = _autocorr_lag1(window)
        autocorr_series.append(ac1)
        variance_series.append(np.var(window))

    autocorr_series = np.array(autocorr_series)
    variance_series = np.array(variance_series)

    t = np.arange(len(autocorr_series))

    # Kendall tau for autocorrelation trend
    valid_ac = ~np.isnan(autocorr_series)
    if valid_ac.sum() > 5:
        tau_ac, p_ac = kendalltau(t[valid_ac], autocorr_series[valid_ac])
    else:
        tau_ac, p_ac = np.nan, np.nan

    # Kendall tau for variance trend
    valid_var = ~np.isnan(variance_series)
    if valid_var.sum() > 5:
        tau_var, p_var = kendalltau(t[valid_var], variance_series[valid_var])
    else:
        tau_var, p_var = np.nan, np.nan

    return {
        'tau_autocorr': float(tau_ac) if np.isfinite(tau_ac) else None,
        'tau_autocorr_pvalue': float(p_ac) if np.isfinite(p_ac) else None,
        'tau_variance': float(tau_var) if np.isfinite(tau_var) else None,
        'tau_variance_pvalue': float(p_var) if np.isfinite(p_var) else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Test critical slowing down detection."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Critical Slowing Down Detection"
    )
    parser.add_argument('--test', action='store_true', help='Run test cases')

    args = parser.parse_args()

    if args.test:
        print("=" * 70)
        print("CRITICAL SLOWING DOWN TESTS")
        print("=" * 70)

        np.random.seed(42)

        # Test 1: Stable noise (no CSD)
        print("\n1. White noise (no CSD expected):")
        noise = np.random.randn(500)
        result = compute(noise)
        print(f"   Autocorr: {result['autocorrelation_lag1']:.3f}")
        print(f"   CSD score: {result['csd_score']:.3f}")
        print(f"   CSD detected: {result['csd_detected']}")

        # Test 2: AR(1) with increasing autocorrelation (simulated CSD)
        print("\n2. AR(1) approaching instability (CSD expected):")
        # Simulate AR(1) where ρ increases from 0.5 to 0.95
        n = 500
        x = np.zeros(n)
        x[0] = np.random.randn()
        for i in range(1, n):
            rho = 0.5 + 0.45 * (i / n)  # ρ: 0.5 → 0.95
            x[i] = rho * x[i-1] + np.sqrt(1 - rho**2) * np.random.randn()
        result = compute(x)
        print(f"   Autocorr: {result['autocorrelation_lag1']:.3f}")
        print(f"   Autocorr trend: {result['autocorr_trend']:.4f}")
        print(f"   Variance trend: {result['variance_trend']:.4f}")
        print(f"   CSD score: {result['csd_score']:.3f}")
        print(f"   CSD detected: {result['csd_detected']}")

        # Test 3: Fold bifurcation approach
        print("\n3. Saddle-node bifurcation approach:")
        # dx/dt = r + x² approaching r=0
        n = 500
        dt = 0.1
        x = np.zeros(n)
        x[0] = -1.0
        for i in range(1, n):
            r = -1 + 0.99 * (i / n)  # r: -1 → -0.01 (approaching bifurcation at r=0)
            # At equilibrium: x* = -sqrt(-r) for r < 0
            # Linearization: recovery rate = 2*sqrt(-r)
            x_eq = -np.sqrt(-r) if r < 0 else 0
            # Noisy dynamics near equilibrium
            x[i] = x[i-1] + dt * (r + x[i-1]**2) + 0.1 * np.random.randn()
            x[i] = max(-2, min(2, x[i]))  # Bound
        result = compute(x, detrend=False)
        print(f"   Autocorr: {result['autocorrelation_lag1']:.3f}")
        print(f"   Variance ratio: {result['variance_ratio']:.2f}x")
        print(f"   CSD score: {result['csd_score']:.3f}")
        print(f"   CSD detected: {result['csd_detected']}")


if __name__ == "__main__":
    main()
