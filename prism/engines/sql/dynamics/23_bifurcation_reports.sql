-- =============================================================================
-- BIFURCATION / CRITICAL SLOWING DOWN REPORTS
-- Early warning signals for regime changes and tipping points
-- =============================================================================

-- Report: CSD Summary
-- Overview of critical slowing down indicators
SELECT
    unit_id,
    signal,
    n_samples,
    variance,
    autocorr_lag1,
    variance_trend,
    autocorr_trend,
    csd_score,
    approaching_bifurcation,
    csd_status
FROM read_parquet('dynamics_bifurcation.parquet')
ORDER BY csd_score DESC;

-- Report: Approaching Bifurcation Alerts
-- Entities showing strong early warning signals
SELECT
    unit_id,
    signal,
    variance,
    autocorr_lag1,
    variance_trend_slope,
    autocorr_trend_slope,
    csd_score,
    csd_status,
    'APPROACHING_BIFURCATION' AS alert_type,
    'Both variance and autocorrelation increasing - system approaching tipping point' AS warning
FROM read_parquet('dynamics_bifurcation.parquet')
WHERE approaching_bifurcation = TRUE
ORDER BY csd_score DESC;

-- Report: CSD Score Distribution
-- Classification of entities by CSD score
SELECT
    csd_status,
    COUNT(*) as n_entities,
    COUNT(DISTINCT unit_id) as n_unique_entities,
    AVG(csd_score) as avg_csd_score,
    AVG(variance) as avg_variance,
    AVG(autocorr_lag1) as avg_autocorr
FROM read_parquet('dynamics_bifurcation.parquet')
GROUP BY csd_status
ORDER BY avg_csd_score DESC;

-- Report: Variance Trend Analysis
-- Entities with increasing variance (early warning)
SELECT
    unit_id,
    signal,
    variance,
    variance_trend,
    variance_trend_p,
    variance_trend_slope,
    csd_score,
    CASE
        WHEN variance_trend = 'increasing' AND variance_trend_p < 0.01 THEN 'RAPID_INCREASE'
        WHEN variance_trend = 'increasing' AND variance_trend_p < 0.05 THEN 'INCREASING'
        WHEN variance_trend = 'decreasing' THEN 'DECREASING'
        ELSE 'STABLE'
    END AS variance_status
FROM read_parquet('dynamics_bifurcation.parquet')
WHERE variance_trend = 'increasing'
ORDER BY variance_trend_slope DESC;

-- Report: Autocorrelation Trend Analysis
-- Entities with increasing autocorrelation (critical slowing)
SELECT
    unit_id,
    signal,
    autocorr_lag1,
    autocorr_lag5,
    autocorr_trend,
    autocorr_trend_p,
    autocorr_trend_slope,
    csd_score,
    CASE
        WHEN autocorr_trend = 'increasing' AND autocorr_trend_p < 0.01 THEN 'RAPID_SLOWING'
        WHEN autocorr_trend = 'increasing' AND autocorr_trend_p < 0.05 THEN 'SLOWING'
        ELSE 'NORMAL'
    END AS slowing_status
FROM read_parquet('dynamics_bifurcation.parquet')
WHERE autocorr_trend = 'increasing'
ORDER BY autocorr_trend_slope DESC;

-- Report: Higher Moments Analysis
-- Skewness and kurtosis for distributional changes
SELECT
    unit_id,
    signal,
    skewness,
    kurtosis,
    variance,
    csd_score,
    CASE
        WHEN ABS(skewness) > 1.0 THEN 'HIGH_SKEW'
        WHEN ABS(skewness) > 0.5 THEN 'MODERATE_SKEW'
        ELSE 'SYMMETRIC'
    END AS skewness_class,
    CASE
        WHEN kurtosis > 3.0 THEN 'HEAVY_TAILS'
        WHEN kurtosis < -1.0 THEN 'LIGHT_TAILS'
        ELSE 'NORMAL_TAILS'
    END AS kurtosis_class
FROM read_parquet('dynamics_bifurcation.parquet')
ORDER BY ABS(skewness) + ABS(kurtosis) DESC;

-- Report: Entity Risk Summary
-- Aggregate risk by entity across all signals
SELECT
    unit_id,
    COUNT(*) as n_signals_analyzed,
    AVG(csd_score) as avg_csd_score,
    MAX(csd_score) as max_csd_score,
    SUM(CASE WHEN approaching_bifurcation THEN 1 ELSE 0 END) as n_bifurcation_warnings,
    SUM(CASE WHEN csd_status = 'CRITICAL' THEN 1 ELSE 0 END) as n_critical,
    SUM(CASE WHEN csd_status = 'WARNING' THEN 1 ELSE 0 END) as n_warning,
    CASE
        WHEN MAX(CASE WHEN approaching_bifurcation THEN 1 ELSE 0 END) = 1 THEN 'CRITICAL_RISK'
        WHEN AVG(csd_score) > 2.0 THEN 'HIGH_RISK'
        WHEN AVG(csd_score) > 1.0 THEN 'ELEVATED_RISK'
        ELSE 'NORMAL'
    END AS entity_risk_level
FROM read_parquet('dynamics_bifurcation.parquet')
GROUP BY unit_id
ORDER BY max_csd_score DESC;

-- Report: Signal Comparison
-- Compare CSD indicators across different signals
SELECT
    signal,
    COUNT(DISTINCT unit_id) as n_entities,
    AVG(variance) as avg_variance,
    AVG(autocorr_lag1) as avg_autocorr,
    AVG(csd_score) as avg_csd_score,
    SUM(CASE WHEN approaching_bifurcation THEN 1 ELSE 0 END) as n_approaching_bifurcation
FROM read_parquet('dynamics_bifurcation.parquet')
GROUP BY signal
ORDER BY avg_csd_score DESC;
