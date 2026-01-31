-- =============================================================================
-- RECURRENCE REPORTS
-- Recurrence Quantification Analysis (RQA)
-- =============================================================================

-- Report: RQA Summary
-- Overview of recurrence metrics across all entities
SELECT
    unit_id,
    n_samples,
    recurrence_rate,
    determinism,
    laminarity,
    trapping_time,
    entropy,
    divergence,
    det_status,
    lam_status,
    CASE
        WHEN determinism < 0.5 THEN 'UNPREDICTABLE'
        WHEN laminarity > 0.7 THEN 'STICKING'
        WHEN divergence > 0.1 THEN 'DIVERGING'
        ELSE 'NORMAL'
    END AS rqa_status
FROM read_parquet('dynamics_recurrence.parquet')
ORDER BY determinism ASC;

-- Report: Determinism Alerts
-- Entities with low or dropping determinism (becoming unpredictable)
SELECT
    unit_id,
    determinism,
    det_status,
    det_trend,
    det_trend_p,
    det_trend_slope,
    CASE
        WHEN determinism < 0.3 THEN 'CRITICAL'
        WHEN determinism < 0.5 THEN 'WARNING'
        WHEN det_trend = 'decreasing' AND det_trend_p < 0.05 THEN 'DEGRADING'
        ELSE 'NORMAL'
    END AS alert_level,
    CASE
        WHEN determinism < 0.3 THEN 'System highly unpredictable - investigate immediately'
        WHEN determinism < 0.5 THEN 'System becoming chaotic - monitor closely'
        WHEN det_trend = 'decreasing' THEN 'Determinism decreasing - early warning'
        ELSE 'Normal operation'
    END AS recommendation
FROM read_parquet('dynamics_recurrence.parquet')
WHERE determinism < 0.7 OR (det_trend = 'decreasing' AND det_trend_p < 0.1)
ORDER BY determinism ASC;

-- Report: Laminarity Alerts (System Sticking)
-- Entities with high laminarity (system getting stuck in states)
SELECT
    unit_id,
    laminarity,
    trapping_time,
    lam_status,
    recurrence_rate,
    CASE
        WHEN laminarity > 0.8 THEN 'SEVERE_STICKING'
        WHEN laminarity > 0.6 THEN 'MODERATE_STICKING'
        ELSE 'NORMAL'
    END AS sticking_severity,
    'High laminarity indicates system dwelling in states - check for seizure/blockage' AS note
FROM read_parquet('dynamics_recurrence.parquet')
WHERE laminarity > 0.5
ORDER BY laminarity DESC;

-- Report: Divergence Analysis
-- Entities with high divergence (instability indicator)
SELECT
    unit_id,
    divergence,
    determinism,
    entropy,
    CASE
        WHEN divergence > 0.2 THEN 'HIGH_DIVERGENCE'
        WHEN divergence > 0.1 THEN 'MODERATE_DIVERGENCE'
        ELSE 'NORMAL'
    END AS divergence_level
FROM read_parquet('dynamics_recurrence.parquet')
WHERE divergence > 0.05
ORDER BY divergence DESC;

-- Report: RQA Status Summary
-- Count of entities by RQA status
SELECT
    det_status,
    lam_status,
    COUNT(*) as n_entities,
    AVG(determinism) as avg_determinism,
    AVG(laminarity) as avg_laminarity,
    AVG(entropy) as avg_entropy
FROM read_parquet('dynamics_recurrence.parquet')
GROUP BY det_status, lam_status
ORDER BY n_entities DESC;

-- Report: Recurrence Entropy Analysis
-- Complexity analysis via entropy
SELECT
    unit_id,
    entropy,
    determinism,
    recurrence_rate,
    CASE
        WHEN entropy > 3.0 THEN 'HIGH_COMPLEXITY'
        WHEN entropy > 2.0 THEN 'MODERATE_COMPLEXITY'
        WHEN entropy > 1.0 THEN 'LOW_COMPLEXITY'
        ELSE 'VERY_SIMPLE'
    END AS complexity_class
FROM read_parquet('dynamics_recurrence.parquet')
ORDER BY entropy DESC;

-- Report: Determinism Trends
-- Track how determinism is changing over time
SELECT
    unit_id,
    det_trend,
    det_trend_p,
    det_trend_slope,
    determinism,
    CASE
        WHEN det_trend = 'decreasing' AND det_trend_p < 0.01 THEN 'RAPID_DEGRADATION'
        WHEN det_trend = 'decreasing' AND det_trend_p < 0.05 THEN 'DEGRADING'
        WHEN det_trend = 'increasing' AND det_trend_p < 0.05 THEN 'IMPROVING'
        ELSE 'STABLE'
    END AS trend_status
FROM read_parquet('dynamics_recurrence.parquet')
WHERE det_trend != 'no trend'
ORDER BY det_trend_p ASC;
