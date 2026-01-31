-- =============================================================================
-- BASELINE REPORTS
-- Fleet and entity baseline statistics
-- =============================================================================

-- Report: Baseline Overview
-- All baseline statistics
SELECT
    metric_source,
    metric_name,
    unit_id,
    mean,
    std,
    median,
    p5,
    p25,
    p75,
    p95,
    min,
    max,
    n_samples,
    cv,
    iqr
FROM read_parquet('baseline.parquet')
ORDER BY metric_source, metric_name, unit_id;

-- Report: Fleet Baselines Only
-- Aggregated fleet baselines
SELECT
    metric_source,
    metric_name,
    mean,
    std,
    median,
    p5,
    p95,
    n_samples,
    cv
FROM read_parquet('baseline.parquet')
WHERE unit_id = 'FLEET'
ORDER BY metric_source, metric_name;

-- Report: High Variance Metrics
-- Metrics with coefficient of variation > 0.5
SELECT
    metric_source,
    metric_name,
    mean,
    std,
    cv,
    CASE
        WHEN cv > 1.0 THEN 'VERY_HIGH_VARIANCE'
        WHEN cv > 0.5 THEN 'HIGH_VARIANCE'
        WHEN cv > 0.25 THEN 'MODERATE_VARIANCE'
        ELSE 'LOW_VARIANCE'
    END AS variance_level,
    'High variance may indicate unstable baseline' AS note
FROM read_parquet('baseline.parquet')
WHERE unit_id = 'FLEET' AND cv > 0.25
ORDER BY cv DESC;

-- Report: Entity vs Fleet Comparison
-- How each entity deviates from fleet baseline
SELECT
    e.metric_source,
    e.metric_name,
    e.unit_id,
    e.mean AS entity_mean,
    f.mean AS fleet_mean,
    e.mean - f.mean AS deviation,
    CASE
        WHEN f.std > 0 THEN (e.mean - f.mean) / f.std
        ELSE 0
    END AS z_from_fleet,
    CASE
        WHEN ABS((e.mean - f.mean) / NULLIF(f.std, 0)) > 2 THEN 'OUTLIER'
        WHEN ABS((e.mean - f.mean) / NULLIF(f.std, 0)) > 1 THEN 'ELEVATED'
        ELSE 'NORMAL'
    END AS deviation_status
FROM read_parquet('baseline.parquet') e
JOIN read_parquet('baseline.parquet') f
    ON e.metric_source = f.metric_source
    AND e.metric_name = f.metric_name
    AND f.unit_id = 'FLEET'
WHERE e.unit_id != 'FLEET'
ORDER BY ABS((e.mean - f.mean) / NULLIF(f.std, 0)) DESC;

-- Report: Metrics with Wide Ranges
-- Metrics where p95 - p5 is large relative to median
SELECT
    metric_source,
    metric_name,
    median,
    p5,
    p95,
    iqr,
    (p95 - p5) AS range_90,
    (p95 - p5) / NULLIF(ABS(median), 0.001) AS relative_range,
    CASE
        WHEN (p95 - p5) / NULLIF(ABS(median), 0.001) > 1.0 THEN 'WIDE_RANGE'
        ELSE 'NORMAL_RANGE'
    END AS range_status
FROM read_parquet('baseline.parquet')
WHERE unit_id = 'FLEET'
ORDER BY relative_range DESC;

-- Report: Baseline Sample Counts
-- Number of samples used for each baseline
SELECT
    metric_source,
    COUNT(DISTINCT metric_name) AS n_metrics,
    AVG(n_samples) AS avg_samples,
    MIN(n_samples) AS min_samples,
    MAX(n_samples) AS max_samples
FROM read_parquet('baseline.parquet')
WHERE unit_id = 'FLEET'
GROUP BY metric_source
ORDER BY avg_samples DESC;
