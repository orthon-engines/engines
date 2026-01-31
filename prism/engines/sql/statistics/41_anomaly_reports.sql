-- =============================================================================
-- ANOMALY REPORTS
-- Deviation detection and scoring
-- =============================================================================

-- Report: Current Anomalies
-- All anomalies sorted by severity
SELECT
    unit_id,
    window_id,
    metric_source,
    metric_name,
    value,
    baseline_mean,
    baseline_std,
    z_score,
    percentile_rank,
    anomaly_severity
FROM read_parquet('anomaly.parquet')
WHERE is_anomaly = TRUE
ORDER BY
    CASE anomaly_severity
        WHEN 'CRITICAL' THEN 1
        WHEN 'WARNING' THEN 2
        ELSE 3
    END,
    ABS(z_score) DESC;

-- Report: Critical Anomalies Only
-- Highest severity anomalies
SELECT
    unit_id,
    window_id,
    metric_source,
    metric_name,
    value,
    z_score,
    'IMMEDIATE ATTENTION' AS action
FROM read_parquet('anomaly.parquet')
WHERE anomaly_severity = 'CRITICAL'
ORDER BY ABS(z_score) DESC;

-- Report: Anomaly Summary by Entity
-- Count of anomalies per entity
SELECT
    unit_id,
    COUNT(*) AS total_anomalies,
    COUNT(CASE WHEN anomaly_severity = 'CRITICAL' THEN 1 END) AS critical,
    COUNT(CASE WHEN anomaly_severity = 'WARNING' THEN 1 END) AS warning,
    AVG(ABS(z_score)) AS avg_abs_z,
    MAX(ABS(z_score)) AS max_abs_z,
    CASE
        WHEN COUNT(CASE WHEN anomaly_severity = 'CRITICAL' THEN 1 END) > 0 THEN 'CRITICAL'
        WHEN COUNT(CASE WHEN anomaly_severity = 'WARNING' THEN 1 END) > 3 THEN 'HIGH'
        WHEN COUNT(*) > 5 THEN 'ELEVATED'
        ELSE 'NORMAL'
    END AS entity_anomaly_status
FROM read_parquet('anomaly.parquet')
WHERE is_anomaly = TRUE
GROUP BY unit_id
ORDER BY total_anomalies DESC;

-- Report: Anomaly Summary by Metric
-- Which metrics are most anomalous
SELECT
    metric_source,
    metric_name,
    COUNT(*) AS total_anomalies,
    COUNT(DISTINCT unit_id) AS affected_entities,
    AVG(z_score) AS avg_z,
    AVG(ABS(z_score)) AS avg_abs_z,
    MAX(ABS(z_score)) AS max_abs_z,
    CASE
        WHEN AVG(z_score) > 0 THEN 'HIGH_TRENDING'
        WHEN AVG(z_score) < 0 THEN 'LOW_TRENDING'
        ELSE 'MIXED'
    END AS anomaly_direction
FROM read_parquet('anomaly.parquet')
WHERE is_anomaly = TRUE
GROUP BY metric_source, metric_name
ORDER BY total_anomalies DESC;

-- Report: Anomaly Timeline
-- Anomalies over time
SELECT
    window_id,
    COUNT(*) AS n_anomalies,
    COUNT(CASE WHEN anomaly_severity = 'CRITICAL' THEN 1 END) AS critical,
    COUNT(CASE WHEN anomaly_severity = 'WARNING' THEN 1 END) AS warning,
    COUNT(DISTINCT unit_id) AS affected_entities,
    AVG(ABS(z_score)) AS avg_severity
FROM read_parquet('anomaly.parquet')
WHERE is_anomaly = TRUE
GROUP BY window_id
ORDER BY window_id;

-- Report: Anomaly Patterns
-- Metrics that are frequently anomalous together
SELECT
    a.metric_name AS metric_1,
    b.metric_name AS metric_2,
    COUNT(*) AS co_occurrence,
    AVG(a.z_score * b.z_score) AS correlation_sign
FROM read_parquet('anomaly.parquet') a
JOIN read_parquet('anomaly.parquet') b
    ON a.unit_id = b.unit_id
    AND a.window_id = b.window_id
    AND a.metric_name < b.metric_name
WHERE a.is_anomaly = TRUE AND b.is_anomaly = TRUE
GROUP BY a.metric_name, b.metric_name
HAVING COUNT(*) > 2
ORDER BY co_occurrence DESC;

-- Report: Z-Score Distribution
-- Distribution of z-scores
SELECT
    CASE
        WHEN ABS(z_score) > 5 THEN '>5 (Extreme)'
        WHEN ABS(z_score) > 4 THEN '4-5 (Very High)'
        WHEN ABS(z_score) > 3 THEN '3-4 (Critical)'
        WHEN ABS(z_score) > 2 THEN '2-3 (Warning)'
        WHEN ABS(z_score) > 1.5 THEN '1.5-2 (Elevated)'
        ELSE '<1.5 (Normal)'
    END AS z_score_bucket,
    COUNT(*) AS n_observations,
    COUNT(DISTINCT unit_id) AS n_entities
FROM read_parquet('anomaly.parquet')
GROUP BY z_score_bucket
ORDER BY
    CASE z_score_bucket
        WHEN '>5 (Extreme)' THEN 1
        WHEN '4-5 (Very High)' THEN 2
        WHEN '3-4 (Critical)' THEN 3
        WHEN '2-3 (Warning)' THEN 4
        WHEN '1.5-2 (Elevated)' THEN 5
        ELSE 6
    END;
