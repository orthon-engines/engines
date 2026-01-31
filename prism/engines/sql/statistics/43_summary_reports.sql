-- =============================================================================
-- SUMMARY REPORTS
-- Executive dashboards and KPIs
-- =============================================================================

-- Report: Executive Dashboard KPIs
-- Key performance indicators
SELECT
    (SELECT COUNT(DISTINCT unit_id) FROM read_parquet('health.parquet')) AS total_entities,
    (SELECT ROUND(AVG(health_score), 1)
     FROM read_parquet('health.parquet')
     WHERE window_id = (SELECT MAX(window_id) FROM read_parquet('health.parquet'))) AS current_avg_health,
    (SELECT COUNT(*)
     FROM read_parquet('health.parquet')
     WHERE window_id = (SELECT MAX(window_id) FROM read_parquet('health.parquet'))
       AND risk_level = 'CRITICAL') AS critical_count,
    (SELECT COUNT(*)
     FROM read_parquet('health.parquet')
     WHERE window_id = (SELECT MAX(window_id) FROM read_parquet('health.parquet'))
       AND risk_level = 'HIGH') AS high_risk_count,
    (SELECT COUNT(*)
     FROM read_parquet('anomaly.parquet')
     WHERE is_anomaly = TRUE
       AND window_id = (SELECT MAX(window_id) FROM read_parquet('anomaly.parquet'))) AS current_anomalies;

-- Report: Health Score Distribution
-- Histogram of health scores
SELECT
    CASE
        WHEN health_score >= 90 THEN '90-100 (Excellent)'
        WHEN health_score >= 80 THEN '80-89 (Good)'
        WHEN health_score >= 70 THEN '70-79 (Fair)'
        WHEN health_score >= 60 THEN '60-69 (Poor)'
        WHEN health_score >= 50 THEN '50-59 (At Risk)'
        ELSE '< 50 (Critical)'
    END AS health_bucket,
    COUNT(*) AS n_entities,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
FROM read_parquet('health.parquet')
WHERE window_id = (SELECT MAX(window_id) FROM read_parquet('health.parquet'))
GROUP BY health_bucket
ORDER BY
    CASE health_bucket
        WHEN '90-100 (Excellent)' THEN 1
        WHEN '80-89 (Good)' THEN 2
        WHEN '70-79 (Fair)' THEN 3
        WHEN '60-69 (Poor)' THEN 4
        WHEN '50-59 (At Risk)' THEN 5
        ELSE 6
    END;

-- Report: Risk Level Trend
-- Risk distribution over time
SELECT
    window_id,
    COUNT(CASE WHEN risk_level = 'LOW' THEN 1 END) AS low,
    COUNT(CASE WHEN risk_level = 'MODERATE' THEN 1 END) AS moderate,
    COUNT(CASE WHEN risk_level = 'HIGH' THEN 1 END) AS high,
    COUNT(CASE WHEN risk_level = 'CRITICAL' THEN 1 END) AS critical,
    ROUND(AVG(health_score), 1) AS avg_health
FROM read_parquet('health.parquet')
GROUP BY window_id
ORDER BY window_id;

-- Report: Top Concerns Summary
-- Most common concerns
SELECT
    primary_concern,
    COUNT(*) AS n_entities,
    ROUND(AVG(health_score), 1) AS avg_health_affected,
    COUNT(CASE WHEN risk_level IN ('HIGH', 'CRITICAL') THEN 1 END) AS high_risk_count
FROM read_parquet('health.parquet')
WHERE window_id = (SELECT MAX(window_id) FROM read_parquet('health.parquet'))
  AND primary_concern != 'None'
GROUP BY primary_concern
ORDER BY n_entities DESC;

-- Report: Health Trend Summary
-- Overall health trajectory
SELECT
    MIN(window_id) AS first_window,
    MAX(window_id) AS last_window,
    (SELECT ROUND(AVG(health_score), 1)
     FROM read_parquet('health.parquet')
     WHERE window_id = (SELECT MIN(window_id) FROM read_parquet('health.parquet'))) AS initial_health,
    (SELECT ROUND(AVG(health_score), 1)
     FROM read_parquet('health.parquet')
     WHERE window_id = (SELECT MAX(window_id) FROM read_parquet('health.parquet'))) AS current_health,
    CASE
        WHEN (SELECT AVG(health_score) FROM read_parquet('health.parquet')
              WHERE window_id = (SELECT MAX(window_id) FROM read_parquet('health.parquet'))) >
             (SELECT AVG(health_score) FROM read_parquet('health.parquet')
              WHERE window_id = (SELECT MIN(window_id) FROM read_parquet('health.parquet'))) + 2
        THEN 'IMPROVING'
        WHEN (SELECT AVG(health_score) FROM read_parquet('health.parquet')
              WHERE window_id = (SELECT MAX(window_id) FROM read_parquet('health.parquet'))) <
             (SELECT AVG(health_score) FROM read_parquet('health.parquet')
              WHERE window_id = (SELECT MIN(window_id) FROM read_parquet('health.parquet'))) - 2
        THEN 'DEGRADING'
        ELSE 'STABLE'
    END AS trend
FROM read_parquet('health.parquet');

-- Report: Recommendation Distribution
-- Count by recommendation type
SELECT
    recommendation,
    COUNT(*) AS n_entities,
    ROUND(AVG(health_score), 1) AS avg_health
FROM read_parquet('health.parquet')
WHERE window_id = (SELECT MAX(window_id) FROM read_parquet('health.parquet'))
GROUP BY recommendation
ORDER BY
    CASE recommendation
        WHEN 'IMMEDIATE INSPECTION REQUIRED' THEN 1
        WHEN 'Schedule maintenance within 1 week' THEN 2
        WHEN 'Monitor closely, plan maintenance' THEN 3
        WHEN 'Normal operation' THEN 4
        ELSE 5
    END;

-- Report: Domain Risk Summary
-- Risk scores by domain
SELECT
    'Stability' AS domain,
    ROUND(AVG(stability_score) * 100, 1) AS avg_risk_pct,
    COUNT(CASE WHEN stability_score > 0.5 THEN 1 END) AS high_risk_count
FROM read_parquet('health.parquet')
WHERE window_id = (SELECT MAX(window_id) FROM read_parquet('health.parquet'))
UNION ALL
SELECT
    'Predictability' AS domain,
    ROUND(AVG(predictability_score) * 100, 1) AS avg_risk_pct,
    COUNT(CASE WHEN predictability_score > 0.5 THEN 1 END) AS high_risk_count
FROM read_parquet('health.parquet')
WHERE window_id = (SELECT MAX(window_id) FROM read_parquet('health.parquet'))
UNION ALL
SELECT
    'Physics' AS domain,
    ROUND(AVG(physics_score) * 100, 1) AS avg_risk_pct,
    COUNT(CASE WHEN physics_score > 0.5 THEN 1 END) AS high_risk_count
FROM read_parquet('health.parquet')
WHERE window_id = (SELECT MAX(window_id) FROM read_parquet('health.parquet'))
UNION ALL
SELECT
    'Topology' AS domain,
    ROUND(AVG(topology_score) * 100, 1) AS avg_risk_pct,
    COUNT(CASE WHEN topology_score > 0.5 THEN 1 END) AS high_risk_count
FROM read_parquet('health.parquet')
WHERE window_id = (SELECT MAX(window_id) FROM read_parquet('health.parquet'))
UNION ALL
SELECT
    'Causality' AS domain,
    ROUND(AVG(causality_score) * 100, 1) AS avg_risk_pct,
    COUNT(CASE WHEN causality_score > 0.5 THEN 1 END) AS high_risk_count
FROM read_parquet('health.parquet')
WHERE window_id = (SELECT MAX(window_id) FROM read_parquet('health.parquet'));

-- Report: Watch List
-- Entities requiring attention
SELECT
    unit_id,
    health_score,
    risk_level,
    primary_concern,
    recommendation
FROM read_parquet('health.parquet')
WHERE window_id = (SELECT MAX(window_id) FROM read_parquet('health.parquet'))
  AND risk_level IN ('MODERATE', 'HIGH', 'CRITICAL')
ORDER BY health_score ASC;
