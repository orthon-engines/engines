-- =============================================================================
-- INTEGRATION / HEALTH REPORTS
-- Unified health assessment combining all engine outputs
-- =============================================================================

-- Report: Health Dashboard
-- Main health overview for all entities
SELECT
    unit_id,
    health_score,
    risk_level,
    stability_score,
    predictability_score,
    physics_score,
    topology_score,
    causality_score,
    primary_concern,
    secondary_concern,
    n_concerns,
    recommendation
FROM read_parquet('health.parquet')
ORDER BY health_score ASC;

-- Report: Critical Alerts
-- Entities requiring immediate attention
SELECT
    unit_id,
    health_score,
    risk_level,
    primary_concern,
    secondary_concern,
    recommendation,
    'IMMEDIATE ACTION REQUIRED' AS urgency
FROM read_parquet('health.parquet')
WHERE risk_level IN ('CRITICAL', 'HIGH')
ORDER BY health_score ASC;

-- Report: Risk Level Distribution
-- Count of entities by risk level
SELECT
    risk_level,
    COUNT(*) AS n_entities,
    AVG(health_score) AS avg_health,
    MIN(health_score) AS min_health,
    MAX(health_score) AS max_health
FROM read_parquet('health.parquet')
GROUP BY risk_level
ORDER BY
    CASE risk_level
        WHEN 'CRITICAL' THEN 1
        WHEN 'HIGH' THEN 2
        WHEN 'MODERATE' THEN 3
        WHEN 'LOW' THEN 4
        ELSE 5
    END;

-- Report: Score Breakdown
-- Detailed score analysis
SELECT
    unit_id,
    health_score,
    stability_score * 100 AS stability_pct,
    predictability_score * 100 AS predictability_pct,
    physics_score * 100 AS physics_pct,
    topology_score * 100 AS topology_pct,
    causality_score * 100 AS causality_pct,
    CASE
        WHEN stability_score >= predictability_score
             AND stability_score >= physics_score
             AND stability_score >= topology_score
             AND stability_score >= causality_score THEN 'STABILITY'
        WHEN predictability_score >= physics_score
             AND predictability_score >= topology_score
             AND predictability_score >= causality_score THEN 'PREDICTABILITY'
        WHEN physics_score >= topology_score
             AND physics_score >= causality_score THEN 'PHYSICS'
        WHEN topology_score >= causality_score THEN 'TOPOLOGY'
        ELSE 'CAUSALITY'
    END AS highest_risk_domain
FROM read_parquet('health.parquet')
ORDER BY health_score ASC;

-- Report: Concern Analysis
-- Most common concerns across fleet
SELECT
    primary_concern AS concern,
    COUNT(*) AS n_entities,
    AVG(health_score) AS avg_health
FROM read_parquet('health.parquet')
WHERE primary_concern != 'None'
GROUP BY primary_concern
ORDER BY n_entities DESC;

-- Report: Fleet Summary
-- Aggregate statistics across all entities
SELECT
    COUNT(*) AS total_entities,
    COUNT(CASE WHEN risk_level = 'LOW' THEN 1 END) AS healthy,
    COUNT(CASE WHEN risk_level = 'MODERATE' THEN 1 END) AS moderate,
    COUNT(CASE WHEN risk_level = 'HIGH' THEN 1 END) AS high_risk,
    COUNT(CASE WHEN risk_level = 'CRITICAL' THEN 1 END) AS critical,
    ROUND(AVG(health_score), 1) AS avg_health,
    MIN(health_score) AS min_health,
    MAX(health_score) AS max_health,
    ROUND(100.0 * COUNT(CASE WHEN risk_level = 'LOW' THEN 1 END) / COUNT(*), 1) AS pct_healthy
FROM read_parquet('health.parquet');

-- Report: Entity Ranking
-- Rank entities by health
SELECT
    unit_id,
    health_score,
    risk_level,
    RANK() OVER (ORDER BY health_score ASC) AS risk_rank,
    primary_concern
FROM read_parquet('health.parquet')
ORDER BY health_score ASC;

-- Report: Recommendation Summary
-- Group by recommended action
SELECT
    recommendation,
    COUNT(*) AS n_entities,
    AVG(health_score) AS avg_health,
    MIN(health_score) AS min_health
FROM read_parquet('health.parquet')
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
-- Average risk by domain across fleet
SELECT
    'Stability' AS domain,
    AVG(stability_score) AS avg_risk_score,
    COUNT(CASE WHEN stability_score > 0.5 THEN 1 END) AS n_high_risk
FROM read_parquet('health.parquet')
UNION ALL
SELECT
    'Predictability' AS domain,
    AVG(predictability_score) AS avg_risk_score,
    COUNT(CASE WHEN predictability_score > 0.5 THEN 1 END) AS n_high_risk
FROM read_parquet('health.parquet')
UNION ALL
SELECT
    'Physics' AS domain,
    AVG(physics_score) AS avg_risk_score,
    COUNT(CASE WHEN physics_score > 0.5 THEN 1 END) AS n_high_risk
FROM read_parquet('health.parquet')
UNION ALL
SELECT
    'Topology' AS domain,
    AVG(topology_score) AS avg_risk_score,
    COUNT(CASE WHEN topology_score > 0.5 THEN 1 END) AS n_high_risk
FROM read_parquet('health.parquet')
UNION ALL
SELECT
    'Causality' AS domain,
    AVG(causality_score) AS avg_risk_score,
    COUNT(CASE WHEN causality_score > 0.5 THEN 1 END) AS n_high_risk
FROM read_parquet('health.parquet');

-- Report: Watch List
-- Entities to monitor closely
SELECT
    unit_id,
    health_score,
    risk_level,
    primary_concern,
    recommendation
FROM read_parquet('health.parquet')
WHERE risk_level IN ('MODERATE', 'HIGH')
ORDER BY health_score ASC;
