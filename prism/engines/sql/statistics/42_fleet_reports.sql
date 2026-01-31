-- =============================================================================
-- FLEET REPORTS
-- Fleet-wide analytics, rankings, and cohorts
-- =============================================================================

-- Report: Entity Rankings
-- Full rankings with all metrics
SELECT
    unit_id,
    health_rank,
    avg_health,
    min_health,
    max_health,
    latest_health,
    health_volatility,
    critical_events,
    high_events,
    cluster,
    health_tier,
    total_anomalies,
    critical_anomalies,
    warning_anomalies
FROM read_parquet('fleet_rankings.parquet')
ORDER BY health_rank;

-- Report: Top Performers
-- Best performing entities
SELECT
    unit_id,
    avg_health,
    health_tier,
    critical_events,
    'TOP_PERFORMER' AS status
FROM read_parquet('fleet_rankings.parquet')
ORDER BY avg_health DESC
LIMIT 10;

-- Report: Bottom Performers
-- Worst performing entities requiring attention
SELECT
    unit_id,
    avg_health,
    health_tier,
    critical_events,
    high_events,
    total_anomalies,
    'NEEDS_ATTENTION' AS status
FROM read_parquet('fleet_rankings.parquet')
ORDER BY avg_health ASC
LIMIT 10;

-- Report: Cluster Summary
-- Statistics by cluster
SELECT
    cluster,
    COUNT(*) AS n_entities,
    AVG(avg_health) AS cluster_avg_health,
    MIN(avg_health) AS cluster_min_health,
    MAX(avg_health) AS cluster_max_health,
    SUM(critical_events) AS total_critical,
    AVG(health_volatility) AS avg_volatility
FROM read_parquet('fleet_rankings.parquet')
GROUP BY cluster
ORDER BY cluster_avg_health DESC;

-- Report: Health Tier Distribution
-- Count of entities by health tier
SELECT
    health_tier,
    COUNT(*) AS n_entities,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_fleet,
    AVG(avg_health) AS tier_avg_health,
    SUM(critical_events) AS tier_total_critical
FROM read_parquet('fleet_rankings.parquet')
GROUP BY health_tier
ORDER BY
    CASE health_tier
        WHEN 'HEALTHY' THEN 1
        WHEN 'MODERATE' THEN 2
        WHEN 'AT_RISK' THEN 3
        WHEN 'CRITICAL' THEN 4
    END;

-- Report: Volatility Analysis
-- Entities with high health volatility
SELECT
    unit_id,
    avg_health,
    health_volatility,
    min_health,
    max_health,
    max_health - min_health AS health_range,
    CASE
        WHEN health_volatility > 15 THEN 'HIGH_VOLATILITY'
        WHEN health_volatility > 10 THEN 'MODERATE_VOLATILITY'
        ELSE 'STABLE'
    END AS volatility_status
FROM read_parquet('fleet_rankings.parquet')
WHERE health_volatility IS NOT NULL
ORDER BY health_volatility DESC;

-- Report: Risk Event Analysis
-- Entities ranked by total risk events
SELECT
    unit_id,
    critical_events,
    high_events,
    critical_events + high_events AS total_risk_events,
    avg_health,
    health_tier
FROM read_parquet('fleet_rankings.parquet')
WHERE critical_events > 0 OR high_events > 0
ORDER BY total_risk_events DESC;

-- Report: Fleet Summary Statistics
-- Overall fleet statistics
SELECT
    COUNT(*) AS total_entities,
    AVG(avg_health) AS fleet_avg_health,
    MIN(avg_health) AS fleet_min_health,
    MAX(avg_health) AS fleet_max_health,
    AVG(health_volatility) AS avg_volatility,
    SUM(critical_events) AS total_critical_events,
    SUM(high_events) AS total_high_events,
    COUNT(CASE WHEN health_tier = 'HEALTHY' THEN 1 END) AS healthy_count,
    COUNT(CASE WHEN health_tier = 'CRITICAL' THEN 1 END) AS critical_count
FROM read_parquet('fleet_rankings.parquet');

-- Report: Cluster Comparison
-- Compare metrics across clusters
SELECT
    cluster,
    COUNT(*) AS n,
    ROUND(AVG(avg_health), 1) AS avg_health,
    ROUND(AVG(health_volatility), 2) AS volatility,
    SUM(critical_events) AS criticals,
    SUM(total_anomalies) AS anomalies
FROM read_parquet('fleet_rankings.parquet')
GROUP BY cluster
ORDER BY avg_health DESC;
