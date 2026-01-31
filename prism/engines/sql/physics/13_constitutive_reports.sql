-- =============================================================================
-- CONSTITUTIVE REPORTS
-- Tracks physical law relationships and detects coefficient drift
-- Examples: Ohm's Law (V=IR), Hooke's Law (F=kx), Darcy's Law (dP=kQ^2)
-- =============================================================================

-- Report: Relationship Summary
-- Overview of all detected constitutive relationships
SELECT
    unit_id,
    relationship_name,
    relationship_type,
    independent_signal,
    dependent_signal,
    coefficient,
    intercept,
    r_squared,
    coefficient_drift_pct,
    trend,
    status
FROM read_parquet('physics_constitutive.parquet')
ORDER BY unit_id, relationship_name;

-- Report: Degraded Relationships
-- Relationships showing significant degradation
SELECT
    unit_id,
    relationship_name,
    coefficient,
    expected_coefficient,
    coefficient_drift_pct,
    r_squared,
    trend,
    status,
    CASE
        WHEN ABS(coefficient_drift_pct) > 50 THEN 'SEVERE_DEGRADATION'
        WHEN ABS(coefficient_drift_pct) > 30 THEN 'MODERATE_DEGRADATION'
        WHEN ABS(coefficient_drift_pct) > 20 THEN 'EARLY_DEGRADATION'
        ELSE 'ACCEPTABLE'
    END AS degradation_level
FROM read_parquet('physics_constitutive.parquet')
WHERE status IN ('DEGRADED', 'DRIFTING')
ORDER BY ABS(coefficient_drift_pct) DESC;

-- Report: Drifting Relationships
-- Relationships with statistically significant trends
SELECT
    unit_id,
    relationship_name,
    coefficient,
    coefficient_drift_pct,
    trend,
    r_squared,
    status,
    CASE
        WHEN trend = 'increasing' THEN 'Coefficient increasing over time'
        WHEN trend = 'decreasing' THEN 'Coefficient decreasing over time'
        ELSE 'No significant trend'
    END AS trend_description
FROM read_parquet('physics_constitutive.parquet')
WHERE trend IN ('increasing', 'decreasing')
ORDER BY ABS(coefficient_drift_pct) DESC;

-- Report: Weak Relationships
-- Relationships with poor R-squared (may indicate model mismatch)
SELECT
    unit_id,
    relationship_name,
    independent_signal,
    dependent_signal,
    r_squared,
    coefficient,
    status
FROM read_parquet('physics_constitutive.parquet')
WHERE r_squared < 0.7
ORDER BY r_squared ASC;

-- Report: Strong Relationships
-- Well-established relationships (high R-squared, stable)
SELECT
    unit_id,
    relationship_name,
    coefficient,
    r_squared,
    coefficient_drift_pct,
    status
FROM read_parquet('physics_constitutive.parquet')
WHERE r_squared >= 0.9 AND status = 'STABLE'
ORDER BY r_squared DESC;

-- Report: Status Summary
-- Count of relationships by status
SELECT
    status,
    COUNT(*) as n_relationships,
    AVG(r_squared) as avg_r_squared,
    AVG(ABS(coefficient_drift_pct)) as avg_drift_pct,
    COUNT(DISTINCT unit_id) as n_entities
FROM read_parquet('physics_constitutive.parquet')
GROUP BY status
ORDER BY n_relationships DESC;

-- Report: Coefficient Statistics by Relationship
-- Statistical summary for each relationship type
SELECT
    relationship_name,
    COUNT(*) as n_entities,
    AVG(coefficient) as avg_coefficient,
    STDDEV(coefficient) as std_coefficient,
    AVG(r_squared) as avg_r_squared,
    AVG(ABS(coefficient_drift_pct)) as avg_drift_pct,
    SUM(CASE WHEN status = 'DEGRADED' THEN 1 ELSE 0 END) as n_degraded,
    SUM(CASE WHEN status = 'DRIFTING' THEN 1 ELSE 0 END) as n_drifting
FROM read_parquet('physics_constitutive.parquet')
GROUP BY relationship_name
ORDER BY n_entities DESC;

-- Report: Actionable Degradation
-- Relationships requiring maintenance attention
SELECT
    unit_id,
    relationship_name,
    coefficient,
    coefficient_drift_pct,
    r_squared,
    status,
    CASE
        WHEN status = 'DEGRADED' AND relationship_name LIKE '%resistance%' THEN 'Check for corrosion or connection issues'
        WHEN status = 'DEGRADED' AND relationship_name LIKE '%stiffness%' THEN 'Inspect for fatigue or material degradation'
        WHEN status = 'DEGRADED' AND relationship_name LIKE '%flow%' THEN 'Check for fouling or blockage'
        WHEN status = 'DEGRADED' AND relationship_name LIKE '%heat%' THEN 'Inspect heat transfer surfaces'
        WHEN status = 'DRIFTING' THEN 'Monitor closely - degradation may be progressing'
        ELSE 'Investigate coefficient drift cause'
    END AS maintenance_action
FROM read_parquet('physics_constitutive.parquet')
WHERE status IN ('DEGRADED', 'DRIFTING')
ORDER BY ABS(coefficient_drift_pct) DESC;
