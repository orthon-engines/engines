-- =============================================================================
-- ENERGY REPORTS
-- Conservation Law: Ein = Eout + Estored + Edissipated
-- =============================================================================

-- Report: Energy Balance Summary
-- Overview of energy balance across all entities
SELECT
    unit_id,
    n_samples,
    power_in_total,
    power_out_total,
    power_dissipated,
    efficiency,
    dissipation_pct,
    balance_residual_pct,
    energy_status,
    CASE
        WHEN ABS(balance_residual_pct) > 10 THEN 'CRITICAL'
        WHEN ABS(balance_residual_pct) > 5 THEN 'WARNING'
        ELSE 'OK'
    END AS balance_severity
FROM read_parquet('physics_energy.parquet')
ORDER BY ABS(balance_residual_pct) DESC;

-- Report: Energy Anomalies
-- Entities with significant energy balance violations
SELECT
    unit_id,
    power_in_total,
    power_out_total,
    balance_residual,
    balance_residual_pct,
    energy_status,
    CASE
        WHEN ABS(balance_residual_pct) > 20 THEN 'SEVERE_IMBALANCE'
        WHEN ABS(balance_residual_pct) > 10 THEN 'MODERATE_IMBALANCE'
        WHEN ABS(balance_residual_pct) > 5 THEN 'MINOR_IMBALANCE'
        ELSE 'ACCEPTABLE'
    END AS anomaly_level
FROM read_parquet('physics_energy.parquet')
WHERE ABS(balance_residual_pct) > 5
ORDER BY ABS(balance_residual_pct) DESC;

-- Report: Efficiency Distribution
-- Statistical summary of efficiency across entities
SELECT
    COUNT(*) as n_entities,
    AVG(efficiency) as avg_efficiency,
    STDDEV(efficiency) as std_efficiency,
    MIN(efficiency) as min_efficiency,
    MAX(efficiency) as max_efficiency,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY efficiency) as p25_efficiency,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY efficiency) as median_efficiency,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY efficiency) as p75_efficiency
FROM read_parquet('physics_energy.parquet')
WHERE efficiency IS NOT NULL;

-- Report: Low Efficiency Entities
-- Entities with concerning efficiency levels
SELECT
    unit_id,
    efficiency,
    power_in_total,
    power_out_total,
    power_dissipated,
    dissipation_pct,
    energy_status
FROM read_parquet('physics_energy.parquet')
WHERE efficiency < 0.7 AND power_in_total > 0
ORDER BY efficiency ASC;

-- Report: High Dissipation Entities
-- Entities losing significant energy to dissipation
SELECT
    unit_id,
    dissipation_pct,
    power_dissipated,
    power_in_total,
    efficiency,
    energy_status
FROM read_parquet('physics_energy.parquet')
WHERE dissipation_pct > 15
ORDER BY dissipation_pct DESC;

-- Report: Energy Status Summary
-- Count of entities by energy status
SELECT
    energy_status,
    COUNT(*) as n_entities,
    AVG(efficiency) as avg_efficiency,
    AVG(ABS(balance_residual_pct)) as avg_residual_pct
FROM read_parquet('physics_energy.parquet')
GROUP BY energy_status
ORDER BY n_entities DESC;
