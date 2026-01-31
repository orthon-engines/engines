-- =============================================================================
-- MASS REPORTS
-- Conservation Law: m_dot_in = m_dot_out + dm/dt
-- =============================================================================

-- Report: Mass Balance Summary
-- Overview of mass balance across all entities
SELECT
    unit_id,
    n_samples,
    flow_in_total,
    flow_out_total,
    accumulation_rate,
    mass_balance_residual,
    mass_balance_residual_pct,
    leak_indicator,
    blockage_indicator,
    mass_status
FROM read_parquet('physics_mass.parquet')
ORDER BY ABS(mass_balance_residual_pct) DESC;

-- Report: Leak Detection
-- Entities with potential leaks (positive mass balance residual)
SELECT
    unit_id,
    flow_in_total,
    flow_out_total,
    leak_indicator,
    leak_indicator / NULLIF(flow_in_total, 0) * 100 AS leak_pct,
    mass_balance_residual_pct,
    mass_status,
    CASE
        WHEN leak_indicator / NULLIF(flow_in_total, 0) > 0.10 THEN 'SIGNIFICANT_LEAK'
        WHEN leak_indicator / NULLIF(flow_in_total, 0) > 0.05 THEN 'MODERATE_LEAK'
        WHEN leak_indicator / NULLIF(flow_in_total, 0) > 0.02 THEN 'MINOR_LEAK'
        ELSE 'NO_LEAK'
    END AS leak_severity
FROM read_parquet('physics_mass.parquet')
WHERE leak_indicator > 0
ORDER BY leak_indicator DESC;

-- Report: Blockage Detection
-- Entities with potential blockages (negative mass balance residual)
SELECT
    unit_id,
    flow_in_total,
    flow_out_total,
    blockage_indicator,
    blockage_indicator / NULLIF(flow_in_total, 0) * 100 AS blockage_pct,
    mass_balance_residual_pct,
    mass_status,
    CASE
        WHEN blockage_indicator / NULLIF(flow_in_total, 0) > 0.10 THEN 'SIGNIFICANT_BLOCKAGE'
        WHEN blockage_indicator / NULLIF(flow_in_total, 0) > 0.05 THEN 'MODERATE_BLOCKAGE'
        WHEN blockage_indicator / NULLIF(flow_in_total, 0) > 0.02 THEN 'MINOR_BLOCKAGE'
        ELSE 'NO_BLOCKAGE'
    END AS blockage_severity
FROM read_parquet('physics_mass.parquet')
WHERE blockage_indicator > 0
ORDER BY blockage_indicator DESC;

-- Report: Mass Balance Anomalies
-- All entities with mass balance issues
SELECT
    unit_id,
    flow_in_total,
    flow_out_total,
    accumulation_rate,
    mass_balance_residual,
    mass_balance_residual_pct,
    mass_status,
    CASE
        WHEN mass_status = 'POTENTIAL_LEAK' THEN 'Check for leaks in system'
        WHEN mass_status = 'POTENTIAL_BLOCKAGE' THEN 'Check for blockages or fouling'
        WHEN mass_status = 'BALANCE_WARNING' THEN 'Investigate mass balance discrepancy'
        ELSE 'No action required'
    END AS recommended_action
FROM read_parquet('physics_mass.parquet')
WHERE mass_status != 'NORMAL';

-- Report: Mass Status Summary
-- Count of entities by mass balance status
SELECT
    mass_status,
    COUNT(*) as n_entities,
    AVG(flow_in_total) as avg_flow_in,
    AVG(ABS(mass_balance_residual_pct)) as avg_residual_pct,
    SUM(leak_indicator) as total_leak_indicator,
    SUM(blockage_indicator) as total_blockage_indicator
FROM read_parquet('physics_mass.parquet')
GROUP BY mass_status
ORDER BY n_entities DESC;

-- Report: Flow Statistics
-- Statistical summary of flow measurements
SELECT
    COUNT(*) as n_entities,
    AVG(flow_in_total) as avg_flow_in,
    AVG(flow_out_total) as avg_flow_out,
    AVG(accumulation_rate) as avg_accumulation,
    STDDEV(flow_in_total) as std_flow_in,
    MIN(flow_in_total) as min_flow_in,
    MAX(flow_in_total) as max_flow_in
FROM read_parquet('physics_mass.parquet');
