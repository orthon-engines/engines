-- =============================================================================
-- ATTRACTOR REPORTS
-- Attractor dimension and structure analysis
-- =============================================================================

-- Report: Attractor Summary
-- Overview of attractor characteristics across all entities
SELECT
    unit_id,
    n_samples,
    correlation_dimension,
    effective_dimension,
    n_significant_modes,
    eigenvalue_ratio,
    dimension_collapse,
    attractor_change,
    CASE
        WHEN dimension_collapse THEN 'COLLAPSING'
        WHEN attractor_change > 0.3 THEN 'CHANGING'
        ELSE 'STABLE'
    END AS attractor_status
FROM read_parquet('dynamics_attractor.parquet')
ORDER BY attractor_change DESC;

-- Report: Dimension Collapse Alerts
-- Entities experiencing dimension collapse (often precedes failure)
SELECT
    unit_id,
    effective_dimension,
    dimension_change,
    attractor_change,
    n_significant_modes,
    'DIMENSION_COLLAPSE' AS alert_type,
    'High priority - dimension collapse often precedes system failure' AS recommendation
FROM read_parquet('dynamics_attractor.parquet')
WHERE dimension_collapse = TRUE
ORDER BY attractor_change DESC;

-- Report: Significant Mode Analysis
-- Analysis of significant modes per entity
SELECT
    unit_id,
    n_significant_modes,
    effective_dimension,
    mp_upper_bound,
    eigenvalue_1,
    eigenvalue_2,
    eigenvalue_ratio,
    CASE
        WHEN n_significant_modes = 1 THEN 'SINGLE_MODE'
        WHEN n_significant_modes = 2 THEN 'TWO_MODE'
        WHEN n_significant_modes <= 5 THEN 'LOW_DIMENSIONAL'
        ELSE 'HIGH_DIMENSIONAL'
    END AS dimensionality_class
FROM read_parquet('dynamics_attractor.parquet')
WHERE n_significant_modes > 0
ORDER BY n_significant_modes;

-- Report: Attractor Change Tracking
-- Entities with significant attractor changes
SELECT
    unit_id,
    attractor_change,
    effective_dimension,
    dimension_change,
    correlation_dimension,
    CASE
        WHEN attractor_change > 0.5 THEN 'MAJOR_CHANGE'
        WHEN attractor_change > 0.3 THEN 'MODERATE_CHANGE'
        WHEN attractor_change > 0.1 THEN 'MINOR_CHANGE'
        ELSE 'STABLE'
    END AS change_severity
FROM read_parquet('dynamics_attractor.parquet')
WHERE attractor_change > 0.1
ORDER BY attractor_change DESC;

-- Report: Correlation Dimension Statistics
-- Summary of fractal dimensions
SELECT
    COUNT(*) as n_entities,
    AVG(correlation_dimension) as avg_corr_dim,
    STDDEV(correlation_dimension) as std_corr_dim,
    MIN(correlation_dimension) as min_corr_dim,
    MAX(correlation_dimension) as max_corr_dim,
    AVG(effective_dimension) as avg_eff_dim
FROM read_parquet('dynamics_attractor.parquet')
WHERE correlation_dimension IS NOT NULL;

-- Report: Eigenvalue Analysis
-- Entities with unusual eigenvalue structure
SELECT
    unit_id,
    eigenvalue_1,
    eigenvalue_2,
    eigenvalue_ratio,
    effective_dimension,
    CASE
        WHEN eigenvalue_ratio > 10 THEN 'HIGHLY_DOMINANT'
        WHEN eigenvalue_ratio > 5 THEN 'DOMINANT'
        WHEN eigenvalue_ratio > 2 THEN 'MODERATE'
        ELSE 'DISTRIBUTED'
    END AS dominance_class
FROM read_parquet('dynamics_attractor.parquet')
WHERE eigenvalue_ratio IS NOT NULL
ORDER BY eigenvalue_ratio DESC;
