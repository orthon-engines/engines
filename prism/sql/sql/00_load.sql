-- ============================================================================
-- ORTHON SQL ENGINES: LOAD & VALIDATE
-- ============================================================================
-- Load observations parquet, create base view, validate schema.
-- This is the foundation for all downstream views.
-- ============================================================================

-- ============================================================================
-- CANONICAL SCHEMA - THE RULE
-- ============================================================================
-- observations.parquet MUST have columns:
--
--   entity_id  : str   - Entity identifier
--   signal_id  : str   - Signal name
--   I          : float - Index (time, cycle, depth, distance, sample)
--   y          : float - Value (the measurement)
--   unit       : str   - Unit of measurement (optional)
--
-- I means I. y means y. No aliases. No mapping.
-- Column mapping happens at INTAKE, not here.
-- ============================================================================

-- ============================================================================
-- 001: CREATE BASE VIEW FROM OBSERVATIONS
-- ============================================================================

CREATE OR REPLACE VIEW v_base AS
SELECT
    entity_id,
    signal_id,
    I,
    y,
    -- Unit column is optional - use NULL if not present
    NULL AS value_unit,
    NULL AS index_dimension,
    NULL AS signal_class
FROM observations;


-- ============================================================================
-- 002: SCHEMA VALIDATION
-- ============================================================================
-- Verify required columns exist and have data

CREATE OR REPLACE VIEW v_schema_validation AS
SELECT
    COUNT(*) AS n_rows,
    COUNT(DISTINCT entity_id) AS n_entities,
    COUNT(DISTINCT signal_id) AS n_signals,
    MIN(I) AS I_min,
    MAX(I) AS I_max,
    CASE
        WHEN COUNT(*) = 0 THEN 'ERROR: No data'
        WHEN COUNT(DISTINCT signal_id) = 0 THEN 'ERROR: No signals'
        ELSE 'OK'
    END AS validation_status
FROM v_base;


-- ============================================================================
-- 003: SIGNAL INVENTORY
-- ============================================================================
-- One row per signal with basic stats

CREATE OR REPLACE VIEW v_signal_inventory AS
SELECT
    entity_id,
    signal_id,
    COUNT(*) AS n_points,
    MIN(I) AS I_min,
    MAX(I) AS I_max,
    MIN(y) AS y_min,
    MAX(y) AS y_max,
    AVG(y) AS y_mean,
    value_unit,
    index_dimension
FROM v_base
GROUP BY entity_id, signal_id, value_unit, index_dimension;


-- ============================================================================
-- 004: DATA QUALITY FLAGS
-- ============================================================================

CREATE OR REPLACE VIEW v_data_quality AS
SELECT
    signal_id,
    entity_id,
    n_points,
    CASE WHEN n_points < 50 THEN TRUE ELSE FALSE END AS insufficient_data,
    CASE WHEN y_min = y_max THEN TRUE ELSE FALSE END AS constant_signal,
    CASE WHEN y_min IS NULL OR y_max IS NULL THEN TRUE ELSE FALSE END AS has_nulls
FROM v_signal_inventory;
