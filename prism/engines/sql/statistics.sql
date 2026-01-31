-- =============================================================================
-- Statistics Engine (SQL)
-- =============================================================================
-- Computes basic statistics for each signal.
-- Input: observations table with (unit_id, signal_id, I, value)
-- Output: signal-level statistics
-- =============================================================================

SELECT
    unit_id,
    signal_id,
    COUNT(*) AS n_points,
    AVG(value) AS mean,
    STDDEV_SAMP(value) AS std,
    MIN(value) AS min,
    MAX(value) AS max,
    MAX(value) - MIN(value) AS range,
    MEDIAN(value) AS median,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY value) AS q1,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY value) AS q3,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY value) - PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY value) AS iqr,
    VARIANCE(value) AS variance,
    STDDEV_SAMP(value) / NULLIF(ABS(AVG(value)), 0) AS cv
FROM observations
GROUP BY unit_id, signal_id
ORDER BY unit_id, signal_id;
