-- =============================================================================
-- Statistics Engine (SQL)
-- =============================================================================
-- Computes basic statistics for each signal.
-- Input: observations table with (cohort, signal_id, I, value)
-- Note: cohort is optional - uses COALESCE to handle NULL
-- Output: signal-level statistics
-- =============================================================================

SELECT
    COALESCE(cohort, '_default') AS cohort,
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
GROUP BY COALESCE(cohort, '_default'), signal_id
ORDER BY cohort, signal_id;
