-- =============================================================================
-- Z-Score Engine (SQL)
-- =============================================================================
-- Computes z-score for each observation within its signal.
-- Input: observations table with (unit_id, signal_id, I, value)
-- Output: enriched observations with z_score and is_anomaly columns
-- =============================================================================

WITH signal_stats AS (
    SELECT
        unit_id,
        signal_id,
        AVG(value) AS mean_value,
        STDDEV_SAMP(value) AS std_value
    FROM observations
    GROUP BY unit_id, signal_id
)
SELECT
    o.unit_id,
    o.signal_id,
    o.I,
    o.value,
    CASE
        WHEN s.std_value > 1e-10 THEN (o.value - s.mean_value) / s.std_value
        ELSE 0
    END AS z_score,
    CASE
        WHEN s.std_value > 1e-10 AND ABS((o.value - s.mean_value) / s.std_value) > 3 THEN TRUE
        ELSE FALSE
    END AS is_anomaly
FROM observations o
INNER JOIN signal_stats s
    ON o.unit_id = s.unit_id
    AND o.signal_id = s.signal_id
ORDER BY o.unit_id, o.signal_id, o.I;
