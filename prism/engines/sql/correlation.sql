-- =============================================================================
-- Correlation Engine (SQL)
-- =============================================================================
-- Computes pairwise Pearson correlation between all signal pairs per entity.
-- Input: observations table with (unit_id, signal_id, I, value)
-- Output: correlation matrix as (unit_id, signal_a, signal_b, correlation)
-- =============================================================================

WITH signal_pairs AS (
    SELECT DISTINCT
        a.unit_id,
        a.signal_id AS signal_a,
        b.signal_id AS signal_b
    FROM observations a
    INNER JOIN observations b
        ON a.unit_id = b.unit_id
        AND a.signal_id < b.signal_id
),
aligned AS (
    SELECT
        sp.unit_id,
        sp.signal_a,
        sp.signal_b,
        a.I,
        a.value AS value_a,
        b.value AS value_b
    FROM signal_pairs sp
    INNER JOIN observations a
        ON sp.unit_id = a.unit_id
        AND sp.signal_a = a.signal_id
    INNER JOIN observations b
        ON sp.unit_id = b.unit_id
        AND sp.signal_b = b.signal_id
        AND a.I = b.I
)
SELECT
    unit_id,
    signal_a,
    signal_b,
    CORR(value_a, value_b) AS correlation,
    COUNT(*) AS n_points
FROM aligned
GROUP BY unit_id, signal_a, signal_b
HAVING COUNT(*) >= 10
ORDER BY unit_id, signal_a, signal_b;
