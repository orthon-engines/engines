-- =============================================================================
-- Correlation Engine (SQL)
-- =============================================================================
-- Computes pairwise Pearson correlation between all signal pairs per entity.
-- Input: observations table with (cohort, signal_id, I, value)
-- Note: cohort is optional - uses COALESCE to handle NULL
-- Output: correlation matrix as (cohort, signal_a, signal_b, correlation)
-- =============================================================================

WITH signal_pairs AS (
    SELECT DISTINCT
        COALESCE(a.cohort, '_default') AS cohort,
        a.signal_id AS signal_a,
        b.signal_id AS signal_b
    FROM observations a
    INNER JOIN observations b
        ON COALESCE(a.cohort, '_default') = COALESCE(b.cohort, '_default')
        AND a.signal_id < b.signal_id
),
aligned AS (
    SELECT
        sp.cohort,
        sp.signal_a,
        sp.signal_b,
        a.I,
        a.value AS value_a,
        b.value AS value_b
    FROM signal_pairs sp
    INNER JOIN observations a
        ON sp.cohort = COALESCE(a.cohort, '_default')
        AND sp.signal_a = a.signal_id
    INNER JOIN observations b
        ON sp.cohort = COALESCE(b.cohort, '_default')
        AND sp.signal_b = b.signal_id
        AND a.I = b.I
)
SELECT
    cohort,
    signal_a,
    signal_b,
    CORR(value_a, value_b) AS correlation,
    COUNT(*) AS n_points
FROM aligned
GROUP BY cohort, signal_a, signal_b
HAVING COUNT(*) >= 10
ORDER BY cohort, signal_a, signal_b;
