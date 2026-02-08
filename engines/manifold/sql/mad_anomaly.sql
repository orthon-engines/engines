-- =============================================================================
-- MAD-Based Anomaly Detection (SQL)
-- =============================================================================
-- Robust anomaly detection using Median Absolute Deviation (MAD).
--
-- Unlike z-score which uses mean/std (sensitive to outliers), MAD uses
-- median/MAD which has a 50% breakdown point - can handle up to 50% outliers.
--
-- MAD = median(|x - median(x)|)
-- Modified Z-score = 0.6745 * (x - median) / MAD
-- (0.6745 is the consistency constant for normal distribution)
--
-- Input: observations table with (unit_id, signal_id, I, value)
-- Output: enriched observations with mad_score and is_anomaly columns
--
-- Threshold: |mad_score| > 3.5 is a common robust threshold
-- (equivalent to ~3σ for Gaussian, but robust to outliers)
-- =============================================================================

-- Step 1: Compute median per signal
WITH signal_medians AS (
    SELECT
        unit_id,
        signal_id,
        MEDIAN(value) AS median_value
    FROM observations
    GROUP BY unit_id, signal_id
),

-- Step 2: Compute absolute deviations from median
abs_deviations AS (
    SELECT
        o.unit_id,
        o.signal_id,
        o.I,
        o.value,
        m.median_value,
        ABS(o.value - m.median_value) AS abs_dev
    FROM observations o
    INNER JOIN signal_medians m
        ON o.unit_id = m.unit_id
        AND o.signal_id = m.signal_id
),

-- Step 3: Compute MAD (median of absolute deviations)
signal_mad AS (
    SELECT
        unit_id,
        signal_id,
        median_value,
        MEDIAN(abs_dev) AS mad_value
    FROM abs_deviations
    GROUP BY unit_id, signal_id, median_value
)

-- Step 4: Compute modified z-score and flag anomalies
SELECT
    o.unit_id,
    o.signal_id,
    o.I,
    o.value,
    s.median_value,
    s.mad_value,
    -- Modified z-score: 0.6745 * (x - median) / MAD
    -- 0.6745 makes it consistent with std for Gaussian
    CASE
        WHEN s.mad_value > 1e-10 THEN 0.6745 * (o.value - s.median_value) / s.mad_value
        ELSE 0
    END AS mad_score,
    -- Robust anomaly threshold: |mad_score| > 3.5
    CASE
        WHEN s.mad_value > 1e-10 AND ABS(0.6745 * (o.value - s.median_value) / s.mad_value) > 3.5 THEN TRUE
        ELSE FALSE
    END AS is_anomaly,
    -- Severity levels based on MAD score
    CASE
        WHEN s.mad_value <= 1e-10 THEN 'CONSTANT'
        WHEN ABS(0.6745 * (o.value - s.median_value) / s.mad_value) > 5.0 THEN 'CRITICAL'
        WHEN ABS(0.6745 * (o.value - s.median_value) / s.mad_value) > 3.5 THEN 'SEVERE'
        WHEN ABS(0.6745 * (o.value - s.median_value) / s.mad_value) > 2.5 THEN 'MODERATE'
        WHEN ABS(0.6745 * (o.value - s.median_value) / s.mad_value) > 2.0 THEN 'MILD'
        ELSE 'NORMAL'
    END AS severity
FROM observations o
INNER JOIN signal_mad s
    ON o.unit_id = s.unit_id
    AND o.signal_id = s.signal_id
ORDER BY o.unit_id, o.signal_id, o.I;


-- =============================================================================
-- Alternative: Rolling MAD for non-stationary data
-- =============================================================================
-- For signals that change over time, use rolling window MAD
-- Window size should match characteristic time of the signal
-- =============================================================================

-- CREATE OR REPLACE VIEW v_rolling_mad_anomaly AS
-- WITH rolling_stats AS (
--     SELECT
--         unit_id,
--         signal_id,
--         I,
--         value,
--         MEDIAN(value) OVER (
--             PARTITION BY unit_id, signal_id
--             ORDER BY I
--             ROWS BETWEEN 50 PRECEDING AND 50 FOLLOWING
--         ) AS rolling_median
--     FROM observations
-- ),
-- rolling_abs_dev AS (
--     SELECT
--         *,
--         ABS(value - rolling_median) AS abs_dev
--     FROM rolling_stats
-- ),
-- rolling_mad AS (
--     SELECT
--         *,
--         MEDIAN(abs_dev) OVER (
--             PARTITION BY unit_id, signal_id
--             ORDER BY I
--             ROWS BETWEEN 50 PRECEDING AND 50 FOLLOWING
--         ) AS rolling_mad
--     FROM rolling_abs_dev
-- )
-- SELECT
--     unit_id,
--     signal_id,
--     I,
--     value,
--     rolling_median,
--     rolling_mad,
--     CASE
--         WHEN rolling_mad > 1e-10 THEN 0.6745 * (value - rolling_median) / rolling_mad
--         ELSE 0
--     END AS mad_score,
--     CASE
--         WHEN rolling_mad > 1e-10 AND ABS(0.6745 * (value - rolling_median) / rolling_mad) > 3.5 THEN TRUE
--         ELSE FALSE
--     END AS is_anomaly
-- FROM rolling_mad;
