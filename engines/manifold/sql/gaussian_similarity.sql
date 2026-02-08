-- =============================================================================
-- Gaussian Similarity Engine (SQL)
-- =============================================================================
-- Pairwise Bhattacharyya-inspired distance between Gaussian fingerprints.
-- Compares all (signal_a, signal_b) pairs within the same cohort.
--
-- Distance = sum of per-feature Bhattacharyya distances, where:
--   DB_k = (1/4) * (mu_a - mu_b)^2 / (s_a^2 + s_b^2) + (1/2) * ln((s_a^2 + s_b^2) / (2 * s_a * s_b))
--
-- Input: gaussian_fingerprint table (from gaussian_fingerprint.sql)
-- Output: pairwise similarity matrix per cohort
-- =============================================================================

WITH feature_distance AS (
    SELECT
        a.cohort,
        a.signal_id AS signal_a,
        b.signal_id AS signal_b,

        -- Per-feature Bhattacharyya distance components
        -- spectral_entropy
        CASE WHEN a.std_spectral_entropy > 1e-10 AND b.std_spectral_entropy > 1e-10 THEN
            0.25 * POWER(a.mean_spectral_entropy - b.mean_spectral_entropy, 2)
                / (POWER(a.std_spectral_entropy, 2) + POWER(b.std_spectral_entropy, 2))
            + 0.5 * LN((POWER(a.std_spectral_entropy, 2) + POWER(b.std_spectral_entropy, 2))
                / (2.0 * a.std_spectral_entropy * b.std_spectral_entropy))
        ELSE NULL END AS db_spectral_entropy,

        -- hurst
        CASE WHEN a.std_hurst > 1e-10 AND b.std_hurst > 1e-10 THEN
            0.25 * POWER(a.mean_hurst - b.mean_hurst, 2)
                / (POWER(a.std_hurst, 2) + POWER(b.std_hurst, 2))
            + 0.5 * LN((POWER(a.std_hurst, 2) + POWER(b.std_hurst, 2))
                / (2.0 * a.std_hurst * b.std_hurst))
        ELSE NULL END AS db_hurst,

        -- sample_entropy
        CASE WHEN a.std_sample_entropy > 1e-10 AND b.std_sample_entropy > 1e-10 THEN
            0.25 * POWER(a.mean_sample_entropy - b.mean_sample_entropy, 2)
                / (POWER(a.std_sample_entropy, 2) + POWER(b.std_sample_entropy, 2))
            + 0.5 * LN((POWER(a.std_sample_entropy, 2) + POWER(b.std_sample_entropy, 2))
                / (2.0 * a.std_sample_entropy * b.std_sample_entropy))
        ELSE NULL END AS db_sample_entropy,

        -- dfa
        CASE WHEN a.std_dfa > 1e-10 AND b.std_dfa > 1e-10 THEN
            0.25 * POWER(a.mean_dfa - b.mean_dfa, 2)
                / (POWER(a.std_dfa, 2) + POWER(b.std_dfa, 2))
            + 0.5 * LN((POWER(a.std_dfa, 2) + POWER(b.std_dfa, 2))
                / (2.0 * a.std_dfa * b.std_dfa))
        ELSE NULL END AS db_dfa,

        -- acf_decay
        CASE WHEN a.std_acf_decay > 1e-10 AND b.std_acf_decay > 1e-10 THEN
            0.25 * POWER(a.mean_acf_decay - b.mean_acf_decay, 2)
                / (POWER(a.std_acf_decay, 2) + POWER(b.std_acf_decay, 2))
            + 0.5 * LN((POWER(a.std_acf_decay, 2) + POWER(b.std_acf_decay, 2))
                / (2.0 * a.std_acf_decay * b.std_acf_decay))
        ELSE NULL END AS db_acf_decay,

        -- kurtosis
        CASE WHEN a.std_kurtosis > 1e-10 AND b.std_kurtosis > 1e-10 THEN
            0.25 * POWER(a.mean_kurtosis - b.mean_kurtosis, 2)
                / (POWER(a.std_kurtosis, 2) + POWER(b.std_kurtosis, 2))
            + 0.5 * LN((POWER(a.std_kurtosis, 2) + POWER(b.std_kurtosis, 2))
                / (2.0 * a.std_kurtosis * b.std_kurtosis))
        ELSE NULL END AS db_kurtosis,

        -- skewness
        CASE WHEN a.std_skewness > 1e-10 AND b.std_skewness > 1e-10 THEN
            0.25 * POWER(a.mean_skewness - b.mean_skewness, 2)
                / (POWER(a.std_skewness, 2) + POWER(b.std_skewness, 2))
            + 0.5 * LN((POWER(a.std_skewness, 2) + POWER(b.std_skewness, 2))
                / (2.0 * a.std_skewness * b.std_skewness))
        ELSE NULL END AS db_skewness,

        -- trend_r2
        CASE WHEN a.std_trend_r2 > 1e-10 AND b.std_trend_r2 > 1e-10 THEN
            0.25 * POWER(a.mean_trend_r2 - b.mean_trend_r2, 2)
                / (POWER(a.std_trend_r2, 2) + POWER(b.std_trend_r2, 2))
            + 0.5 * LN((POWER(a.std_trend_r2, 2) + POWER(b.std_trend_r2, 2))
                / (2.0 * a.std_trend_r2 * b.std_trend_r2))
        ELSE NULL END AS db_trend_r2,

        -- recurrence_rate
        CASE WHEN a.std_recurrence_rate > 1e-10 AND b.std_recurrence_rate > 1e-10 THEN
            0.25 * POWER(a.mean_recurrence_rate - b.mean_recurrence_rate, 2)
                / (POWER(a.std_recurrence_rate, 2) + POWER(b.std_recurrence_rate, 2))
            + 0.5 * LN((POWER(a.std_recurrence_rate, 2) + POWER(b.std_recurrence_rate, 2))
                / (2.0 * a.std_recurrence_rate * b.std_recurrence_rate))
        ELSE NULL END AS db_recurrence_rate,

        -- determinism
        CASE WHEN a.std_determinism > 1e-10 AND b.std_determinism > 1e-10 THEN
            0.25 * POWER(a.mean_determinism - b.mean_determinism, 2)
                / (POWER(a.std_determinism, 2) + POWER(b.std_determinism, 2))
            + 0.5 * LN((POWER(a.std_determinism, 2) + POWER(b.std_determinism, 2))
                / (2.0 * a.std_determinism * b.std_determinism))
        ELSE NULL END AS db_determinism,

        -- Fingerprint volatility difference
        ABS(a.fingerprint_volatility - b.fingerprint_volatility) AS volatility_diff

    FROM gaussian_fingerprint a
    JOIN gaussian_fingerprint b
        ON a.cohort = b.cohort
        AND a.signal_id < b.signal_id
)

SELECT
    cohort,
    signal_a,
    signal_b,

    -- Total Bhattacharyya distance (sum of per-feature distances)
    (
        COALESCE(db_spectral_entropy, 0) +
        COALESCE(db_hurst, 0) +
        COALESCE(db_sample_entropy, 0) +
        COALESCE(db_dfa, 0) +
        COALESCE(db_acf_decay, 0) +
        COALESCE(db_kurtosis, 0) +
        COALESCE(db_skewness, 0) +
        COALESCE(db_trend_r2, 0) +
        COALESCE(db_recurrence_rate, 0) +
        COALESCE(db_determinism, 0)
    ) AS bhattacharyya_distance,

    -- Number of features contributing to distance
    (
        CASE WHEN db_spectral_entropy IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN db_hurst IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN db_sample_entropy IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN db_dfa IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN db_acf_decay IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN db_kurtosis IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN db_skewness IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN db_trend_r2 IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN db_recurrence_rate IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN db_determinism IS NOT NULL THEN 1 ELSE 0 END
    ) AS n_features,

    -- Normalized distance (per-feature average)
    CASE WHEN (
        CASE WHEN db_spectral_entropy IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN db_hurst IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN db_sample_entropy IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN db_dfa IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN db_acf_decay IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN db_kurtosis IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN db_skewness IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN db_trend_r2 IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN db_recurrence_rate IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN db_determinism IS NOT NULL THEN 1 ELSE 0 END
    ) > 0 THEN
        (
            COALESCE(db_spectral_entropy, 0) +
            COALESCE(db_hurst, 0) +
            COALESCE(db_sample_entropy, 0) +
            COALESCE(db_dfa, 0) +
            COALESCE(db_acf_decay, 0) +
            COALESCE(db_kurtosis, 0) +
            COALESCE(db_skewness, 0) +
            COALESCE(db_trend_r2, 0) +
            COALESCE(db_recurrence_rate, 0) +
            COALESCE(db_determinism, 0)
        ) / NULLIF(
            CASE WHEN db_spectral_entropy IS NOT NULL THEN 1 ELSE 0 END +
            CASE WHEN db_hurst IS NOT NULL THEN 1 ELSE 0 END +
            CASE WHEN db_sample_entropy IS NOT NULL THEN 1 ELSE 0 END +
            CASE WHEN db_dfa IS NOT NULL THEN 1 ELSE 0 END +
            CASE WHEN db_acf_decay IS NOT NULL THEN 1 ELSE 0 END +
            CASE WHEN db_kurtosis IS NOT NULL THEN 1 ELSE 0 END +
            CASE WHEN db_skewness IS NOT NULL THEN 1 ELSE 0 END +
            CASE WHEN db_trend_r2 IS NOT NULL THEN 1 ELSE 0 END +
            CASE WHEN db_recurrence_rate IS NOT NULL THEN 1 ELSE 0 END +
            CASE WHEN db_determinism IS NOT NULL THEN 1 ELSE 0 END
        , 0)
    ELSE NULL END AS normalized_distance,

    -- Similarity score: exp(-distance) maps to (0, 1]
    EXP(-(
        COALESCE(db_spectral_entropy, 0) +
        COALESCE(db_hurst, 0) +
        COALESCE(db_sample_entropy, 0) +
        COALESCE(db_dfa, 0) +
        COALESCE(db_acf_decay, 0) +
        COALESCE(db_kurtosis, 0) +
        COALESCE(db_skewness, 0) +
        COALESCE(db_trend_r2, 0) +
        COALESCE(db_recurrence_rate, 0) +
        COALESCE(db_determinism, 0)
    )) AS similarity,

    volatility_diff,

    -- Per-feature distances (for diagnostics)
    db_spectral_entropy,
    db_hurst,
    db_sample_entropy,
    db_dfa,
    db_acf_decay,
    db_kurtosis,
    db_skewness,
    db_trend_r2,
    db_recurrence_rate,
    db_determinism

FROM feature_distance
ORDER BY cohort, bhattacharyya_distance;
