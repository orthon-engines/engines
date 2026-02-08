-- =============================================================================
-- Gaussian Fingerprint Engine (SQL)
-- =============================================================================
-- Builds probabilistic fingerprints from windowed engine outputs in
-- signal_vector.parquet. Each signal gets a Gaussian summary:
--   mean vector, std vector, key cross-correlations, fingerprint volatility.
--
-- One row per (signal_id, cohort).
--
-- Input: signal_vector table with (signal_id, cohort, I, <engine_columns>)
-- Output: Gaussian fingerprint per signal
-- =============================================================================

WITH numeric_cols AS (
    -- Identify numeric engine output columns (exclude metadata)
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = 'signal_vector'
      AND data_type IN ('DOUBLE', 'FLOAT', 'BIGINT', 'INTEGER', 'SMALLINT', 'TINYINT', 'HUGEINT')
      AND column_name NOT IN ('I', 'signal_id', 'cohort')
),

-- Per-signal statistics across all windows
signal_stats AS (
    SELECT
        COALESCE(cohort, '_default') AS cohort,
        signal_id,
        COUNT(*) AS n_windows,

        -- Mean vector (representative center)
        AVG(CASE WHEN (SELECT COUNT(*) FROM numeric_cols WHERE column_name = 'spectral_entropy') > 0
            THEN spectral_entropy ELSE NULL END) AS mean_spectral_entropy,
        AVG(CASE WHEN (SELECT COUNT(*) FROM numeric_cols WHERE column_name = 'hurst') > 0
            THEN hurst ELSE NULL END) AS mean_hurst,
        AVG(CASE WHEN (SELECT COUNT(*) FROM numeric_cols WHERE column_name = 'sample_entropy') > 0
            THEN sample_entropy ELSE NULL END) AS mean_sample_entropy,
        AVG(CASE WHEN (SELECT COUNT(*) FROM numeric_cols WHERE column_name = 'dfa') > 0
            THEN dfa ELSE NULL END) AS mean_dfa,
        AVG(CASE WHEN (SELECT COUNT(*) FROM numeric_cols WHERE column_name = 'acf_decay') > 0
            THEN acf_decay ELSE NULL END) AS mean_acf_decay,
        AVG(CASE WHEN (SELECT COUNT(*) FROM numeric_cols WHERE column_name = 'kurtosis') > 0
            THEN kurtosis ELSE NULL END) AS mean_kurtosis,
        AVG(CASE WHEN (SELECT COUNT(*) FROM numeric_cols WHERE column_name = 'skewness') > 0
            THEN skewness ELSE NULL END) AS mean_skewness,
        AVG(CASE WHEN (SELECT COUNT(*) FROM numeric_cols WHERE column_name = 'trend_r2') > 0
            THEN trend_r2 ELSE NULL END) AS mean_trend_r2,
        AVG(CASE WHEN (SELECT COUNT(*) FROM numeric_cols WHERE column_name = 'recurrence_rate') > 0
            THEN recurrence_rate ELSE NULL END) AS mean_recurrence_rate,
        AVG(CASE WHEN (SELECT COUNT(*) FROM numeric_cols WHERE column_name = 'determinism') > 0
            THEN determinism ELSE NULL END) AS mean_determinism,

        -- Std vector (variability of each feature across windows)
        STDDEV_SAMP(CASE WHEN (SELECT COUNT(*) FROM numeric_cols WHERE column_name = 'spectral_entropy') > 0
            THEN spectral_entropy ELSE NULL END) AS std_spectral_entropy,
        STDDEV_SAMP(CASE WHEN (SELECT COUNT(*) FROM numeric_cols WHERE column_name = 'hurst') > 0
            THEN hurst ELSE NULL END) AS std_hurst,
        STDDEV_SAMP(CASE WHEN (SELECT COUNT(*) FROM numeric_cols WHERE column_name = 'sample_entropy') > 0
            THEN sample_entropy ELSE NULL END) AS std_sample_entropy,
        STDDEV_SAMP(CASE WHEN (SELECT COUNT(*) FROM numeric_cols WHERE column_name = 'dfa') > 0
            THEN dfa ELSE NULL END) AS std_dfa,
        STDDEV_SAMP(CASE WHEN (SELECT COUNT(*) FROM numeric_cols WHERE column_name = 'acf_decay') > 0
            THEN acf_decay ELSE NULL END) AS std_acf_decay,
        STDDEV_SAMP(CASE WHEN (SELECT COUNT(*) FROM numeric_cols WHERE column_name = 'kurtosis') > 0
            THEN kurtosis ELSE NULL END) AS std_kurtosis,
        STDDEV_SAMP(CASE WHEN (SELECT COUNT(*) FROM numeric_cols WHERE column_name = 'skewness') > 0
            THEN skewness ELSE NULL END) AS std_skewness,
        STDDEV_SAMP(CASE WHEN (SELECT COUNT(*) FROM numeric_cols WHERE column_name = 'trend_r2') > 0
            THEN trend_r2 ELSE NULL END) AS std_trend_r2,
        STDDEV_SAMP(CASE WHEN (SELECT COUNT(*) FROM numeric_cols WHERE column_name = 'recurrence_rate') > 0
            THEN recurrence_rate ELSE NULL END) AS std_recurrence_rate,
        STDDEV_SAMP(CASE WHEN (SELECT COUNT(*) FROM numeric_cols WHERE column_name = 'determinism') > 0
            THEN determinism ELSE NULL END) AS std_determinism,

        -- Key cross-correlations (within-signal, across features)
        CORR(spectral_entropy, hurst) AS corr_entropy_hurst,
        CORR(spectral_entropy, sample_entropy) AS corr_spectral_sample_entropy,
        CORR(hurst, dfa) AS corr_hurst_dfa,
        CORR(kurtosis, skewness) AS corr_kurtosis_skewness,

    FROM signal_vector
    GROUP BY COALESCE(cohort, '_default'), signal_id
)

SELECT
    cohort,
    signal_id,
    n_windows,

    -- Mean vector
    mean_spectral_entropy,
    mean_hurst,
    mean_sample_entropy,
    mean_dfa,
    mean_acf_decay,
    mean_kurtosis,
    mean_skewness,
    mean_trend_r2,
    mean_recurrence_rate,
    mean_determinism,

    -- Std vector
    std_spectral_entropy,
    std_hurst,
    std_sample_entropy,
    std_dfa,
    std_acf_decay,
    std_kurtosis,
    std_skewness,
    std_trend_r2,
    std_recurrence_rate,
    std_determinism,

    -- Cross-correlations
    corr_entropy_hurst,
    corr_spectral_sample_entropy,
    corr_hurst_dfa,
    corr_kurtosis_skewness,

    -- Fingerprint volatility: average of std vector (how much does this signal's profile change?)
    (
        COALESCE(std_spectral_entropy, 0) +
        COALESCE(std_hurst, 0) +
        COALESCE(std_sample_entropy, 0) +
        COALESCE(std_dfa, 0) +
        COALESCE(std_acf_decay, 0) +
        COALESCE(std_kurtosis, 0) +
        COALESCE(std_skewness, 0) +
        COALESCE(std_trend_r2, 0) +
        COALESCE(std_recurrence_rate, 0) +
        COALESCE(std_determinism, 0)
    ) / 10.0 AS fingerprint_volatility

FROM signal_stats
ORDER BY cohort, signal_id;
