-- Spectral Structure Report
-- Queries structure.parquet for PSD and coherence metrics

-- Summary statistics
SELECT
    COUNT(*) as n_entities,
    AVG(spectral_mean_coherence) as avg_mean_coherence,
    AVG(spectral_max_coherence) as avg_max_coherence,
    AVG(spectral_n_significant_coherence) as avg_n_significant,
    AVG(spectral_avg_dominant_frequency) as avg_dominant_freq,
    AVG(spectral_avg_spectral_entropy) as avg_spectral_entropy
FROM read_parquet('structure.parquet');

-- Coherence distribution
SELECT
    CASE
        WHEN spectral_mean_coherence < 0.3 THEN 'low'
        WHEN spectral_mean_coherence < 0.6 THEN 'moderate'
        ELSE 'high'
    END as coherence_level,
    COUNT(*) as n_entities,
    AVG(spectral_mean_coherence) as avg_coherence,
    AVG(spectral_n_significant_coherence) as avg_n_significant
FROM read_parquet('structure.parquet')
GROUP BY 1
ORDER BY avg_coherence;

-- Highly coherent systems (synchronized signals)
SELECT
    unit_id,
    spectral_mean_coherence,
    spectral_max_coherence,
    spectral_n_significant_coherence,
    spectral_n_signals
FROM read_parquet('structure.parquet')
WHERE spectral_mean_coherence > 0.6
ORDER BY spectral_mean_coherence DESC
LIMIT 20;

-- Low spectral entropy (narrowband/periodic signals)
SELECT
    unit_id,
    spectral_avg_spectral_entropy,
    spectral_avg_dominant_frequency,
    spectral_mean_coherence
FROM read_parquet('structure.parquet')
WHERE spectral_avg_spectral_entropy < 2.0
ORDER BY spectral_avg_spectral_entropy
LIMIT 20;

-- High spectral entropy (broadband/noise-like)
SELECT
    unit_id,
    spectral_avg_spectral_entropy,
    spectral_avg_dominant_frequency,
    spectral_mean_coherence
FROM read_parquet('structure.parquet')
WHERE spectral_avg_spectral_entropy > 4.0
ORDER BY spectral_avg_spectral_entropy DESC
LIMIT 20;
