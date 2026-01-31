-- Wavelet Structure Report
-- Queries structure.parquet for wavelet coherence metrics

-- Summary statistics
SELECT
    COUNT(*) as n_entities,
    AVG(wavelet_mean_coherence) as avg_mean_coherence,
    AVG(wavelet_max_coherence) as avg_max_coherence,
    AVG(wavelet_avg_wavelet_entropy) as avg_wavelet_entropy,
    AVG(wavelet_avg_temporal_variability) as avg_temporal_variability
FROM read_parquet('structure.parquet');

-- Wavelet coherence vs spectral coherence comparison
SELECT
    unit_id,
    wavelet_mean_coherence,
    spectral_mean_coherence,
    (wavelet_mean_coherence - spectral_mean_coherence) as coherence_diff
FROM read_parquet('structure.parquet')
WHERE wavelet_mean_coherence IS NOT NULL
  AND spectral_mean_coherence IS NOT NULL
ORDER BY ABS(wavelet_mean_coherence - spectral_mean_coherence) DESC
LIMIT 20;

-- High temporal variability (non-stationary systems)
SELECT
    unit_id,
    wavelet_avg_temporal_variability,
    wavelet_mean_coherence,
    wavelet_avg_wavelet_entropy
FROM read_parquet('structure.parquet')
WHERE wavelet_avg_temporal_variability > 1.0
ORDER BY wavelet_avg_temporal_variability DESC
LIMIT 20;

-- Low wavelet entropy (scale-localized signals)
SELECT
    unit_id,
    wavelet_avg_wavelet_entropy,
    wavelet_mean_coherence,
    wavelet_avg_temporal_variability
FROM read_parquet('structure.parquet')
WHERE wavelet_avg_wavelet_entropy < 1.5
ORDER BY wavelet_avg_wavelet_entropy
LIMIT 20;

-- Wavelet coherence distribution
SELECT
    CASE
        WHEN wavelet_mean_coherence < 0.3 THEN 'low'
        WHEN wavelet_mean_coherence < 0.6 THEN 'moderate'
        ELSE 'high'
    END as coherence_level,
    COUNT(*) as n_entities,
    AVG(wavelet_mean_coherence) as avg_coherence,
    AVG(wavelet_avg_temporal_variability) as avg_variability
FROM read_parquet('structure.parquet')
GROUP BY 1
ORDER BY avg_coherence;
