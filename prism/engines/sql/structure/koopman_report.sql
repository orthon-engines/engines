-- Koopman/DMD Structure Report
-- Queries structure.parquet for Dynamic Mode Decomposition metrics

-- Summary statistics
SELECT
    COUNT(*) as n_entities,
    AVG(koopman_reconstruction_error) as avg_reconstruction_error,
    AVG(koopman_spectral_radius) as avg_spectral_radius,
    AVG(koopman_dominant_frequency) as avg_dominant_frequency,
    AVG(koopman_mode_coherence) as avg_mode_coherence
FROM read_parquet('structure.parquet');

-- Stability classification by spectral radius
SELECT
    CASE
        WHEN koopman_spectral_radius < 0.95 THEN 'decaying'
        WHEN koopman_spectral_radius < 1.05 THEN 'stable'
        ELSE 'growing'
    END as stability_class,
    COUNT(*) as n_entities,
    AVG(koopman_spectral_radius) as avg_spectral_radius,
    AVG(koopman_reconstruction_error) as avg_error
FROM read_parquet('structure.parquet')
GROUP BY 1
ORDER BY avg_spectral_radius;

-- Best DMD fits (low reconstruction error)
SELECT
    unit_id,
    koopman_reconstruction_error,
    koopman_rank,
    koopman_mode_coherence,
    koopman_spectral_radius
FROM read_parquet('structure.parquet')
WHERE koopman_reconstruction_error < 0.1
ORDER BY koopman_reconstruction_error
LIMIT 20;

-- Unstable systems (growing modes)
SELECT
    unit_id,
    koopman_spectral_radius,
    koopman_growth_rate_1,
    koopman_dominant_frequency,
    koopman_reconstruction_error
FROM read_parquet('structure.parquet')
WHERE koopman_spectral_radius > 1.05
ORDER BY koopman_spectral_radius DESC
LIMIT 20;

-- Dominant frequencies
SELECT
    unit_id,
    koopman_dominant_frequency,
    koopman_frequency_1,
    koopman_frequency_2,
    koopman_mode_coherence
FROM read_parquet('structure.parquet')
WHERE koopman_dominant_frequency IS NOT NULL
ORDER BY koopman_dominant_frequency DESC
LIMIT 20;
