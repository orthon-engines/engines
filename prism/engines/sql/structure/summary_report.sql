-- Structure Analysis Summary Report
-- Combines metrics from all structure engines

-- Overall summary
SELECT
    COUNT(*) as n_entities,
    -- Covariance
    AVG(covariance_mean_correlation) as avg_correlation,
    AVG(covariance_condition_number) as avg_condition_number,
    -- Eigenvalue
    AVG(eigenvalue_n_significant) as avg_n_significant_eigenvalues,
    AVG(eigenvalue_participation_ratio) as avg_participation_ratio,
    -- Koopman
    AVG(koopman_reconstruction_error) as avg_dmd_error,
    AVG(koopman_spectral_radius) as avg_spectral_radius,
    -- Spectral
    AVG(spectral_mean_coherence) as avg_spectral_coherence,
    -- Wavelet
    AVG(wavelet_mean_coherence) as avg_wavelet_coherence,
    AVG(wavelet_avg_temporal_variability) as avg_temporal_variability
FROM read_parquet('structure.parquet');

-- Entity complexity ranking
SELECT
    unit_id,
    -- Correlation structure
    covariance_mean_correlation,
    -- Effective dimensionality
    eigenvalue_n_significant,
    eigenvalue_participation_ratio,
    -- Dynamic stability
    koopman_spectral_radius,
    koopman_reconstruction_error,
    -- Frequency coherence
    spectral_mean_coherence,
    wavelet_mean_coherence,
    -- Complexity score (composite)
    (eigenvalue_participation_ratio +
     spectral_avg_spectral_entropy +
     wavelet_avg_temporal_variability) / 3 as complexity_score
FROM read_parquet('structure.parquet')
ORDER BY complexity_score DESC
LIMIT 20;

-- Anomaly detection: entities with unusual structure
SELECT
    unit_id,
    covariance_condition_number,
    eigenvalue_tracy_widom_pvalue,
    koopman_spectral_radius,
    spectral_mean_coherence
FROM read_parquet('structure.parquet')
WHERE
    -- Ill-conditioned covariance
    covariance_condition_number > 100
    -- Significant eigenvalue structure
    OR eigenvalue_tracy_widom_pvalue < 0.01
    -- Unstable dynamics
    OR koopman_spectral_radius > 1.1
    -- Highly coherent
    OR spectral_mean_coherence > 0.8
ORDER BY unit_id;

-- System type classification based on structural signatures
SELECT
    CASE
        WHEN koopman_spectral_radius < 0.9 THEN 'strongly_damped'
        WHEN koopman_spectral_radius < 1.0 THEN 'damped'
        WHEN koopman_spectral_radius < 1.1 THEN 'marginally_stable'
        ELSE 'unstable'
    END as stability_class,
    CASE
        WHEN eigenvalue_participation_ratio < 2 THEN 'low_dimensional'
        WHEN eigenvalue_participation_ratio < 5 THEN 'moderate_dimensional'
        ELSE 'high_dimensional'
    END as dimensionality_class,
    CASE
        WHEN spectral_mean_coherence < 0.3 THEN 'independent'
        WHEN spectral_mean_coherence < 0.6 THEN 'weakly_coupled'
        ELSE 'strongly_coupled'
    END as coupling_class,
    COUNT(*) as n_entities
FROM read_parquet('structure.parquet')
GROUP BY 1, 2, 3
ORDER BY n_entities DESC;
