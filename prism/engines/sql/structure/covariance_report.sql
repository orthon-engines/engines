-- Covariance Structure Report
-- Queries structure.parquet for covariance/correlation metrics

-- Summary statistics
SELECT
    COUNT(*) as n_entities,
    AVG(covariance_mean_correlation) as avg_mean_correlation,
    AVG(covariance_max_correlation) as avg_max_correlation,
    AVG(covariance_condition_number) as avg_condition_number,
    MIN(covariance_condition_number) as min_condition_number,
    MAX(covariance_condition_number) as max_condition_number
FROM read_parquet('structure.parquet');

-- Entities with high correlation (potential redundancy)
SELECT
    unit_id,
    covariance_mean_correlation,
    covariance_max_correlation,
    covariance_n_signals
FROM read_parquet('structure.parquet')
WHERE covariance_mean_correlation > 0.7
ORDER BY covariance_mean_correlation DESC
LIMIT 20;

-- Entities with ill-conditioned matrices (numerical instability risk)
SELECT
    unit_id,
    covariance_condition_number,
    covariance_determinant,
    covariance_trace
FROM read_parquet('structure.parquet')
WHERE covariance_condition_number > 100
ORDER BY covariance_condition_number DESC
LIMIT 20;

-- Correlation distribution
SELECT
    CASE
        WHEN covariance_mean_correlation < 0.2 THEN 'uncorrelated'
        WHEN covariance_mean_correlation < 0.5 THEN 'weak'
        WHEN covariance_mean_correlation < 0.7 THEN 'moderate'
        ELSE 'strong'
    END as correlation_level,
    COUNT(*) as n_entities,
    AVG(covariance_mean_correlation) as avg_correlation
FROM read_parquet('structure.parquet')
GROUP BY 1
ORDER BY avg_correlation;
