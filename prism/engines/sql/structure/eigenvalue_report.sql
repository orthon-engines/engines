-- Eigenvalue Structure Report
-- Queries structure.parquet for eigenvalue/PCA metrics

-- Summary statistics
SELECT
    COUNT(*) as n_entities,
    AVG(eigenvalue_n_significant) as avg_n_significant,
    AVG(eigenvalue_participation_ratio) as avg_participation_ratio,
    AVG(eigenvalue_spectral_entropy) as avg_spectral_entropy
FROM read_parquet('structure.parquet');

-- Effective dimensionality distribution
SELECT
    eigenvalue_n_significant,
    COUNT(*) as n_entities,
    AVG(eigenvalue_participation_ratio) as avg_participation_ratio
FROM read_parquet('structure.parquet')
GROUP BY eigenvalue_n_significant
ORDER BY eigenvalue_n_significant;

-- Entities with high dimensionality (complex dynamics)
SELECT
    unit_id,
    eigenvalue_n_significant,
    eigenvalue_participation_ratio,
    eigenvalue_spectral_entropy,
    eigenvalue_n_signals
FROM read_parquet('structure.parquet')
WHERE eigenvalue_n_significant > 5
ORDER BY eigenvalue_n_significant DESC
LIMIT 20;

-- Entities with low dimensionality (simple/redundant)
SELECT
    unit_id,
    eigenvalue_n_significant,
    eigenvalue_participation_ratio,
    eigenvalue_n_signals
FROM read_parquet('structure.parquet')
WHERE eigenvalue_n_significant <= 2
  AND eigenvalue_n_signals > 5
ORDER BY eigenvalue_participation_ratio
LIMIT 20;

-- Tracy-Widom significance (unusual largest eigenvalue)
SELECT
    unit_id,
    eigenvalue_tracy_widom_pvalue,
    eigenvalue_1,
    eigenvalue_mp_threshold
FROM read_parquet('structure.parquet')
WHERE eigenvalue_tracy_widom_pvalue < 0.05
ORDER BY eigenvalue_tracy_widom_pvalue
LIMIT 20;
