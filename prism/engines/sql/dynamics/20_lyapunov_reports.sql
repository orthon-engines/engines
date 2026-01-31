-- =============================================================================
-- LYAPUNOV REPORTS
-- Stability analysis via Lyapunov exponents
-- =============================================================================

-- Report: Lyapunov Summary
-- Overview of stability across all entities
SELECT
    unit_id,
    n_samples,
    lyapunov,
    stability_class,
    is_significant,
    surrogate_p_value,
    embedding_dim,
    embedding_tau,
    lyapunov_trend,
    lyapunov_trend_slope,
    CASE
        WHEN lyapunov > 0.5 THEN 'HIGHLY_CHAOTIC'
        WHEN lyapunov > 0.1 THEN 'CHAOTIC'
        WHEN lyapunov > 0 THEN 'MARGINAL'
        WHEN lyapunov > -0.1 THEN 'PERIODIC'
        ELSE 'STABLE'
    END AS stability_status
FROM read_parquet('dynamics_lyapunov.parquet')
ORDER BY lyapunov DESC;

-- Report: Stability Alerts
-- Entities with concerning stability characteristics
SELECT
    unit_id,
    lyapunov,
    stability_class,
    is_significant,
    lyapunov_trend,
    lyapunov_trend_p,
    CASE
        WHEN lyapunov > 0.5 THEN 'CRITICAL'
        WHEN lyapunov > 0.1 THEN 'WARNING'
        WHEN lyapunov_trend = 'increasing' AND lyapunov_trend_p < 0.05 THEN 'DESTABILIZING'
        ELSE 'NORMAL'
    END AS alert_level
FROM read_parquet('dynamics_lyapunov.parquet')
WHERE lyapunov > 0 OR (lyapunov_trend = 'increasing' AND lyapunov_trend_p < 0.1)
ORDER BY lyapunov DESC;

-- Report: Significant Chaos
-- Entities where positive Lyapunov is statistically significant
SELECT
    unit_id,
    lyapunov,
    surrogate_p_value,
    surrogate_z_score,
    stability_class,
    embedding_dim
FROM read_parquet('dynamics_lyapunov.parquet')
WHERE is_significant = TRUE AND lyapunov > 0
ORDER BY lyapunov DESC;

-- Report: Destabilizing Trends
-- Entities where Lyapunov exponent is trending upward
SELECT
    unit_id,
    lyapunov,
    lyapunov_trend,
    lyapunov_trend_slope,
    lyapunov_trend_p,
    stability_class
FROM read_parquet('dynamics_lyapunov.parquet')
WHERE lyapunov_trend = 'increasing' AND lyapunov_trend_p < 0.1
ORDER BY lyapunov_trend_slope DESC;

-- Report: Stability Class Summary
-- Count of entities by stability classification
SELECT
    stability_class,
    COUNT(*) as n_entities,
    AVG(lyapunov) as avg_lyapunov,
    MIN(lyapunov) as min_lyapunov,
    MAX(lyapunov) as max_lyapunov,
    SUM(CASE WHEN is_significant THEN 1 ELSE 0 END) as n_significant
FROM read_parquet('dynamics_lyapunov.parquet')
GROUP BY stability_class
ORDER BY avg_lyapunov DESC;

-- Report: Embedding Statistics
-- Summary of embedding parameters used
SELECT
    embedding_dim,
    embedding_tau,
    COUNT(*) as n_entities,
    AVG(lyapunov) as avg_lyapunov
FROM read_parquet('dynamics_lyapunov.parquet')
WHERE embedding_dim IS NOT NULL
GROUP BY embedding_dim, embedding_tau
ORDER BY n_entities DESC;
