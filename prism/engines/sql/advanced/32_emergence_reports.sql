-- =============================================================================
-- EMERGENCE REPORTS
-- Synergy, redundancy, and information decomposition
-- =============================================================================

-- Report: High Synergy Triplets
-- Signal combinations with emergent information
SELECT
    unit_id,
    source_1,
    source_2,
    target,
    synergy,
    synergy_ratio,
    total_info,
    CASE
        WHEN synergy_ratio > 0.5 THEN 'HIGH_EMERGENCE'
        WHEN synergy_ratio > 0.3 THEN 'MODERATE_EMERGENCE'
        ELSE 'LOW_EMERGENCE'
    END AS emergence_level
FROM read_parquet('emergence_triplets.parquet')
WHERE synergy_ratio > 0.2
ORDER BY synergy_ratio DESC;

-- Report: Redundant Signal Pairs
-- Signals that provide overlapping information
SELECT
    unit_id,
    source_1,
    source_2,
    target,
    redundancy,
    total_info,
    redundancy / NULLIF(total_info, 0) AS redundancy_ratio,
    'Consider sensor consolidation' AS recommendation
FROM read_parquet('emergence_triplets.parquet')
WHERE redundancy / NULLIF(total_info, 0) > 0.7
ORDER BY redundancy DESC;

-- Report: Unique Information Sources
-- Signals providing non-overlapping information
SELECT
    unit_id,
    source_1,
    source_2,
    target,
    unique_1,
    unique_2,
    CASE
        WHEN unique_1 > unique_2 * 2 THEN source_1
        WHEN unique_2 > unique_1 * 2 THEN source_2
        ELSE 'BALANCED'
    END AS dominant_source,
    'Unique information not captured by other signals' AS note
FROM read_parquet('emergence_triplets.parquet')
WHERE unique_1 > 0.1 OR unique_2 > 0.1
ORDER BY unique_1 + unique_2 DESC;

-- Report: Pairwise Mutual Information
-- Information shared between signal pairs
SELECT
    unit_id,
    signal_a,
    signal_b,
    mutual_information,
    CASE
        WHEN mutual_information > 1.0 THEN 'HIGH_MI'
        WHEN mutual_information > 0.5 THEN 'MODERATE_MI'
        WHEN mutual_information > 0.1 THEN 'LOW_MI'
        ELSE 'INDEPENDENT'
    END AS mi_level
FROM read_parquet('emergence_pairwise.parquet')
ORDER BY mutual_information DESC;

-- Report: Emergence Summary by Entity
-- Aggregate emergence metrics
SELECT
    unit_id,
    n_samples,
    n_signals,
    n_pairs,
    n_triplets,
    total_synergy,
    total_redundancy,
    total_unique,
    emergence_ratio,
    redundancy_ratio,
    mean_pairwise_mi,
    CASE
        WHEN emergence_ratio > 0.3 THEN 'HIGH_EMERGENCE'
        WHEN emergence_ratio > 0.15 THEN 'MODERATE'
        ELSE 'LOW_EMERGENCE'
    END AS emergence_status
FROM read_parquet('emergence_summary.parquet')
ORDER BY emergence_ratio DESC;

-- Report: Information Distribution
-- How is information distributed across triplets
SELECT
    unit_id,
    AVG(redundancy) AS avg_redundancy,
    AVG(synergy) AS avg_synergy,
    AVG(unique_1 + unique_2) AS avg_unique,
    AVG(synergy_ratio) AS avg_synergy_ratio,
    COUNT(*) AS n_triplets
FROM read_parquet('emergence_triplets.parquet')
GROUP BY unit_id
ORDER BY avg_synergy_ratio DESC;

-- Report: Target Signal Analysis
-- Which signals are best predicted by combinations
SELECT
    target,
    COUNT(*) AS n_source_pairs,
    AVG(total_info) AS avg_total_info,
    AVG(synergy) AS avg_synergy,
    MAX(total_info) AS max_total_info
FROM read_parquet('emergence_triplets.parquet')
GROUP BY target
ORDER BY avg_total_info DESC;

-- Report: Redundant Sensors Alert
-- Sensors that may be redundant
SELECT DISTINCT
    unit_id,
    signal_a,
    signal_b,
    mutual_information,
    'HIGH_REDUNDANCY' AS alert_type,
    'These signals share most information - consider consolidation' AS recommendation
FROM read_parquet('emergence_pairwise.parquet')
WHERE mutual_information > 2.0
ORDER BY mutual_information DESC;
