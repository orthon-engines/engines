-- =============================================================================
-- TOPOLOGY REPORTS
-- Persistent homology and topological data analysis
-- =============================================================================

-- Report: Topology Summary
-- Overview of topological features per entity
SELECT
    unit_id,
    n_samples,
    embedding_dim,
    embedding_tau,
    betti_0,
    betti_1,
    betti_2,
    topological_complexity,
    fragmentation,
    topology_change,
    CASE
        WHEN fragmentation THEN 'FRAGMENTED'
        WHEN betti_1 > 2 THEN 'COMPLEX_LOOPS'
        WHEN topology_change > 0.5 THEN 'CHANGING'
        ELSE 'STABLE'
    END AS topology_status
FROM read_parquet('topology.parquet')
ORDER BY topological_complexity DESC;

-- Report: Fragmentation Alerts
-- Entities with disconnected attractors
SELECT
    unit_id,
    betti_0,
    topological_complexity,
    persistence_entropy_h0,
    'FRAGMENTATION' AS alert_type,
    'Attractor has split into multiple components' AS warning
FROM read_parquet('topology.parquet')
WHERE fragmentation = TRUE
ORDER BY betti_0 DESC;

-- Report: Loop Structure Analysis
-- Entities with significant H1 features (loops)
SELECT
    unit_id,
    betti_1,
    total_persistence_h1,
    max_persistence_h1,
    persistence_entropy_h1,
    CASE
        WHEN betti_1 > 3 THEN 'MANY_LOOPS'
        WHEN betti_1 > 1 THEN 'MODERATE_LOOPS'
        WHEN betti_1 = 1 THEN 'SINGLE_LOOP'
        ELSE 'NO_LOOPS'
    END AS loop_structure
FROM read_parquet('topology.parquet')
WHERE betti_1 > 0
ORDER BY betti_1 DESC;

-- Report: Topological Complexity Ranking
-- Rank entities by complexity
SELECT
    unit_id,
    topological_complexity,
    betti_0 + betti_1 + betti_2 AS total_betti,
    persistence_entropy_h0,
    persistence_entropy_h1,
    CASE
        WHEN topological_complexity > 10 THEN 'HIGH_COMPLEXITY'
        WHEN topological_complexity > 5 THEN 'MODERATE'
        ELSE 'LOW_COMPLEXITY'
    END AS complexity_class
FROM read_parquet('topology.parquet')
ORDER BY topological_complexity DESC;

-- Report: Topology Change Alerts
-- Entities with significant structural changes
SELECT
    unit_id,
    topology_change,
    betti_1,
    fragmentation,
    CASE
        WHEN topology_change > 1.0 THEN 'MAJOR_CHANGE'
        WHEN topology_change > 0.5 THEN 'MODERATE_CHANGE'
        WHEN topology_change > 0.2 THEN 'MINOR_CHANGE'
        ELSE 'STABLE'
    END AS change_severity
FROM read_parquet('topology.parquet')
WHERE topology_change > 0.2
ORDER BY topology_change DESC;

-- Report: Persistence Entropy Summary
-- Entropy of persistence diagrams
SELECT
    unit_id,
    persistence_entropy_h0,
    persistence_entropy_h1,
    CASE
        WHEN persistence_entropy_h1 > 2.0 THEN 'HIGH_H1_ENTROPY'
        WHEN persistence_entropy_h1 > 1.0 THEN 'MODERATE'
        ELSE 'LOW'
    END AS entropy_status
FROM read_parquet('topology.parquet')
ORDER BY persistence_entropy_h1 DESC;

-- Report: Embedding Parameters Used
-- Summary of embedding choices
SELECT
    embedding_dim,
    embedding_tau,
    COUNT(*) AS n_entities,
    AVG(topological_complexity) AS avg_complexity,
    AVG(betti_1) AS avg_betti_1
FROM read_parquet('topology.parquet')
WHERE embedding_dim IS NOT NULL
GROUP BY embedding_dim, embedding_tau
ORDER BY n_entities DESC;
