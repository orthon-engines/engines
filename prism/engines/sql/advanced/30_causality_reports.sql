-- =============================================================================
-- CAUSALITY REPORTS
-- Causal network analysis via Granger causality and Transfer Entropy
-- =============================================================================

-- Report: Significant Causal Edges
-- All edges with p < 0.05
SELECT
    unit_id,
    source,
    target,
    granger_f,
    granger_p,
    transfer_entropy,
    'SIGNIFICANT' AS edge_status
FROM read_parquet('causality_edges.parquet')
WHERE is_significant = TRUE
ORDER BY transfer_entropy DESC;

-- Report: Top Causal Links by Transfer Entropy
-- Strongest information flow relationships
SELECT
    unit_id,
    source,
    target,
    transfer_entropy,
    granger_f,
    granger_p,
    CASE
        WHEN transfer_entropy > 0.5 THEN 'STRONG'
        WHEN transfer_entropy > 0.2 THEN 'MODERATE'
        ELSE 'WEAK'
    END AS link_strength
FROM read_parquet('causality_edges.parquet')
WHERE transfer_entropy > 0.1
ORDER BY transfer_entropy DESC;

-- Report: Causal Network Summary
-- Network-level metrics per entity
SELECT
    unit_id,
    n_samples,
    n_signals,
    density,
    hierarchy,
    n_feedback_loops,
    top_driver,
    top_driver_flow,
    top_sink,
    bottleneck,
    bottleneck_centrality,
    mean_te,
    n_significant_edges,
    CASE
        WHEN n_feedback_loops > 5 THEN 'COMPLEX'
        WHEN hierarchy < 0.3 THEN 'CIRCULAR'
        WHEN hierarchy > 0.7 THEN 'HIERARCHICAL'
        ELSE 'MODERATE'
    END AS network_type
FROM read_parquet('causality_network.parquet')
ORDER BY density DESC;

-- Report: Feedback Loop Alerts
-- Entities with many feedback loops (potential instability)
SELECT
    unit_id,
    n_feedback_loops,
    hierarchy,
    density,
    CASE
        WHEN n_feedback_loops > 5 THEN 'WARNING'
        WHEN n_feedback_loops > 3 THEN 'ELEVATED'
        ELSE 'NORMAL'
    END AS loop_status,
    'Multiple feedback loops may amplify disturbances' AS note
FROM read_parquet('causality_network.parquet')
WHERE n_feedback_loops > 2
ORDER BY n_feedback_loops DESC;

-- Report: Driver Analysis
-- Which signals drive the system
SELECT
    unit_id,
    top_driver,
    top_driver_flow,
    bottleneck,
    bottleneck_centrality,
    'Primary driver of system dynamics' AS role
FROM read_parquet('causality_network.parquet')
WHERE top_driver_flow > 0
ORDER BY top_driver_flow DESC;

-- Report: Sink Analysis
-- Which signals are driven by others
SELECT
    unit_id,
    top_sink,
    top_sink_flow,
    'Responds to other signals' AS role
FROM read_parquet('causality_network.parquet')
WHERE top_sink_flow < 0
ORDER BY top_sink_flow ASC;

-- Report: Network Density by Entity
-- How interconnected is each entity's signal network
SELECT
    unit_id,
    density,
    n_significant_edges,
    n_signals,
    CASE
        WHEN density > 0.5 THEN 'HIGHLY_CONNECTED'
        WHEN density > 0.2 THEN 'MODERATE'
        ELSE 'SPARSE'
    END AS connectivity_status
FROM read_parquet('causality_network.parquet')
ORDER BY density DESC;
