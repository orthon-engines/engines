-- =============================================================================
-- PRISM C-MAPSS PHYSICS VALIDATION QUERIES
-- Run after indicator_vector and laplace complete
-- =============================================================================

-- 1. LEADING INDICATORS: Which sensors move first in degradation?
-- Look for sensors whose metrics shift earliest relative to RUL
WITH sensor_shifts AS (
    SELECT 
        indicator_id,
        engine_id,
        MIN(CASE WHEN abs(metric_value - lag_value) > 2 * metric_std THEN cycle END) as first_shift_cycle,
        MAX(cycle) as failure_cycle
    FROM (
        SELECT 
            indicator_id,
            SPLIT_PART(indicator_id, '_', 1) as engine_id,
            obs_date as cycle,
            metric_name,
            metric_value,
            LAG(metric_value, 10) OVER (PARTITION BY indicator_id, metric_name ORDER BY obs_date) as lag_value,
            STDDEV(metric_value) OVER (PARTITION BY indicator_id, metric_name) as metric_std
        FROM indicator_vectors
        WHERE metric_name IN ('hurst_exponent', 'entropy_permutation', 'garch_persistence')
    )
    WHERE lag_value IS NOT NULL
    GROUP BY indicator_id, engine_id
),
lead_times AS (
    SELECT 
        indicator_id,
        engine_id,
        failure_cycle - first_shift_cycle as lead_cycles
    FROM sensor_shifts
    WHERE first_shift_cycle IS NOT NULL
)
SELECT 
    REPLACE(indicator_id, engine_id || '_', '') as sensor,
    AVG(lead_cycles) as avg_lead_cycles,
    STDDEV(lead_cycles) as std_lead_cycles,
    COUNT(*) as n_engines
FROM lead_times
GROUP BY sensor
ORDER BY avg_lead_cycles DESC;


-- 2. PROPAGATION PATH: Transfer entropy between sensors
-- Which sensors drive which?
SELECT 
    indicator_from as source_sensor,
    indicator_to as target_sensor,
    AVG(effective_te) as avg_te,
    AVG(CASE WHEN p_value < 0.05 THEN 1 ELSE 0 END) as pct_significant,
    COUNT(*) as n_observations
FROM transfer_entropy_results
WHERE effective_te > 0.05
GROUP BY indicator_from, indicator_to
HAVING pct_significant > 0.5
ORDER BY avg_te DESC
LIMIT 20;


-- 3. PHASE TRANSITION: When does geometry change?
-- Look for discontinuities in hull volume, cluster count, mass ratio
WITH geometry_deltas AS (
    SELECT 
        cohort_id,
        window_end,
        hull_volume,
        hull_volume - LAG(hull_volume) OVER (PARTITION BY cohort_id ORDER BY window_end) as d_hull,
        cluster_count,
        cluster_count - LAG(cluster_count) OVER (PARTITION BY cohort_id ORDER BY window_end) as d_clusters,
        mass_ratio,
        mass_ratio - LAG(mass_ratio) OVER (PARTITION BY cohort_id ORDER BY window_end) as d_mass_ratio
    FROM geometry_snapshots
)
SELECT 
    cohort_id as engine_id,
    window_end as transition_cycle,
    d_hull,
    d_clusters,
    d_mass_ratio,
    CASE 
        WHEN abs(d_hull) > 0.2 THEN 'hull_collapse'
        WHEN d_clusters != 0 THEN 'topology_change'
        WHEN abs(d_mass_ratio) > 0.3 THEN 'mass_concentration'
    END as transition_type
FROM geometry_deltas
WHERE abs(d_hull) > 0.2 
   OR d_clusters != 0 
   OR abs(d_mass_ratio) > 0.3
ORDER BY cohort_id, window_end;


-- 4. COHORT STRUCTURE: Do sensor groupings match subsystems?
-- Compare clustering results to known sensor groupings
SELECT 
    cluster_label,
    ARRAY_AGG(DISTINCT REPLACE(indicator_id, engine_id || '_', '')) as sensors,
    COUNT(DISTINCT indicator_id) as n_sensors,
    AVG(silhouette_score) as cohesion
FROM clustering_results
GROUP BY cluster_label
ORDER BY cluster_label;


-- 5. DEGRADATION SIGNATURE: Laplace frequency profile at failure
-- What does the frequency fingerprint look like near failure?
WITH rul_bands AS (
    SELECT 
        indicator_id,
        obs_date,
        metric_name,
        metric_value,
        CASE 
            WHEN rul > 100 THEN 'healthy'
            WHEN rul BETWEEN 50 AND 100 THEN 'degrading'
            WHEN rul BETWEEN 20 AND 50 THEN 'critical'
            WHEN rul < 20 THEN 'failing'
        END as phase
    FROM indicator_vectors iv
    JOIN rul_labels rl ON iv.indicator_id = rl.indicator_id AND iv.obs_date = rl.cycle
    WHERE metric_name LIKE 'laplace_%'
)
SELECT 
    REPLACE(indicator_id, SPLIT_PART(indicator_id, '_', 1) || '_', '') as sensor,
    phase,
    metric_name,
    AVG(metric_value) as avg_value,
    STDDEV(metric_value) as std_value
FROM rul_bands
GROUP BY sensor, phase, metric_name
ORDER BY sensor, metric_name, 
    CASE phase 
        WHEN 'healthy' THEN 1 
        WHEN 'degrading' THEN 2 
        WHEN 'critical' THEN 3 
        WHEN 'failing' THEN 4 
    END;


-- 6. STRESS ACCUMULATION: Energy and tension trajectory
SELECT 
    cohort_id as engine_id,
    window_end as cycle,
    energy,
    tension,
    break_count,
    SUM(break_count) OVER (PARTITION BY cohort_id ORDER BY window_end) as cumulative_breaks
FROM geometry_snapshots
ORDER BY cohort_id, window_end;


-- 7. SENSOR IMPORTANCE: Which sensors have highest discriminative power?
-- Based on JFD scores from characterization
SELECT 
    indicator_id,
    engine as engine_name,
    metric_name as key_metric,
    effect_size,
    grade
FROM characterization_results
WHERE grade IN ('A', 'B')
ORDER BY effect_size DESC;


-- 8. COINTEGRATION STRUCTURE: Which sensors are bound together?
SELECT 
    indicator_a,
    indicator_b,
    beta as cointegration_coefficient,
    half_life,
    is_cointegrated
FROM cointegration_pairs
WHERE is_cointegrated = TRUE
ORDER BY half_life;


-- 9. SUMMARY REPORT: Engine-level physics fingerprint
SELECT 
    engine_id,
    -- Leading indicators
    FIRST(leading_sensor) as primary_lead_sensor,
    FIRST(lead_cycles) as lead_time,
    -- Phase transition
    FIRST(transition_cycle) as phase_transition_at,
    FIRST(transition_type) as transition_type,
    -- Final state
    LAST(hull_volume) as final_hull_volume,
    LAST(mass_ratio) as final_mass_ratio,
    LAST(energy) as final_energy,
    -- Failure
    MAX(cycle) as failure_cycle
FROM engine_summary_view
GROUP BY engine_id;