-- =============================================================================
-- MOMENTUM REPORTS
-- Conservation Laws: Sum(tau) = I*alpha, Sum(F) = m*a
-- =============================================================================

-- Report: Momentum Balance Summary
-- Overview of momentum balance across all entities
SELECT
    unit_id,
    n_samples,
    torque_applied,
    torque_load,
    torque_friction,
    net_torque,
    expected_torque,
    torque_residual,
    angular_acceleration,
    vibration_energy,
    momentum_status
FROM read_parquet('physics_momentum.parquet')
ORDER BY ABS(torque_residual) DESC;

-- Report: Torque Imbalance Detection
-- Entities with significant torque imbalance
SELECT
    unit_id,
    torque_applied,
    torque_load,
    net_torque,
    expected_torque,
    torque_residual,
    ABS(torque_residual) / NULLIF(ABS(torque_applied), 0) * 100 AS residual_pct,
    momentum_status,
    CASE
        WHEN ABS(torque_residual) / NULLIF(ABS(torque_applied), 0) > 0.20 THEN 'SEVERE_IMBALANCE'
        WHEN ABS(torque_residual) / NULLIF(ABS(torque_applied), 0) > 0.10 THEN 'MODERATE_IMBALANCE'
        WHEN ABS(torque_residual) / NULLIF(ABS(torque_applied), 0) > 0.05 THEN 'MINOR_IMBALANCE'
        ELSE 'BALANCED'
    END AS imbalance_severity
FROM read_parquet('physics_momentum.parquet')
WHERE torque_residual IS NOT NULL
ORDER BY ABS(torque_residual) DESC;

-- Report: High Vibration Entities
-- Entities with excessive vibration energy
SELECT
    unit_id,
    vibration_energy,
    torque_residual,
    angular_acceleration,
    momentum_status,
    CASE
        WHEN vibration_energy > 2.0 THEN 'CRITICAL'
        WHEN vibration_energy > 1.0 THEN 'HIGH'
        WHEN vibration_energy > 0.5 THEN 'ELEVATED'
        ELSE 'NORMAL'
    END AS vibration_level
FROM read_parquet('physics_momentum.parquet')
WHERE vibration_energy > 0.5
ORDER BY vibration_energy DESC;

-- Report: Friction Analysis
-- Analysis of friction torque across entities
SELECT
    unit_id,
    torque_friction,
    torque_applied,
    torque_friction / NULLIF(torque_applied, 0) * 100 AS friction_pct,
    vibration_energy,
    momentum_status
FROM read_parquet('physics_momentum.parquet')
WHERE torque_friction IS NOT NULL AND torque_friction > 0
ORDER BY torque_friction DESC;

-- Report: Momentum Anomalies
-- All entities with momentum issues
SELECT
    unit_id,
    net_torque,
    torque_residual,
    vibration_energy,
    momentum_status,
    CASE
        WHEN momentum_status = 'IMBALANCE' THEN 'Check for mechanical issues, bearing wear, or coupling problems'
        WHEN momentum_status = 'HIGH_VIBRATION' THEN 'Investigate vibration source, check alignment and balance'
        ELSE 'System operating normally'
    END AS recommended_action
FROM read_parquet('physics_momentum.parquet')
WHERE momentum_status != 'NORMAL';

-- Report: Momentum Status Summary
-- Count of entities by momentum status
SELECT
    momentum_status,
    COUNT(*) as n_entities,
    AVG(ABS(torque_residual)) as avg_torque_residual,
    AVG(vibration_energy) as avg_vibration_energy,
    MAX(vibration_energy) as max_vibration_energy
FROM read_parquet('physics_momentum.parquet')
GROUP BY momentum_status
ORDER BY n_entities DESC;

-- Report: Angular Dynamics
-- Analysis of angular motion characteristics
SELECT
    unit_id,
    angular_velocity_mean,
    angular_acceleration,
    torque_applied,
    expected_torque,
    vibration_energy
FROM read_parquet('physics_momentum.parquet')
WHERE angular_velocity_mean IS NOT NULL
ORDER BY ABS(angular_acceleration) DESC;
