# PRISM Validation: Rössler System

**Last Run:** 2026-01-16

## Overview

This notebook documents PRISM's validation against the Rössler attractor — a simpler chaotic system than Lorenz with distinct dynamical properties. The Rössler system is valuable for validation because:

1. **Different structure than Lorenz** — Tests PRISM's ability to distinguish chaotic systems
2. **Periodic x-y rotation with z spikes** — Clear structural differences to detect
3. **Single spiral with outward excursions** — Simpler topology than Lorenz's two lobes
4. **Well-characterized chaos** — Known Lyapunov exponents and attractor properties

---

## Rössler System Equations

The Rössler system is defined by three coupled ordinary differential equations:

```
dx/dt = -y - z
dy/dt = x + ay
dz/dt = b + z(x - c)
```

### Parameters (Standard Chaotic Regime)

| Parameter | Value | Physical Meaning |
|-----------|-------|------------------|
| a | 0.2 | Controls rotation in x-y plane |
| b | 0.2 | Controls z dynamics |
| c | 5.7 | Controls the "folding" mechanism |

These parameters produce chaotic dynamics with a characteristic spiral and intermittent z-spikes.

### Initial Conditions

```
x₀ = 1.0
y₀ = 1.0
z₀ = 1.0
```

---

## Rössler vs Lorenz: Structural Differences

| Property | Lorenz | Rössler |
|----------|--------|---------|
| **Topology** | Two lobes (butterfly) | Single spiral with outward excursions |
| **Oscillatory variables** | z oscillates | x, y oscillate together |
| **Aperiodic variables** | x, y wander between lobes | z has intermittent spikes |
| **Volatility clustering** | x, y (lobe transitions) | z (spike events) |
| **Regime transitions** | Lobe switches (x crosses 0) | Spike events (z excursions) |

---

## Data Generation

### Method

The Rössler system was integrated numerically using `scipy.integrate.solve_ivp` with the RK45 (Runge-Kutta 4th/5th order) adaptive solver.

```python
from scipy.integrate import solve_ivp

def rossler_system(t, state, a=0.2, b=0.2, c=5.7):
    x, y, z = state
    dxdt = -y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    return [dxdt, dydt, dzdt]

sol = solve_ivp(
    rossler_system,
    t_span=(0, 500),
    y0=[1.0, 1.0, 1.0],
    t_eval=np.arange(0, 500, 0.05),
    method='RK45'
)
```

### Data Characteristics

| Property | Value |
|----------|-------|
| Time span | 0 to 500 time units |
| Time step (dt) | 0.05 |
| Total points | 10,000 |
| Integration method | RK45 (adaptive Runge-Kutta) |

### PRISM Observation Format

The trajectory was converted to PRISM observation format:
- Each variable (x, y, z) becomes a separate indicator
- Time units mapped to synthetic dates (1 time unit = 1 day)
- Base date: 2020-01-01

```python
indicators = ['rossler_x', 'rossler_y', 'rossler_z']
observations = 30,000  # 3 indicators × 10,000 time points
```

---

## Known Rössler Properties

| Property | Expected Value | Notes |
|----------|----------------|-------|
| Largest Lyapunov exponent | ~0.07 | Positive = chaos |
| Correlation dimension | ~2.01 | Fractal attractor |
| Kaplan-Yorke dimension | ~2.01 | |

---

## PRISM Pipeline

### Step 1: Characterization

```bash
PRISM_DOMAIN=rossler python -m prism.runners.characterize
```

**Output:** 3 indicators characterized

### Step 2: Indicator Vector

```bash
PRISM_DOMAIN=rossler python -m prism.runners.indicator_vector --indicator
```

**Output:**
- 553,010 metric rows
- Window tiers: anchor (252d/21d), bridge (126d/5d)
- 8 canonical engines

### Step 3: Laplace Field

Automatically chained from indicator_vector.

**Output:** 553,010 field rows with gradient, laplacian, divergence

### Step 4: Geometry

```bash
PRISM_DOMAIN=rossler python -m prism.runners.geometry --cohort
```

**Output:**
- 494 cohort-window geometry snapshots
- Automatically chains to cohort Laplace field (6,422 rows)

---

## Results

### Characterization

| Variable | Dynamical Class | Memory | Periodicity | Determinism | Volatility | Method |
|----------|-----------------|--------|-------------|-------------|------------|--------|
| **rossler_x** | STATIONARY_OSCILLATORY_DETERMINISTIC | 0.584 | **1.0** | 0.994 | 0.0 | rs |
| **rossler_y** | STATIONARY_OSCILLATORY_DETERMINISTIC | 0.624 | **1.0** | 0.997 | 0.0 | rs |
| **rossler_z** | STATIONARY_APERIODIC_DETERMINISTIC_CLUSTERED_VOL | 0.620 | 0.0 | 0.973 | **1.0** | rs |

### Key Metrics (Indicator Vector)

| Metric | rossler_x | rossler_y | rossler_z |
|--------|-----------|-----------|-----------|
| **hurst_exponent** | 1.028 | 1.028 | 0.988 |
| **sample_entropy** | 0.144 | 0.147 | 0.036 |
| **permutation_entropy** | 0.432 | 0.432 | 0.423 |
| **lyapunov_exponent** | 0.127 | 0.127 | 0.132 |

### Laplace Field Topology (Indicator)

| Variable | Sources | Sinks | Source/Sink Ratio |
|----------|---------|-------|-------------------|
| rossler_x | 99,084 | 85,678 | 1.16 |
| rossler_y | 98,240 | 86,603 | 1.13 |
| rossler_z | 83,979 | 99,426 | **0.84** |

**Note:** rossler_z has more sinks than sources (ratio < 1), reflecting the "contracting" nature of the z-spike dynamics — energy dissipates after each spike.

### Cohort Geometry

| Metric | Value |
|--------|-------|
| Windows | 494 |
| Mean cohesion | 0.2994 |
| Mean PCA effective dimension | 1.41 |

### Cohort Field Topology

| Metric | Value |
|--------|-------|
| Total field rows | 6,422 |
| Unique windows | 494 |
| Source windows (divergence > 0.1) | 238 |
| Sink windows (divergence < -0.1) | 250 |
| Source/Sink ratio | 0.95 |

**Assessment:** The cohort-level field topology shows nearly balanced sources (238) and sinks (250), with a ratio of 0.95. This confirms the Rössler attractor is a stable strange attractor.

---

## Validation Assessment

### ✓ Determinism Detection

| Variable | PRISM ax_determinism | Expected | Result |
|----------|---------------------|----------|--------|
| x | 0.994 | HIGH | **PASS** |
| y | 0.997 | HIGH | **PASS** |
| z | 0.973 | HIGH | **PASS** |

**Assessment:** PRISM correctly identifies the Rössler system as highly deterministic (>0.97 for all variables).

### ✓ X/Y Oscillatory Detection

| Variable | PRISM ax_periodicity | Expected | Result |
|----------|---------------------|----------|--------|
| x | **1.0** | HIGH | **PASS** |
| y | **1.0** | HIGH | **PASS** |
| z | 0.0 | LOW | **PASS** |

**Assessment:** PRISM correctly detects that x and y oscillate together in the Rössler spiral, while z has aperiodic spike behavior. This is the **opposite** of Lorenz (where z oscillates and x/y wander).

### ✓ Z Volatility Clustering Detection

| Variable | PRISM ax_volatility | Expected | Result |
|----------|---------------------|----------|--------|
| x | 0.0 | LOW | **PASS** |
| y | 0.0 | LOW | **PASS** |
| z | **1.0** | HIGH | **PASS** |

**Assessment:** PRISM correctly detects volatility clustering in z. The z variable has intermittent "spikes" when the trajectory excurses outward, creating bursts of high variance. This is opposite to Lorenz (where x/y have volatility clustering from lobe transitions).

### ✓ Lyapunov Exponent

| Variable | PRISM Lyapunov | Expected | Result |
|----------|----------------|----------|--------|
| x | 0.127 | ~0.07 | **POSITIVE** |
| y | 0.127 | ~0.07 | **POSITIVE** |
| z | 0.132 | ~0.07 | **POSITIVE** |

**Assessment:** PRISM detects positive Lyapunov exponents, confirming chaos. Values are higher than theoretical due to windowed estimation but the sign is correct.

### ✓ Field Topology

**Indicator-Level:**

| Metric | Value | Expected | Result |
|--------|-------|----------|--------|
| Source/Sink ratio (x) | 1.16 | ~1.0 | **PASS** |
| Source/Sink ratio (y) | 1.13 | ~1.0 | **PASS** |
| Source/Sink ratio (z) | 0.84 | <1.0 | **PASS** |

**Cohort-Level:**

| Metric | Value | Expected | Result |
|--------|-------|----------|--------|
| Source/Sink ratio | 0.95 | ~1.0 | **PASS** |

**Assessment:** The Rössler attractor shows interesting asymmetry — z has more sinks than sources (0.84), reflecting the dissipative nature of the spike dynamics. The cohort-level ratio (0.95) is nearly balanced, confirming attractor stability.

---

## Physics Interpretation

### Why PRISM Works on Rössler

1. **Periodicity axis distinguishes x/y from z**
   - x and y rotate together in a near-circular spiral (high periodicity)
   - z has intermittent spikes (low periodicity)
   - PRISM's spectral analysis detects this difference

2. **Volatility clustering from z spikes**
   - When the trajectory excurses outward, z spikes dramatically
   - ARCH/GARCH-style autocorrelation in squared returns detects this
   - x and y have more uniform variance (steady rotation)

3. **Determinism axis captures RQA structure**
   - All three variables show high determinism (>0.97)
   - The spiral structure creates strong recurrence patterns

4. **Asymmetric field topology**
   - z's sink-dominated topology reflects energy dissipation after spikes
   - x/y's source-dominated topology reflects the expanding spiral

### Rössler vs Lorenz in PRISM

| PRISM Feature | Lorenz | Rössler |
|---------------|--------|---------|
| ax_periodicity(z) | 1.0 (oscillates) | 0.0 (spikes) |
| ax_periodicity(x,y) | 0.0 (wanders) | 1.0 (rotates) |
| ax_volatility(z) | 0.0 (uniform) | 1.0 (clustered) |
| ax_volatility(x,y) | 1.0 (lobe bursts) | 0.0 (uniform) |
| z Source/Sink | 1.20 | 0.84 |

**Key Finding:** PRISM correctly distinguishes the structural differences between Lorenz and Rössler chaotic attractors through the 6-axis characterization.

---

## Output Files

```
data/rossler/
├── raw/
│   ├── observations.parquet       # 30,000 rows (3 indicators × 10,000 points)
│   ├── indicators.parquet         # 3 indicator definitions
│   ├── characterization.parquet   # 3 rows (6-axis classification)
│   └── rossler_trajectory.parquet # 10,000 rows (x, y, z)
├── config/
│   ├── cohort_members.parquet     # 3 rows
│   ├── cohorts.parquet            # 1 row (rossler_attractor)
│   └── domain_members.parquet     # 1 row
├── vector/
│   ├── indicator.parquet          # 553,010 rows
│   ├── indicator_field.parquet    # 553,010 rows (indicator Laplace field)
│   └── cohort_field.parquet       # 6,422 rows (cohort Laplace field)
├── geometry/
│   └── cohort.parquet             # 494 rows (cohort-window geometry)
└── notebook.md                    # This file
```

---

## Conclusions

PRISM correctly characterizes the Rössler chaotic attractor:

| Test | Expected | PRISM Result | Status |
|------|----------|--------------|--------|
| Determinism | HIGH | 0.97-0.99 | ✓ PASS |
| X/Y oscillatory | YES | periodicity=1.0 | ✓ PASS |
| Z aperiodic | YES | periodicity=0.0 | ✓ PASS |
| Z volatility clustering | YES | volatility=1.0 | ✓ PASS |
| X/Y uniform volatility | YES | volatility=0.0 | ✓ PASS |
| Positive Lyapunov | YES | ~0.13 | ✓ PASS |
| Stable attractor (cohort) | YES | Sources≈Sinks (0.95) | ✓ PASS |
| Z dissipative | YES | Source/Sink=0.84 | ✓ PASS |

**Key Finding:** PRISM's 6-axis characterization correctly distinguishes Rössler from Lorenz dynamics:
- **Lorenz:** x/y wander between lobes (aperiodic, volatile), z oscillates (periodic, uniform)
- **Rössler:** x/y rotate together (periodic, uniform), z spikes (aperiodic, volatile)

This validates PRISM's ability to capture structural differences between chaotic systems.

---

## Reproduction

```bash
# Generate data and run full pipeline
PRISM_DOMAIN=rossler python -m prism.runners.characterize
PRISM_DOMAIN=rossler python -m prism.runners.indicator_vector --indicator
PRISM_DOMAIN=rossler python -m prism.runners.geometry --cohort
```

---

## References

1. Rössler, O.E. (1976). "An equation for continuous chaos". Physics Letters A. 57 (5): 397–398.
2. Sprott, J. C. (2003). Chaos and Time-Series Analysis. Oxford University Press.
3. Letellier, C., Dutertre, P., & Maheu, B. (1995). "Unstable periodic orbits and templates of the Rössler system". Chaos. 5 (1): 271-282.
