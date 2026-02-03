# MANIFEST CONTRACT

## What ORTHON Delivers, What PRISM Reads

**This document is the contract between ORTHON and PRISM.**
**PRISM reads this spec. PRISM never reads ORTHON code.**
**If this document and ORTHON disagree, file a bug against ORTHON.**

---

## Files ORTHON Produces

ORTHON places these files in the data directory before PRISM runs:

| File | Format | Required | Description |
|------|--------|----------|-------------|
| `observations.parquet` | Parquet | YES | Raw signal data |
| `typology.parquet` | Parquet | YES | Per-signal classification |
| `manifest.yaml` | YAML | YES | Engine selection + parameters |

PRISM must not run without all three files present.

---

## observations.parquet Schema

```
signal_id   str       Required. Signal name (e.g. "temp_1", "vib_x")
I           UInt32    Required. Sequential index 0,1,2,3... per signal_id
value       Float64   Required. The measurement
unit_id     str       Optional. Pass-through label. NEVER in groupby.
```

### Rules
- I is sequential per signal_id. No gaps. Starts at 0.
- I is NOT a timestamp. It's an index.
- unit_id is cargo. PRISM passes it through to output. Never groups by it.
- If unit_id is missing, treat as single entity.

---

## typology.parquet Schema

One row per signal_id. Contains the 10-dimension identity card.

```
signal_id           str       Signal name (joins to observations)
continuity          str       CONSTANT | CONTINUOUS | DISCRETE | EVENT
stationarity        str       STATIONARY | NON_STATIONARY | DIFFERENCE_STATIONARY | TREND_STATIONARY
temporal_pattern    str       PERIODIC | QUASI_PERIODIC | TRENDING | CHAOTIC | RANDOM | MEAN_REVERTING | CONSTANT
memory              str       LONG_MEMORY | SHORT_MEMORY | ANTI_PERSISTENT
complexity          str       LOW | MEDIUM | HIGH
distribution        str       GAUSSIAN | HEAVY_TAILED | LIGHT_TAILED | SKEWED_RIGHT | SKEWED_LEFT
amplitude           str       SMOOTH | NOISY | IMPULSIVE | MIXED
spectral            str       NARROWBAND | BROADBAND | HARMONIC | ONE_OVER_F
volatility          str       HOMOSCEDASTIC | HETEROSCEDASTIC | VOLATILITY_CLUSTERING
determinism         str       DETERMINISTIC | STOCHASTIC | MIXED
```

### Raw Measures (also in typology.parquet, used for window computation)
```
seasonal_period     Float64   Detected period in samples (from ACF peaks). Null if none.
dominant_freq       Float64   Dominant frequency from FFT (cycles per sample). Null if none.
acf_half_life       Float64   First lag where |ACF| < 0.5. Null if never crossed.
acf_decay_lag       Float64   First lag where |ACF| < 1/e. Null if never crossed.
acf_decayed         bool      True if ACF crossed threshold. False if persistent.
n_samples           UInt32    Total samples for this signal.
```

### PRISM's Use of typology.parquet
PRISM does NOT classify. PRISM may read typology.parquet for:
- Filtering constant signals (continuity = CONSTANT → skip)
- Validating manifest engine selections if needed
- Passing typology fields through to output for ORTHON reporting

PRISM must NEVER write to typology.parquet or modify its contents.

---

## manifest.yaml Schema (v2.4)

The manifest is ORTHON's complete order to PRISM. PRISM executes exactly
what the manifest says. No more, no less.

```yaml
version: "2.4"
job_id: "orthon-20260202-143052"

paths:
  observations: "observations.parquet"
  typology: "typology.parquet"
  output_dir: "output/"

summary:
  total_signals: 14
  active_signals: 11
  constant_signals: 3
  all_signal_engines:
    - crest_factor
    - entropy
    - harmonics
    - hurst
    - kurtosis
    - lyapunov
    - rate_of_change
    - skewness
    - spectral
  all_rolling_engines:
    - rolling_crest_factor
    - rolling_entropy
    - rolling_kurtosis
    - rolling_skewness
    - rolling_volatility

params:
  default_window: 128
  default_stride: 64

# Per-signal configuration
signals:
  temp_1:
    engines:
      - crest_factor
      - entropy
      - hurst
      - kurtosis
      - rate_of_change
      - skewness
    rolling_engines:
      - rolling_kurtosis
      - rolling_entropy
      - rolling_crest_factor
      - rolling_skewness
    window_size: 320
    window_method: acf_half_life      # how ORTHON determined the window
    window_confidence: high           # high | medium | low
    stride: 80                        # 75% overlap (non-stationary)
    derivative_depth: 2               # max derivative order
    eigenvalue_budget: 5              # max eigenvalues to compute
    typology:                         # identity card (read-only context)
      continuity: CONTINUOUS
      stationarity: NON_STATIONARY
      temporal_pattern: TRENDING
      memory: LONG_MEMORY
      complexity: MEDIUM
      distribution: GAUSSIAN
      amplitude: SMOOTH
      spectral: ONE_OVER_F
      volatility: HETEROSCEDASTIC
      determinism: MIXED
    visualizations:                   # ORTHON's viz recommendations (informational)
      - trend_overlay
    output_hints:                     # how PRISM should format output
      spectral:
        mode: summary                 # summary | per_bin
    unit_id: "entity_001"             # pass-through

  vib_x:
    engines:
      - attractor
      - crest_factor
      - cycle_counting
      - entropy
      - frequency_bands
      - harmonics
      - kurtosis
      - lyapunov
      - skewness
      - spectral
    rolling_engines: []
    window_size: 480
    window_method: period
    window_confidence: high
    stride: 240
    derivative_depth: 1
    eigenvalue_budget: 5
    typology:
      continuity: CONTINUOUS
      stationarity: STATIONARY
      temporal_pattern: PERIODIC
      memory: SHORT_MEMORY
      complexity: MEDIUM
      distribution: HEAVY_TAILED
      amplitude: IMPULSIVE
      spectral: HARMONIC
      volatility: HOMOSCEDASTIC
      determinism: DETERMINISTIC
    visualizations:
      - waterfall
      - phase_portrait
      - recurrence
      - spectral_density
    output_hints:
      spectral:
        mode: per_bin               # waterfall-ready output
      harmonics:
        n_harmonics: 5
        include_thd: true
      attractor:
        mode: trajectory            # full trajectory for phase portrait

# Constant signals — PRISM skips these entirely
skip_signals:
  - valve_status
  - mode_flag
  - config_id

# Pair engines — run on all active signal pairs
pair_engines:
  - granger
  - transfer_entropy

symmetric_pair_engines:
  - correlation
  - mutual_info
  - cointegration
```

---

## How PRISM Reads the Manifest

### Step 1: Load and Validate
```python
manifest = yaml.safe_load(open('manifest.yaml'))
assert manifest['version'] == '2.0'
```

### Step 2: Skip Constants
```python
skip = set(manifest.get('skip_signals', []))
```

### Step 3: Per-Signal Execution
For each signal in `manifest['signals']`:
1. Read `engines` list → run those signal-level engines
2. Read `rolling_engines` list → run those rolling engines
3. Use `window_size` and `stride` for windowing
4. Use `derivative_depth` for how many derivatives to compute
5. Use `eigenvalue_budget` for SVD truncation in state_geometry
6. Use `output_hints` to configure engine output format

### Step 4: Pair Execution
Run `pair_engines` on all ordered pairs of active signals.
Run `symmetric_pair_engines` on all unordered pairs.

### Step 5: Write Output
PRISM writes parquet files to `manifest['paths']['output_dir']`.

---

## Engine Names Are Canonical

Every engine name in the manifest maps to exactly one Python module in PRISM.
If PRISM doesn't recognize an engine name, it logs a warning and skips it.
PRISM never invents engines that aren't in the manifest.

### Signal Engines (signal_vector/signal/)
```
kurtosis          skewness          crest_factor
entropy           hurst             spectral
harmonics         frequency_bands   lyapunov
garch             attractor         dmd
pulsation_index   rate_of_change    time_constant
cycle_counting    basin             lof
envelope          rms               peak
```

### Rolling Engines (signal_vector/rolling/)
```
rolling_kurtosis      rolling_skewness      rolling_entropy
rolling_crest_factor  rolling_hurst         rolling_lyapunov
rolling_volatility    rolling_pulsation
rolling_rms           rolling_mean          rolling_std
rolling_range         rolling_envelope
derivatives           manifold              stability
```

### Pair Engines (geometry_pairwise/)
```
granger               transfer_entropy
```

### Symmetric Pair Engines (geometry_pairwise/)
```
correlation           mutual_info           cointegration
```

### Deprecated Engines (still present, will be removed)
```
rms    peak    envelope    rolling_rms    rolling_mean
rolling_std    rolling_range    rolling_envelope
```

---

## output_hints Reference

Output hints tell PRISM HOW to produce output, not WHAT to compute.
The engine runs the same math either way — hints control output format.

### spectral
```yaml
spectral:
  mode: per_bin     # Output amplitude per frequency bin per window
                    # → Enables waterfall chart in explorer
  # OR
  mode: summary     # Output summary metrics only (centroid, entropy, flatness)
                    # → Standard scalar features
```

### harmonics
```yaml
harmonics:
  n_harmonics: 5        # Track this many harmonics (default 3)
  include_thd: true     # Include Total Harmonic Distortion
```

### attractor
```yaml
attractor:
  mode: trajectory      # Output full reconstructed trajectory
                        # → Enables phase portrait in explorer
  # OR
  mode: summary         # Output dimension, Lyapunov only
```

### garch
```yaml
garch:
  mode: full            # Output conditional variance series
                        # → Enables volatility overlay
  # OR
  mode: basic           # Output GARCH params only (alpha, beta, omega)
```

---

## Visualization Recommendations

The `visualizations` field is informational. PRISM does not act on it.
It exists so the explorer (ORTHON's static HTML) knows what charts to offer.

| Visualization | When ORTHON Recommends It |
|---------------|--------------------------|
| waterfall | PERIODIC/QUASI_PERIODIC or HARMONIC + spectral engine present |
| phase_portrait | CHAOTIC or DETERMINISTIC + attractor engine present |
| trend_overlay | TRENDING or NON_STATIONARY + rate_of_change present |
| recurrence | DETERMINISTIC/MIXED + attractor present |
| volatility_map | HETEROSCEDASTIC/VOLATILITY_CLUSTERING + rolling_volatility present |
| spectral_density | Any spectral character + spectral engine present |

PRISM produces the data. ORTHON decides the view.

---

## Window Determination

ORTHON determines the window from typology. PRISM uses what the manifest says.

| method | How ORTHON Computed It |
|--------|----------------------|
| period | 4 x seasonal_period (capture 4 complete cycles) |
| acf_half_life | 4 x acf_half_life (capture decorrelation) |
| long_memory | 8 x acf_decay_lag (ACF never decayed) |
| non_stationary_cap | n_samples / 10 (keep window local) |
| default | 128 (nothing else available) |

All windows clamped to [32, 2048] and never exceed n_samples / 2.

PRISM trusts these values. If a window seems wrong, the fix is in
ORTHON's typology or window_recommender, not in PRISM.

---

## Boundary Rules

1. **PRISM reads manifest.yaml. PRISM never reads ORTHON source code.**
2. **PRISM computes numbers. PRISM never classifies.**
3. **PRISM never writes to typology.parquet.**
4. **PRISM never modifies observations.parquet.**
5. **If manifest says run engine X, PRISM runs engine X.**
6. **If manifest says skip signal Y, PRISM skips signal Y.**
7. **If PRISM finds a bug in the manifest, PRISM logs it and continues.**
8. **PRISM does not second-guess ORTHON's window, engine, or typology decisions.**

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025 | 4-category typology, flat engine lists |
| 2.0 | 2026-02 | 10-dimension typology, per-signal config, output hints, visualizations |
| 2.2 | 2026-02 | Inclusive engine gating ("if maybe, run it"), nested cohorts |
| 2.4 | 2026-02 | System window, multi-scale representation (spectral vs trajectory) |
