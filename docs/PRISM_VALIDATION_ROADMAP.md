# PRISM Validation Roadmap

## Hard Science Benchmarks for Physics-Based Validation

**Goal**: Validate PRISM on problems with known physics/equations where ground truth is analytical or empirical — not ML benchmarks, but real science published in hard science literature.

**Why**: PRISM is not machine learning. It's measurement. Validation should come from physics, not ML competitions.

---

## Current Status

### Completed
- [x] NASA C-MAPSS FD001 (turbofan RUL) — **RMSE 4.37** vs ML SOTA 6.62
- [x] Tennessee Eastman Process (chemical process) — Control structure discovered

### In Progress
- [ ] NASA C-MAPSS FD002, FD003, FD004
- [ ] Additional NASA prognostics datasets

---

## Benchmark Categories

### Category 1: Classical Physics Systems (Chaotic Dynamics)

Known equations, testable dynamics, analytical ground truth.

#### 1.1 Lorenz System ⭐ PRIORITY
```
Equations:
  dx/dt = σ(y - x)
  dy/dt = x(ρ - z) - y  
  dz/dt = xy - βz

Parameters: σ=10, ρ=28, β=8/3

Ground Truth: Analytical solution, known attractor structure
```

**PRISM Tests**:
- Can PRISM detect lobe transitions (regime changes)?
- Does field topology reflect attractor structure?
- Do sources/sinks map to physical dynamics?

**Data**: Generate from equations (trivial)

**Why Perfect**: Gradual dynamics, known regime transitions, pure physics

---

#### 1.2 Rössler Attractor
```
Equations:
  dx/dt = -y - z
  dy/dt = x + ay
  dz/dt = b + z(x - c)

Parameters: a=0.2, b=0.2, c=5.7
```

**PRISM Tests**:
- Different attractor structure than Lorenz
- Single-scroll vs double-scroll dynamics

---

#### 1.3 Double Pendulum
```
Classical mechanics, chaotic motion
Exact Lagrangian equations known
Energy conservation testable
```

**PRISM Tests**:
- Can PRISM detect transition to chaos?
- Does field potential track energy?

---

#### 1.4 Lotka-Volterra (Predator-Prey) ⭐ PRIORITY
```
Equations:
  dx/dt = αx - βxy  (prey)
  dy/dt = δxy - γy  (predator)

Parameters: α=1.1, β=0.4, δ=0.1, γ=0.4
```

**PRISM Tests**:
- Oscillatory regime cycles
- Can PRISM detect population collapse risk?
- Source/sink classification of species

**Why Good**: Biology application, regime cycles, interpretable

---

### Category 2: Chemical Kinetics

Real experimental data with known rate equations.

#### 2.1 NIST Chemical Kinetics Database ⭐ PRIORITY
```
Source: kinetics.nist.gov
Data: 38,000+ reaction records, 11,700+ reactant pairs
Format: Rate parameters A, n, Ea/R

Ground Truth: k = A × (T/298)^n × exp(-Ea/RT)
```

**PRISM Tests**:
- Given concentration time series, can PRISM recover rate behavior?
- Do field dynamics match Arrhenius kinetics?

---

#### 2.2 ReSpecTh Database
```
Source: Nature Scientific Data (2025)
Focus: Combustion kinetics, gas-phase reactions
Data: Experimental reaction data with uncertainty
```

---

#### 2.3 SABIO-RK (Biochemical Reactions)
```
Source: sabio.h-its.org
Focus: Enzyme kinetics
Ground Truth: Michaelis-Menten dynamics

v = Vmax × [S] / (Km + [S])
```

**PRISM Tests**:
- Can PRISM detect substrate saturation regimes?
- Does field topology reflect enzyme kinetics?

---

### Category 3: PDE Systems (Physics Simulations)

#### 3.1 The Well (PolymathicAI)
```
Source: github.com/PolymathicAI/the_well
Published: NeurIPS 2024
Size: 15TB total
```

**Relevant Datasets**:

| Dataset | Physics | PRISM Relevance |
|---------|---------|-----------------|
| Gray-Scott Reaction-Diffusion | Self-assembly, patterns | Regime formation |
| Euler Equations | Compressible gas, shocks | Stress propagation |
| Shallow Water | Fluid waves | Wave dynamics |
| MHD Turbulence | Magnetohydrodynamics | Coupled system |

---

#### 3.2 PDEBench
```
Source: NeurIPS 2022
Focus: Time-dependent PDEs with known equations
```

**Datasets**:
- 1D Advection equation
- 1D Burgers equation (shock formation)
- 2D Diffusion-Reaction
- 2D Shallow Water
- Navier-Stokes

All have **known governing equations** with numerical ground truth.

---

### Category 4: Real Experimental Data

#### 4.1 Seismic Data (USGS)
```
Source: earthquake.usgs.gov
Data: Stress measurements, seismograms
Physics: Stress buildup → release (earthquake)
```

**PRISM Tests**:
- Source/sink dynamics in fault systems
- Stress accumulation detection
- Pre-earthquake regime changes

---

#### 4.2 NOAA Climate Stations
```
Source: noaa.gov
Data: Temperature, pressure, humidity time series
Physics: Seasonal transitions, weather patterns
```

**PRISM Tests**:
- Gradual seasonal regime transitions
- Climate pattern detection

**Note**: Keep purely physical. Avoid anything that could be construed as climate risk/finance.

---

#### 4.3 PhysioNet (Medical Time Series)
```
Source: physionet.org
Data: ECG, EEG, ICU monitoring
Physics: Physiological dynamics
```

**Relevant Datasets**:
- MIMIC-III (ICU patient data)
- MIT-BIH Arrhythmia
- EEG Motor Movement

**PRISM Tests**:
- Patient deterioration detection
- Cardiac regime changes
- Gradual physiological decline

---

#### 4.4 Nuclear Decay Data
```
Source: IAEA databases
Physics: Exponential decay (known analytical solution)

N(t) = N₀ × exp(-λt)
```

**PRISM Tests**:
- Can PRISM field potential track decay?
- Simplest possible physics validation

---

## Priority Order

### Tier 1: Perfect Fit (Start Here)

| # | Benchmark | Why |
|---|-----------|-----|
| 1 | **Lorenz System** | Known equations, regime transitions, generate data |
| 2 | **Lotka-Volterra** | Biology, cycles, interpretable |
| 3 | **NIST Chemical Kinetics** | Real chemistry, known rate laws |

### Tier 2: Good Fit

| # | Benchmark | Why |
|---|-----------|-----|
| 4 | Seismic (USGS) | Real data, stress dynamics |
| 5 | PhysioNet ICU | Patient deterioration |
| 6 | Gray-Scott (The Well) | Pattern formation |

### Tier 3: Extended Validation

| # | Benchmark | Why |
|---|-----------|-----|
| 7 | Rössler Attractor | Different chaos structure |
| 8 | PDEBench Burgers | Shock formation |
| 9 | Double Pendulum | Classical mechanics |

---

## Validation Protocol

For each benchmark:

### Step 1: Data Acquisition
- Download or generate data
- Document source and parameters
- No preprocessing (raw PRISM)

### Step 2: Run PRISM
- Standard PRISM pipeline
- No domain-specific tuning
- Document exact configuration

### Step 3: Compare to Ground Truth
- Known equations: Compare field topology to analytical structure
- Known events: Compare PRISM detection to labeled events
- Known parameters: Compare PRISM features to physical parameters

### Step 4: Document Results
- PRISM output
- Ground truth comparison
- Physics interpretation
- Limitations observed

---

## Success Criteria

### For a benchmark to "pass":

1. **Field topology makes physical sense**
   - Sources/sinks align with physical interpretation
   - Stress propagation matches known dynamics

2. **Quantitative agreement** (where applicable)
   - Correlation with ground truth > 0.7
   - Or: Correct regime detection > 80%

3. **No domain tuning required**
   - Same PRISM code as other domains
   - Physics emerges from data

---

## The Story (If This Works)

```
"PRISM validated on:

 - NASA turbofan data (engineering)
 - Tennessee Eastman (chemical engineering)  
 - Lorenz system (physics)
 - Lotka-Volterra (biology)
 - Chemical kinetics (chemistry)
 - Seismic data (geophysics)

 Same math. No domain tuning. Physics emerges.

 This is not a machine learning method.
 This is a measurement framework."
```

**That's the Santa Fe talk.**
**That's the arXiv paper.**
**That's the NASA SBIR.**

---

## Notes

### What We're NOT Doing
- ML benchmark competitions (different game)
- Finance/economics data (Fidelity concerns)
- Anything requiring domain-specific preprocessing

### What We ARE Doing
- Physics validation with known ground truth
- Cross-domain proof of concept
- Interpretable results that match theory

---

## References

### Databases
- NIST Chemical Kinetics: https://kinetics.nist.gov
- The Well: https://github.com/PolymathicAI/the_well
- PDEBench: NeurIPS 2022
- PhysioNet: https://physionet.org
- USGS Earthquakes: https://earthquake.usgs.gov
- ReSpecTh: Nature Scientific Data (2025)
- SABIO-RK: https://sabio.h-its.org

### Papers
- Lorenz (1963): "Deterministic Nonperiodic Flow"
- Lotka-Volterra: Standard ecology textbooks
- PDEBench: "An Extensive Benchmark for Scientific ML" (NeurIPS 2022)
- The Well: "A Large-Scale Collection of Diverse Physics Simulations" (NeurIPS 2024)

---

## Next Steps

1. [ ] Generate Lorenz system data
2. [ ] Run PRISM on Lorenz
3. [ ] Document field topology vs attractor structure
4. [ ] If successful, proceed to Lotka-Volterra
5. [ ] Build validation table across all domains

---

*Last Updated: January 2025*
*Author: Jason Rudder / PRISM Observatory*
