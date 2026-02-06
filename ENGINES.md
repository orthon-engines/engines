# PRISM Engines & Primitives Reference

> Mathematical foundations for signal processing, dynamical systems analysis, and predictive diagnostics.

---

## Table of Contents

1. [Signal Engines](#signal-engines)
2. [State Engines](#state-engines)
3. [Pairwise Engines](#pairwise-engines)
4. [Dynamics Engines](#dynamics-engines)
5. [Primitives Reference](#primitives-reference)
6. [Pipeline Stage Mapping](#pipeline-stage-mapping)

---

## Signal Engines

### Distribution Shape

#### Kurtosis
**Stage**: `01_signal_vector`

Measures tail heaviness (peakedness) of the distribution.

$$\gamma_2 = \frac{E[(X - \mu)^4]}{\sigma^4} - 3$$

| Value | Interpretation |
|-------|----------------|
| $\gamma_2 = 0$ | Gaussian (mesokurtic) |
| $\gamma_2 > 0$ | Heavy tails (leptokurtic) |
| $\gamma_2 < 0$ | Light tails (platykurtic) |

#### Skewness
**Stage**: `01_signal_vector`

Measures asymmetry of the distribution.

$$\gamma_1 = \frac{E[(X - \mu)^3]}{\sigma^3}$$

| Value | Interpretation |
|-------|----------------|
| $\gamma_1 > 0$ | Right-skewed (long right tail) |
| $\gamma_1 = 0$ | Symmetric |
| $\gamma_1 < 0$ | Left-skewed (long left tail) |

#### Crest Factor
**Stage**: `01_signal_vector`

Ratio of peak to RMS — measures impulsiveness.

$$CF = \frac{\max|X|}{\sqrt{E[X^2]}}$$

| Value | Interpretation |
|-------|----------------|
| $CF \approx 1.41$ | Sine wave |
| $CF \approx 3$ | Gaussian noise |
| $CF \gg 3$ | Impulsive signal |

---

### Spectral Analysis

#### Spectral Entropy
**Stage**: `01_signal_vector`

Measures spectral flatness — how spread the energy is across frequencies.

$$H_{spectral} = -\sum_{i} p_i \log(p_i) \bigg/ \log(N)$$

where $p_i = P_i / \sum_j P_j$ is the normalized power at frequency bin $i$.

| Value | Interpretation |
|-------|----------------|
| $H \approx 0$ | Pure tone (concentrated) |
| $H \approx 1$ | White noise (flat spectrum) |

#### Spectral Centroid
**Stage**: `01_signal_vector`

Center of mass of the power spectrum — the "brightness" of the signal.

$$f_c = \frac{\sum_f f \cdot P(f)}{\sum_f P(f)}$$

#### Spectral Slope
**Stage**: `01_signal_vector`

Log-log slope of power vs frequency — characterizes noise color.

$$\log_{10}(P) \approx \alpha \cdot \log_{10}(f) + \beta$$

| Slope $\alpha$ | Noise Type |
|----------------|------------|
| $\alpha \approx 0$ | White noise |
| $\alpha \approx -1$ | Pink noise (1/f) |
| $\alpha \approx -2$ | Red/Brownian noise |

#### Dominant Frequency
**Stage**: `01_signal_vector`

Frequency with maximum power.

$$f_{dom} = \arg\max_f |FFT(x)|^2$$

---

### Harmonics

#### Total Harmonic Distortion (THD)
**Stage**: `01_signal_vector`

Measures harmonic purity of periodic signals.

$$THD = \frac{\sqrt{\sum_{k=2}^{N} H_k^2}}{H_1} \times 100\%$$

where $H_k$ is the amplitude of the $k$-th harmonic.

| THD | Interpretation |
|-----|----------------|
| $< 5\%$ | Pure tone |
| $> 10\%$ | Significant distortion |

#### Signal-to-Noise Ratio (SNR)
**Stage**: `01_signal_vector`

$$SNR_{dB} = 10 \cdot \log_{10}\left(\frac{P_{signal}}{P_{noise}}\right)$$

---

### Memory & Long-Range Dependence

#### Hurst Exponent
**Stage**: `01_signal_vector`

Measures long-range dependence via rescaled range (R/S) analysis.

$$H = \frac{\log(R/S)}{\log(N)}$$

where $R/S = \frac{\max(\text{cumsum}) - \min(\text{cumsum})}{\sigma}$

| Value | Interpretation |
|-------|----------------|
| $H < 0.5$ | Mean-reverting (anti-persistent) |
| $H = 0.5$ | Random walk |
| $H > 0.5$ | Trending (persistent) |

#### Detrended Fluctuation Analysis (DFA)
**Stage**: `01_signal_vector`

More robust to nonstationarity than R/S method.

$$F(n) \sim n^\alpha$$

| $\alpha$ | Interpretation |
|----------|----------------|
| $\alpha < 0.5$ | Anti-correlated |
| $\alpha = 0.5$ | White noise |
| $\alpha = 1.0$ | Pink noise (1/f) |
| $\alpha = 1.5$ | Brownian motion |

#### ACF Half-Life
**Stage**: `01_signal_vector`

First lag where autocorrelation drops below 0.5.

$$\tau_{1/2} = \min\{\tau : ACF(\tau) < 0.5\}$$

---

### Complexity & Entropy

#### Sample Entropy
**Stage**: `01_signal_vector`

Measures signal irregularity — conditional probability that similar patterns remain similar.

$$SampEn(m, r) = -\log\frac{\phi^{m+1}(r)}{\phi^m(r)}$$

where $\phi^m(r)$ counts pattern matches of length $m$ within tolerance $r$ (typically $0.2\sigma$).

| Value | Interpretation |
|-------|----------------|
| Low | Regular, predictable |
| High | Irregular, complex |

#### Permutation Entropy
**Stage**: `01_signal_vector`

Complexity based on ordinal patterns.

$$H_{perm} = -\sum_{\pi} p(\pi) \log_2 p(\pi) \bigg/ \log_2(m!)$$

where $p(\pi)$ is the probability of ordinal pattern $\pi$ with embedding dimension $m$.

---

### Trend Analysis

#### Linear Trend
**Stage**: `01_signal_vector`

$$y = m \cdot t + b$$

**Outputs**: `trend_slope` ($m$), `trend_r2` ($R^2$), `detrend_std`

#### CUSUM Range
**Stage**: `01_signal_vector`

Cumulative sum of deviations — measures trend persistence.

$$CUSUM_t = \sum_{i=1}^{t} (y_i - \bar{y})$$

$$CUSUM_{range} = \frac{\max(CUSUM) - \min(CUSUM)}{\sigma_y}$$

#### Variance Growth
**Stage**: `01_signal_vector`

$$\text{Var}(\text{cumsum}(y)) \propto n^\alpha$$

| $\alpha$ | Interpretation |
|----------|----------------|
| $\alpha \approx 1$ | Stationary |
| $\alpha > 1$ | Trending / non-stationary |

---

### Stationarity Tests

#### Augmented Dickey-Fuller (ADF)
**Stage**: `01_signal_vector`

Tests for unit root (non-stationarity).

$$\Delta y_t = \alpha y_{t-1} + \sum_i \beta_i \Delta y_{t-i} + \varepsilon_t$$

- $H_0$: $\alpha = 0$ (unit root, non-stationary)
- $H_1$: $\alpha < 0$ (stationary)

| p-value | Interpretation |
|---------|----------------|
| $p < 0.05$ | Stationary |
| $p > 0.05$ | Non-stationary |

#### Variance Ratio Test
**Stage**: `01_signal_vector`

$$VR(q) = \frac{\text{Var}(y_q)}{q \cdot \text{Var}(y_1)}$$

| VR(q) | Interpretation |
|-------|----------------|
| $VR = 1$ | Random walk |
| $VR < 1$ | Mean-reverting |
| $VR > 1$ | Trending |

---

### Dynamical Systems (Signal-Level)

#### Lyapunov Exponent
**Stage**: `01_signal_vector`, `11_dynamics`

Measures sensitive dependence on initial conditions (chaos).

$$\lambda_1 = \lim_{t \to \infty} \frac{1}{t} \log\frac{d(t)}{d_0}$$

where $d(t)$ is the divergence of nearby trajectories.

| $\lambda$ | Interpretation |
|-----------|----------------|
| $\lambda > 0$ | Chaotic |
| $\lambda \approx 0$ | Quasi-periodic |
| $\lambda < 0$ | Stable / converging |

**Rosenstein Algorithm**:
$$\lambda \approx \frac{1}{K} \sum_{j=1}^{K} \log\frac{d_j(k)}{d_j(0)}$$

#### Correlation Dimension
**Stage**: `01_signal_vector`

Fractal dimension of the attractor.

$$D_2 = \lim_{r \to 0} \frac{\log C(r)}{\log r}$$

where $C(r) = \frac{2}{N(N-1)} \sum_{i<j} \Theta(r - \|x_i - x_j\|)$ is the correlation sum.

---

### Recurrence Quantification Analysis (RQA)

#### Recurrence Matrix
**Stage**: `01_signal_vector`

$$R_{ij} = \Theta(\varepsilon - \|x_i - x_j\|)$$

where $\varepsilon$ is typically 10% of $\sigma$.

#### Recurrence Rate
**Stage**: `01_signal_vector`

$$RR = \frac{1}{N^2} \sum_{i,j} R_{ij}$$

#### Determinism
**Stage**: `01_signal_vector`

Fraction of recurrence points forming diagonal lines (predictability).

$$DET = \frac{\sum_{l \geq l_{min}} l \cdot P(l)}{\sum_{i,j} R_{ij}}$$

| DET | Interpretation |
|-----|----------------|
| Low | Stochastic |
| High | Deterministic |

---

### Volatility Modeling

#### GARCH(1,1)
**Stage**: `01_signal_vector`

Models time-varying volatility (heteroscedasticity).

$$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

| Parameter | Meaning |
|-----------|---------|
| $\omega$ | Baseline variance |
| $\alpha$ | Shock persistence |
| $\beta$ | Volatility persistence |
| $\alpha + \beta$ | Total persistence (< 1 for stationarity) |

---

### Dynamic Mode Decomposition (DMD)

#### DMD
**Stage**: `01_signal_vector`

Extracts spatiotemporal modes from time series.

$$X' \approx A \cdot X$$

$$A_{reduced} = U_r^T X' V_r \Sigma_r^{-1}$$

**Eigenvalue interpretation**:
- $|\lambda| < 1$: Stable (decaying)
- $|\lambda| = 1$: Neutral (oscillatory)
- $|\lambda| > 1$: Unstable (growing)

**Continuous-time**:
$$\omega = \frac{\log(\lambda)}{\Delta t}$$
- $\text{Re}(\omega)$: Growth rate
- $\text{Im}(\omega) / 2\pi$: Frequency

---

## State Engines

### Centroid (State Vector)
**Stage**: `02_state_vector`

WHERE the system is in feature space.

$$\text{centroid} = \frac{1}{N} \sum_{i=1}^{N} \vec{x}_i$$

**Dispersion metrics**:
- `mean_distance`: $\bar{d} = \frac{1}{N} \sum_i \|\vec{x}_i - \text{centroid}\|$
- `max_distance`: $d_{max} = \max_i \|\vec{x}_i - \text{centroid}\|$

---

### Eigendecomposition (State Geometry)
**Stage**: `03_state_geometry`

HOW the system is distributed — the SHAPE.

$$\text{Centered} = U \Sigma V^T$$

$$\lambda_k = \frac{\sigma_k^2}{N-1}$$

#### Effective Dimension
**Stage**: `03_state_geometry`

Number of "essential" dimensions (Rényi entropy approximation).

$$D_{eff} = \frac{(\sum_k \lambda_k)^2}{\sum_k \lambda_k^2}$$

| $D_{eff}$ | Interpretation |
|-----------|----------------|
| $D_{eff} = D$ | Uniform variance (isotropic) |
| $D_{eff} = 1$ | One dominant direction (collapsed) |

> **Key insight**: $D_{eff}$ shows 63% importance in RUL prediction (Avery Rudder).

#### Eigenvalue Entropy
**Stage**: `03_state_geometry`

$$H_\lambda = -\sum_k p_k \log(p_k) \bigg/ \log(K)$$

where $p_k = \lambda_k / \sum_j \lambda_j$.

#### Condition Number
**Stage**: `03_state_geometry`

$$\kappa = \frac{\lambda_{max}}{\lambda_{min}}$$

High $\kappa$ indicates numerical instability or near-singular covariance.

---

## Pairwise Engines

### Correlation
**Stage**: `10_pairwise`

#### Pearson Correlation

$$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

#### Cross-Correlation

$$R_{XY}(\tau) = E[(X_t - \mu_X)(Y_{t+\tau} - \mu_Y)] / (\sigma_X \sigma_Y)$$

#### Mutual Information
**Stage**: `10_pairwise`

Total dependence (linear + nonlinear).

$$I(X; Y) = \sum_{x,y} p(x,y) \log\frac{p(x,y)}{p(x)p(y)}$$

---

### Causality
**Stage**: `12_information_flow`

#### Granger Causality

Tests if past $X$ improves prediction of $Y$.

- **Restricted**: $Y_t \sim Y_{t-1}, Y_{t-2}, \ldots$
- **Unrestricted**: $Y_t \sim Y_{t-1}, \ldots + X_{t-1}, \ldots$

$$F = \frac{(SSR_r - SSR_u) / k}{SSR_u / (n - k_u)}$$

High $F$, low $p$ → $X$ Granger-causes $Y$.

#### Transfer Entropy

Model-free information flow.

$$TE_{X \to Y} = I(Y_{t+1}; X_t \mid Y_t)$$

$$TE_{normalized} = \frac{TE}{H(Y)}$$

---

## Dynamics Engines

### Critical Slowing Down
**Stage**: `11_dynamics`

Early warning signals for bifurcations (Scheffer 2009).

As system approaches tipping point:
1. **Autocorrelation increases** (slower recovery)
2. **Variance increases** (larger fluctuations)
3. **Skewness grows** (asymmetric perturbations)

#### Recovery Rate

$$r = -\ln(\rho_1) \quad \text{for } |\rho_1| < 1$$

Lower recovery rate → approaching bifurcation.

#### CSD Score

Weighted composite (0-1 scale):

| Component | Weight |
|-----------|--------|
| Autocorr level | 0.30 |
| Autocorr trend | 0.25 |
| Variance trend | 0.25 |
| Variance ratio | 0.20 |

> **Warning**: CSD detects B-tipping (bifurcation-induced). R-tipping (rate-induced) may have NO warning.

---

### Break Detection
**Stage**: `05_breaks`

Detects discontinuities: steps (Heaviside), impulses (Dirac).

#### CUSUM Detector (Page's test)

$$C^+ = \max(0, C^+ + (x - \mu)/\sigma - 0.5)$$

#### Break Characterization

| Metric | Formula |
|--------|---------|
| Magnitude | $\|step\| / MAD$ |
| Sharpness | $magnitude / duration$ |
| SNR | $raw\_magnitude / noise\_floor$ |

---

## Primitives Reference

### Individual Signal

| Primitive | Formula | Output |
|-----------|---------|--------|
| `mean` | $\mu = \frac{1}{n}\sum x_i$ | float |
| `std` | $\sigma = \sqrt{\frac{1}{n-1}\sum(x_i - \mu)^2}$ | float |
| `rms` | $\sqrt{E[X^2]}$ | float |
| `derivative` | $dy/dt$ (central diff) | array |
| `integral` | $\int y \, dt$ (trapezoidal) | array |
| `curvature` | $\kappa = \frac{|d^2y/dt^2|}{(1 + (dy/dt)^2)^{3/2}}$ | array |

### Information Theory

| Primitive | Formula | Range |
|-----------|---------|-------|
| `shannon_entropy` | $H = -\sum p(x) \log p(x)$ | $[0, \log n]$ |
| `renyi_entropy` | $H_\alpha = \frac{1}{1-\alpha}\log\sum p(x)^\alpha$ | varies |
| `kl_divergence` | $D_{KL} = \sum p(x) \log\frac{p(x)}{q(x)}$ | $[0, \infty)$ |
| `js_divergence` | $D_{JS} = \frac{1}{2}(D_{KL}(P\|M) + D_{KL}(Q\|M))$ | $[0, 1]$ |

### Embedding

| Primitive | Description |
|-----------|-------------|
| `time_delay_embedding` | $\vec{x}(i) = [x(i), x(i+\tau), \ldots, x(i+(d-1)\tau)]$ |
| `optimal_delay` | First minimum of mutual information |
| `optimal_dimension` | False nearest neighbors (FNN) threshold |

### Topology

| Primitive | Description |
|-----------|-------------|
| `persistence_diagram` | Vietoris-Rips filtration: $H_0$ (components), $H_1$ (loops), $H_2$ (voids) |
| `betti_numbers` | Counts of topological features |
| `wasserstein_distance` | Distance between persistence diagrams |

---

## Pipeline Stage Mapping

```
observations.parquet
        │
        ▼
┌───────────────────┐
│ 01_signal_vector  │ ← Per-signal features (kurtosis, hurst, entropy, spectral, ...)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ 02_state_vector   │ ← System centroid (WHERE)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ 03_state_geometry │ ← Eigenvalues, effective_dim (SHAPE)
└───────────────────┘
        │
        ├──────────────────────────────┐
        ▼                              ▼
┌───────────────────┐        ┌───────────────────┐
│ 04_cohorts        │        │ 05_breaks         │
│ (aggregation)     │        │ (discontinuities) │
└───────────────────┘        └───────────────────┘
        │
        ▼
┌───────────────────┐
│ 10_pairwise       │ ← Correlation, mutual info
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ 11_dynamics       │ ← Lyapunov, RQA, CSD
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ 12_information    │ ← Granger, transfer entropy
│    _flow          │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ 13_topology       │ ← Network structure
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ 21_statistics     │ ← Summary stats (SQL)
└───────────────────┘
```

---

## Engine Index

| Engine | Stage | Key Outputs |
|--------|-------|-------------|
| statistics | 01 | kurtosis, skewness, crest_factor |
| spectral | 01 | spectral_entropy, spectral_slope, dominant_freq |
| harmonics | 01 | thd, fundamental_freq, harmonic_2x |
| hurst | 01 | hurst, dfa, hurst_r2 |
| complexity | 01 | sample_entropy, permutation_entropy |
| memory | 01 | acf_lag1, acf_half_life |
| trend | 01 | trend_slope, trend_r2, cusum_range |
| adf_stat | 01 | adf_stat, adf_pvalue |
| variance_ratio | 01 | variance_ratio, variance_ratio_stat |
| lyapunov | 01, 11 | lyapunov, embedding_dim |
| rqa | 01 | recurrence_rate, determinism |
| garch | 01 | garch_alpha, garch_beta, garch_persistence |
| dmd | 01 | dmd_dominant_freq, dmd_growth_rate |
| centroid | 02 | centroid, mean_distance |
| eigendecomp | 03 | eigenvalues, effective_dim, condition_number |
| correlation | 10 | correlation, max_xcorr, mutual_info |
| causality | 12 | granger_f, transfer_entropy |
| csd | 11 | csd_score, recovery_rate |
| breaks | 05 | magnitude, sharpness, snr |

---

## References

- Rosenstein, M. T., Collins, J. J., & De Luca, C. J. (1993). A practical method for calculating largest Lyapunov exponents from small data sets.
- Scheffer, M., et al. (2009). Early-warning signals for critical transitions. Nature.
- Grassberger, P., & Procaccia, I. (1983). Characterization of strange attractors. Physical Review Letters.
- Takens, F. (1981). Detecting strange attractors in turbulence. Lecture Notes in Mathematics.
- Webber, C. L., & Zbilut, J. P. (1994). Dynamical assessment of physiological systems and states using recurrence plot strategies. Journal of Applied Physiology.

---

*Generated for PRISM v2.2.0*
