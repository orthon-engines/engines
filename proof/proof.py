"""
PRISM Proof Tool - Show your work.

For any (entity, timestamp), shows:
1. Raw sensor values
2. Window of data used
3. Each engine's formula (LaTeX)
4. Step-by-step calculation
5. Final output

"This isn't a black box. Verify it yourself."

Usage:
    from proof import Proof
    
    proof = Proof(observations, vector_df)
    proof.explain(entity_id="1", timestamp=50.0, signal="s_11", engine="hurst")
    proof.to_latex()  # Export for paper/presentation
"""

import numpy as np
import polars as pl
from typing import Optional, Dict, List, Any
from dataclasses import dataclass


@dataclass
class EngineProof:
    """Proof of calculation for one engine."""
    engine: str
    signal: str
    entity_id: str
    timestamp: float
    window_start: float
    window_end: float
    raw_values: np.ndarray
    formula_latex: str
    steps: List[Dict[str, Any]]
    result: float


# =============================================================================
# LATEX FORMULAS FOR EACH ENGINE
# =============================================================================

FORMULAS = {
    'hurst': {
        'name': 'Hurst Exponent',
        'formula': r'''H = \frac{\log(R/S)}{\log(n)}''',
        'description': 'Measures long-range dependence. H > 0.5 indicates persistence, H < 0.5 indicates mean reversion.',
        'full_derivation': r'''
\textbf{Hurst Exponent (Rescaled Range Analysis)}

Given a time series $X = \{x_1, x_2, \ldots, x_n\}$:

1. \textbf{Compute mean:}
   $$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

2. \textbf{Create mean-adjusted series:}
   $$Y_t = x_t - \bar{x}$$

3. \textbf{Compute cumulative deviation:}
   $$Z_t = \sum_{i=1}^{t} Y_i$$

4. \textbf{Compute range:}
   $$R = \max(Z_t) - \min(Z_t)$$

5. \textbf{Compute standard deviation:}
   $$S = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2}$$

6. \textbf{Rescaled range:}
   $$R/S = \frac{R}{S}$$

7. \textbf{Hurst exponent:}
   $$H = \frac{\log(R/S)}{\log(n)}$$

\textbf{Interpretation:}
\begin{itemize}
    \item $H = 0.5$: Random walk (no memory)
    \item $H > 0.5$: Persistent (trend-following)
    \item $H < 0.5$: Anti-persistent (mean-reverting)
\end{itemize}
'''
    },
    
    'entropy': {
        'name': 'Shannon Entropy',
        'formula': r'''H(X) = -\sum_{i=1}^{n} p_i \log_2(p_i)''',
        'description': 'Measures unpredictability/complexity of the signal distribution.',
        'full_derivation': r'''
\textbf{Shannon Entropy}

Given a time series $X = \{x_1, x_2, \ldots, x_n\}$:

1. \textbf{Discretize into bins:}
   Partition the range $[\min(X), \max(X)]$ into $k$ equal-width bins.

2. \textbf{Count occurrences:}
   $$n_i = \text{count of values in bin } i$$

3. \textbf{Compute probabilities:}
   $$p_i = \frac{n_i}{n}$$

4. \textbf{Shannon entropy:}
   $$H(X) = -\sum_{i=1}^{k} p_i \log_2(p_i)$$

   where $0 \log(0) \equiv 0$ by convention.

\textbf{Interpretation:}
\begin{itemize}
    \item $H = 0$: Perfectly predictable (all values in one bin)
    \item $H = \log_2(k)$: Maximum entropy (uniform distribution)
    \item Higher $H$: More complex/unpredictable signal
\end{itemize}

\textbf{Normalized entropy:}
$$H_{norm} = \frac{H(X)}{\log_2(k)}$$
'''
    },
    
    'garch': {
        'name': 'GARCH(1,1) Volatility',
        'formula': r'''\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2''',
        'description': 'Models time-varying volatility. Captures volatility clustering.',
        'full_derivation': r'''
\textbf{GARCH(1,1) Model}

Given returns $r_t = x_t - x_{t-1}$ (or log returns):

1. \textbf{Mean equation:}
   $$r_t = \mu + \epsilon_t$$
   where $\epsilon_t = \sigma_t z_t$ and $z_t \sim N(0,1)$

2. \textbf{Variance equation (GARCH):}
   $$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

3. \textbf{Parameter constraints:}
   \begin{itemize}
       \item $\omega > 0$ (base variance)
       \item $\alpha \geq 0$ (shock impact)
       \item $\beta \geq 0$ (persistence)
       \item $\alpha + \beta < 1$ (stationarity)
   \end{itemize}

4. \textbf{Estimation:}
   Parameters $(\omega, \alpha, \beta)$ estimated via Maximum Likelihood.

5. \textbf{Output:}
   Conditional variance $\sigma_t^2$ at each timestep.

\textbf{Interpretation:}
\begin{itemize}
    \item High $\alpha$: Volatility reacts strongly to shocks
    \item High $\beta$: Volatility is persistent
    \item $\alpha + \beta \approx 1$: Long memory in volatility
\end{itemize}
'''
    },
    
    'lyapunov': {
        'name': 'Lyapunov Exponent',
        'formula': r'''\lambda = \lim_{t \to \infty} \frac{1}{t} \ln \frac{|\delta(t)|}{|\delta_0|}''',
        'description': 'Measures sensitivity to initial conditions. Positive λ indicates chaos.',
        'full_derivation': r'''
\textbf{Lyapunov Exponent (Largest)}

Measures the rate of separation of infinitesimally close trajectories.

1. \textbf{Embed time series:}
   Create delay vectors $\mathbf{x}_i = (x_i, x_{i+\tau}, x_{i+2\tau}, \ldots, x_{i+(m-1)\tau})$
   
   where $\tau$ = delay, $m$ = embedding dimension.

2. \textbf{Find nearest neighbors:}
   For each $\mathbf{x}_i$, find nearest neighbor $\mathbf{x}_j$ where $|i - j| > \text{mean period}$.

3. \textbf{Track divergence:}
   $$d_i(k) = \|\mathbf{x}_{i+k} - \mathbf{x}_{j+k}\|$$

4. \textbf{Average log divergence:}
   $$\langle \ln d(k) \rangle = \frac{1}{N} \sum_{i=1}^{N} \ln d_i(k)$$

5. \textbf{Lyapunov exponent:}
   $$\lambda = \frac{d}{dk} \langle \ln d(k) \rangle$$
   
   (Slope of log divergence vs time)

\textbf{Interpretation:}
\begin{itemize}
    \item $\lambda > 0$: Chaotic (exponential divergence)
    \item $\lambda = 0$: Marginally stable
    \item $\lambda < 0$: Stable/periodic (trajectories converge)
\end{itemize}
'''
    },
    
    'spectral': {
        'name': 'Spectral Analysis',
        'formula': r'''S(f) = \left| \mathcal{F}\{x(t)\} \right|^2 = \left| \sum_{t=0}^{N-1} x_t e^{-2\pi i f t / N} \right|^2''',
        'description': 'Decomposes signal into frequency components via FFT.',
        'full_derivation': r'''
\textbf{Spectral Analysis (Power Spectral Density)}

1. \textbf{Discrete Fourier Transform:}
   $$X_k = \sum_{t=0}^{N-1} x_t e^{-2\pi i k t / N}$$

2. \textbf{Power Spectral Density:}
   $$S_k = \frac{1}{N} |X_k|^2$$

3. \textbf{Frequency bins:}
   $$f_k = \frac{k}{N \Delta t}$$
   
   where $\Delta t$ is the sampling interval.

\textbf{Derived features:}

\textbf{Dominant frequency:}
$$f_{dom} = \arg\max_k S_k$$

\textbf{Spectral centroid:}
$$f_c = \frac{\sum_k f_k S_k}{\sum_k S_k}$$

\textbf{Spectral entropy:}
$$H_s = -\sum_k p_k \log(p_k)$$
where $p_k = S_k / \sum_j S_j$

\textbf{Band power ratios:}
$$R = \frac{\sum_{k \in \text{band}_1} S_k}{\sum_{k \in \text{band}_2} S_k}$$
'''
    },
    
    'statistical': {
        'name': 'Statistical Moments',
        'formula': r'''\mu_n = \mathbb{E}[(X - \mu)^n]''',
        'description': 'Basic distributional properties: mean, variance, skewness, kurtosis.',
        'full_derivation': r'''
\textbf{Statistical Moments}

Given $X = \{x_1, \ldots, x_n\}$:

\textbf{Mean (1st moment):}
$$\mu = \bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

\textbf{Variance (2nd central moment):}
$$\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2$$

\textbf{Skewness (3rd standardized moment):}
$$\gamma_1 = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{x_i - \bar{x}}{\sigma} \right)^3$$

Interpretation: $\gamma_1 > 0$ right-skewed, $\gamma_1 < 0$ left-skewed

\textbf{Kurtosis (4th standardized moment):}
$$\gamma_2 = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{x_i - \bar{x}}{\sigma} \right)^4 - 3$$

Interpretation: $\gamma_2 > 0$ heavy tails, $\gamma_2 < 0$ light tails

\textbf{Additional statistics:}
\begin{itemize}
    \item Range: $\max(X) - \min(X)$
    \item IQR: $Q_3 - Q_1$
    \item Coefficient of variation: $CV = \sigma / \mu$
\end{itemize}
'''
    },
}


class Proof:
    """
    Generate mathematical proofs of PRISM calculations.
    
    "Show your work" - for any (entity, timestamp, signal, engine),
    displays the raw data, formula, intermediate steps, and result.
    """
    
    def __init__(
        self,
        observations: pl.DataFrame,
        vector_df: Optional[pl.DataFrame] = None,
        geometry_df: Optional[pl.DataFrame] = None,
        state_df: Optional[pl.DataFrame] = None,
        window_size: int = 50,
    ):
        """
        Initialize proof tool with PRISM data.
        
        Args:
            observations: Standardized observations (entity_id, timestamp, signal, value)
            vector_df: Vector layer output (optional, for verification)
            geometry_df: Geometry layer output (optional)
            state_df: State layer output (optional)
            window_size: Default window size for calculations
        """
        self.observations = observations
        self.vector_df = vector_df
        self.geometry_df = geometry_df
        self.state_df = state_df
        self.window_size = window_size
    
    def snapshot(
        self, 
        entity_id: str, 
        timestamp: float,
    ) -> pl.DataFrame:
        """Get all signal values at a specific timestamp."""
        return self.observations.filter(
            (pl.col('entity_id') == str(entity_id)) & 
            (pl.col('timestamp') == timestamp)
        ).pivot(
            index=['entity_id', 'timestamp'],
            on='signal',
            values='value'
        )
    
    def window(
        self,
        entity_id: str,
        timestamp: float,
        signal: str,
        size: Optional[int] = None,
    ) -> pl.DataFrame:
        """Get window of values leading up to timestamp for one signal."""
        size = size or self.window_size
        start_t = max(1, timestamp - size + 1)
        
        return self.observations.filter(
            (pl.col('entity_id') == str(entity_id)) &
            (pl.col('signal') == signal) &
            (pl.col('timestamp') >= start_t) &
            (pl.col('timestamp') <= timestamp)
        ).sort('timestamp')
    
    def explain(
        self,
        entity_id: str,
        timestamp: float,
        signal: str,
        engine: str,
        window_size: Optional[int] = None,
    ) -> EngineProof:
        """
        Generate full proof for one engine calculation.
        
        Args:
            entity_id: Entity to analyze
            timestamp: Time point
            signal: Signal name (e.g., 's_11')
            engine: Engine name (e.g., 'hurst', 'entropy', 'garch')
            window_size: Override default window size
            
        Returns:
            EngineProof with all calculation details
        """
        window_size = window_size or self.window_size
        
        # Get window of raw data
        window_df = self.window(entity_id, timestamp, signal, window_size)
        raw_values = window_df['value'].to_numpy()
        
        if len(raw_values) < 10:
            raise ValueError(f"Insufficient data: only {len(raw_values)} points in window")
        
        # Get formula info
        if engine not in FORMULAS:
            raise ValueError(f"Unknown engine: {engine}. Available: {list(FORMULAS.keys())}")
        
        formula_info = FORMULAS[engine]
        
        # Compute with step-by-step tracking
        steps, result = self._compute_with_steps(engine, raw_values)
        
        return EngineProof(
            engine=engine,
            signal=signal,
            entity_id=str(entity_id),
            timestamp=timestamp,
            window_start=window_df['timestamp'].min(),
            window_end=window_df['timestamp'].max(),
            raw_values=raw_values,
            formula_latex=formula_info['full_derivation'],
            steps=steps,
            result=result,
        )
    
    def _compute_with_steps(
        self, 
        engine: str, 
        values: np.ndarray,
    ) -> tuple:
        """Compute engine output with intermediate steps."""
        
        if engine == 'hurst':
            return self._hurst_steps(values)
        elif engine == 'entropy':
            return self._entropy_steps(values)
        elif engine == 'garch':
            return self._garch_steps(values)
        elif engine == 'lyapunov':
            return self._lyapunov_steps(values)
        elif engine == 'statistical':
            return self._statistical_steps(values)
        elif engine == 'spectral':
            return self._spectral_steps(values)
        else:
            return [], np.nan
    
    def _hurst_steps(self, values: np.ndarray) -> tuple:
        """Hurst exponent with steps."""
        steps = []
        n = len(values)
        
        # Step 1: Mean
        mean = np.mean(values)
        steps.append({
            'step': 1,
            'name': 'Compute mean',
            'formula': r'\bar{x} = \frac{1}{n} \sum x_i',
            'result': mean,
        })
        
        # Step 2: Mean-adjusted series
        Y = values - mean
        steps.append({
            'step': 2,
            'name': 'Mean-adjusted series',
            'formula': r'Y_t = x_t - \bar{x}',
            'result': f'Array of {len(Y)} values, first 5: {Y[:5].round(4).tolist()}',
        })
        
        # Step 3: Cumulative deviation
        Z = np.cumsum(Y)
        steps.append({
            'step': 3,
            'name': 'Cumulative deviation',
            'formula': r'Z_t = \sum_{i=1}^{t} Y_i',
            'result': f'Array of {len(Z)} values, range: [{Z.min():.4f}, {Z.max():.4f}]',
        })
        
        # Step 4: Range
        R = np.max(Z) - np.min(Z)
        steps.append({
            'step': 4,
            'name': 'Range',
            'formula': r'R = \max(Z_t) - \min(Z_t)',
            'result': R,
        })
        
        # Step 5: Standard deviation
        S = np.std(values, ddof=0)
        steps.append({
            'step': 5,
            'name': 'Standard deviation',
            'formula': r'S = \sqrt{\frac{1}{n} \sum (x_i - \bar{x})^2}',
            'result': S,
        })
        
        # Step 6: R/S
        RS = R / S if S > 0 else np.nan
        steps.append({
            'step': 6,
            'name': 'Rescaled range',
            'formula': r'R/S',
            'result': RS,
        })
        
        # Step 7: Hurst
        H = np.log(RS) / np.log(n) if RS > 0 else np.nan
        steps.append({
            'step': 7,
            'name': 'Hurst exponent',
            'formula': r'H = \frac{\log(R/S)}{\log(n)}',
            'result': H,
        })
        
        return steps, H
    
    def _entropy_steps(self, values: np.ndarray, bins: int = 20) -> tuple:
        """Shannon entropy with steps."""
        steps = []
        
        # Step 1: Define bins
        min_val, max_val = np.min(values), np.max(values)
        steps.append({
            'step': 1,
            'name': 'Define bin range',
            'formula': r'[\min(X), \max(X)]',
            'result': f'[{min_val:.4f}, {max_val:.4f}], {bins} bins',
        })
        
        # Step 2: Histogram
        counts, bin_edges = np.histogram(values, bins=bins)
        steps.append({
            'step': 2,
            'name': 'Count per bin',
            'formula': r'n_i = \text{count in bin } i',
            'result': f'Counts: {counts.tolist()}',
        })
        
        # Step 3: Probabilities
        probs = counts / len(values)
        probs = probs[probs > 0]  # Remove zeros for log
        steps.append({
            'step': 3,
            'name': 'Probabilities',
            'formula': r'p_i = n_i / n',
            'result': f'{len(probs)} non-zero bins',
        })
        
        # Step 4: Entropy
        H = -np.sum(probs * np.log2(probs))
        steps.append({
            'step': 4,
            'name': 'Shannon entropy',
            'formula': r'H = -\sum p_i \log_2(p_i)',
            'result': H,
        })
        
        # Step 5: Normalized
        H_max = np.log2(bins)
        H_norm = H / H_max
        steps.append({
            'step': 5,
            'name': 'Normalized entropy',
            'formula': r'H_{norm} = H / \log_2(k)',
            'result': H_norm,
        })
        
        return steps, H
    
    def _garch_steps(self, values: np.ndarray) -> tuple:
        """GARCH volatility with steps (simplified)."""
        steps = []
        
        # Step 1: Returns
        returns = np.diff(values)
        steps.append({
            'step': 1,
            'name': 'Compute returns',
            'formula': r'r_t = x_t - x_{t-1}',
            'result': f'{len(returns)} returns, std={np.std(returns):.4f}',
        })
        
        # Step 2: Squared returns (proxy for variance)
        sq_returns = returns ** 2
        steps.append({
            'step': 2,
            'name': 'Squared returns',
            'formula': r'\epsilon_t^2 = r_t^2',
            'result': f'Mean sq return: {np.mean(sq_returns):.6f}',
        })
        
        # Step 3: EWMA variance (simplified GARCH proxy)
        alpha = 0.1
        variance = np.zeros(len(sq_returns))
        variance[0] = sq_returns[0]
        for t in range(1, len(sq_returns)):
            variance[t] = alpha * sq_returns[t-1] + (1-alpha) * variance[t-1]
        
        steps.append({
            'step': 3,
            'name': 'EWMA variance (GARCH proxy)',
            'formula': r'\sigma_t^2 = \alpha \epsilon_{t-1}^2 + (1-\alpha) \sigma_{t-1}^2',
            'result': f'Final variance: {variance[-1]:.6f}',
        })
        
        # Step 4: Volatility
        volatility = np.sqrt(variance[-1])
        steps.append({
            'step': 4,
            'name': 'Conditional volatility',
            'formula': r'\sigma_t = \sqrt{\sigma_t^2}',
            'result': volatility,
        })
        
        return steps, volatility
    
    def _lyapunov_steps(self, values: np.ndarray) -> tuple:
        """Lyapunov exponent with steps (simplified)."""
        steps = []
        
        # Step 1: Embedding params
        tau = 1  # delay
        m = 3    # dimension
        steps.append({
            'step': 1,
            'name': 'Embedding parameters',
            'formula': r'\tau = 1, m = 3',
            'result': f'Delay={tau}, Dimension={m}',
        })
        
        # Step 2: Create embedded vectors
        N = len(values) - (m - 1) * tau
        if N < 10:
            return steps, np.nan
            
        embedded = np.array([values[i:i + m * tau:tau] for i in range(N)])
        steps.append({
            'step': 2,
            'name': 'Delay embedding',
            'formula': r'\mathbf{x}_i = (x_i, x_{i+\tau}, \ldots)',
            'result': f'{N} embedded vectors of dim {m}',
        })
        
        # Step 3: Find nearest neighbors and track divergence
        # (Simplified: use adjacent points)
        divergences = []
        for i in range(N - 10):
            d0 = np.linalg.norm(embedded[i] - embedded[i+1])
            d1 = np.linalg.norm(embedded[i+5] - embedded[i+6])
            if d0 > 1e-10:
                divergences.append(np.log(d1 / d0) / 5)
        
        steps.append({
            'step': 3,
            'name': 'Track divergence',
            'formula': r'\lambda \approx \frac{1}{\Delta t} \ln \frac{d(t+\Delta t)}{d(t)}',
            'result': f'{len(divergences)} divergence estimates',
        })
        
        # Step 4: Average
        lyap = np.mean(divergences) if divergences else np.nan
        steps.append({
            'step': 4,
            'name': 'Lyapunov exponent',
            'formula': r'\lambda = \langle \text{divergence rate} \rangle',
            'result': lyap,
        })
        
        return steps, lyap
    
    def _statistical_steps(self, values: np.ndarray) -> tuple:
        """Statistical moments with steps."""
        steps = []
        n = len(values)
        
        mean = np.mean(values)
        steps.append({'step': 1, 'name': 'Mean', 'formula': r'\mu', 'result': mean})
        
        var = np.var(values)
        steps.append({'step': 2, 'name': 'Variance', 'formula': r'\sigma^2', 'result': var})
        
        std = np.std(values)
        steps.append({'step': 3, 'name': 'Std Dev', 'formula': r'\sigma', 'result': std})
        
        if std > 0:
            skew = np.mean(((values - mean) / std) ** 3)
            kurt = np.mean(((values - mean) / std) ** 4) - 3
        else:
            skew, kurt = 0, 0
        
        steps.append({'step': 4, 'name': 'Skewness', 'formula': r'\gamma_1', 'result': skew})
        steps.append({'step': 5, 'name': 'Kurtosis', 'formula': r'\gamma_2', 'result': kurt})
        
        return steps, {'mean': mean, 'var': var, 'std': std, 'skew': skew, 'kurt': kurt}
    
    def _spectral_steps(self, values: np.ndarray) -> tuple:
        """Spectral analysis with steps."""
        steps = []
        n = len(values)
        
        # FFT
        fft = np.fft.fft(values)
        psd = np.abs(fft[:n//2]) ** 2 / n
        freqs = np.fft.fftfreq(n)[:n//2]
        
        steps.append({
            'step': 1, 
            'name': 'FFT', 
            'formula': r'X_k = \mathcal{F}\{x_t\}',
            'result': f'{len(psd)} frequency bins',
        })
        
        steps.append({
            'step': 2,
            'name': 'Power spectral density',
            'formula': r'S_k = |X_k|^2 / N',
            'result': f'Total power: {np.sum(psd):.4f}',
        })
        
        # Dominant frequency
        dom_idx = np.argmax(psd[1:]) + 1  # Skip DC
        dom_freq = freqs[dom_idx]
        steps.append({
            'step': 3,
            'name': 'Dominant frequency',
            'formula': r'f_{dom} = \arg\max S_k',
            'result': dom_freq,
        })
        
        # Spectral centroid
        centroid = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
        steps.append({
            'step': 4,
            'name': 'Spectral centroid',
            'formula': r'f_c = \sum f_k S_k / \sum S_k',
            'result': centroid,
        })
        
        return steps, {'dom_freq': dom_freq, 'centroid': centroid, 'total_power': np.sum(psd)}
    
    def to_markdown(self, proof: EngineProof) -> str:
        """Export proof as Markdown with LaTeX."""
        formula_info = FORMULAS[proof.engine]
        
        md = f"""# PRISM Proof: {formula_info['name']}

## Context
- **Entity:** {proof.entity_id}
- **Signal:** {proof.signal}
- **Timestamp:** {proof.timestamp}
- **Window:** [{proof.window_start}, {proof.window_end}] ({len(proof.raw_values)} points)

## Formula

{formula_info['description']}

$$
{formula_info['formula']}
$$

## Raw Data (first 10 values)

```
{proof.raw_values[:10].round(4).tolist()}
```

## Step-by-Step Calculation

"""
        for step in proof.steps:
            md += f"""### Step {step['step']}: {step['name']}

$$
{step['formula']}
$$

**Result:** `{step['result']}`

"""
        
        md += f"""## Final Result

**{formula_info['name']} = {proof.result if isinstance(proof.result, (int, float)) else proof.result}**

---

## Full Derivation

{proof.formula_latex}
"""
        return md
    
    def to_latex(self, proof: EngineProof) -> str:
        """Export proof as standalone LaTeX document."""
        formula_info = FORMULAS[proof.engine]
        
        latex = r"""\documentclass{article}
\usepackage{amsmath, amssymb}
\usepackage{booktabs}
\usepackage{geometry}
\geometry{margin=1in}

\title{PRISM Proof: """ + formula_info['name'] + r"""}
\author{Generated by PRISM}
\date{}

\begin{document}
\maketitle

\section{Context}
\begin{itemize}
    \item \textbf{Entity:} """ + str(proof.entity_id) + r"""
    \item \textbf{Signal:} """ + proof.signal + r"""
    \item \textbf{Timestamp:} """ + str(proof.timestamp) + r"""
    \item \textbf{Window:} [""" + str(proof.window_start) + r""", """ + str(proof.window_end) + r"""] (""" + str(len(proof.raw_values)) + r""" points)
\end{itemize}

\section{Formula}

""" + formula_info['description'] + r"""

$$""" + formula_info['formula'] + r"""$$

\section{Step-by-Step Calculation}

"""
        for step in proof.steps:
            latex += r"""
\subsection{Step """ + str(step['step']) + r""": """ + step['name'] + r"""}
$$""" + step['formula'] + r"""$$
\textbf{Result:} """ + str(step['result']) + r"""

"""
        
        latex += r"""
\section{Final Result}

\textbf{""" + formula_info['name'] + r""" = """ + str(proof.result) + r"""}

\section{Full Derivation}

""" + proof.formula_latex + r"""

\end{document}
"""
        return latex
    
    def print_proof(self, proof: EngineProof) -> None:
        """Pretty print proof to console."""
        formula_info = FORMULAS[proof.engine]
        
        print("=" * 70)
        print(f"PRISM PROOF: {formula_info['name'].upper()}")
        print("=" * 70)
        print(f"\nEntity: {proof.entity_id}")
        print(f"Signal: {proof.signal}")
        print(f"Timestamp: {proof.timestamp}")
        print(f"Window: [{proof.window_start}, {proof.window_end}] ({len(proof.raw_values)} points)")
        
        print(f"\n{formula_info['description']}")
        
        print("\n" + "-" * 70)
        print("RAW DATA (first 10 values)")
        print("-" * 70)
        print(proof.raw_values[:10].round(4))
        
        print("\n" + "-" * 70)
        print("STEP-BY-STEP CALCULATION")
        print("-" * 70)
        
        for step in proof.steps:
            print(f"\nStep {step['step']}: {step['name']}")
            print(f"  Formula: {step['formula']}")
            print(f"  Result:  {step['result']}")
        
        print("\n" + "=" * 70)
        print(f"FINAL RESULT: {proof.result}")
        print("=" * 70)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("PRISM Proof Tool")
    print("Usage: from proof import Proof")
    print("       proof = Proof(observations)")
    print("       p = proof.explain(entity_id='1', timestamp=50, signal='s_11', engine='hurst')")
    print("       proof.print_proof(p)")
    print("       proof.to_latex(p)")
