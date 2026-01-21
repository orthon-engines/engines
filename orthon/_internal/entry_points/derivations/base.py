"""
PRISM Derivation Base Classes
=============================

Core classes for capturing and rendering mathematical derivations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Any, Optional, Union
from datetime import datetime
import numpy as np


@dataclass
class DerivationStep:
    """Single step in a mathematical derivation."""
    step_number: int
    title: str
    equation: str           # The formula in text/unicode
    calculation: str        # Showing actual values substituted
    result: Union[float, List[float], np.ndarray]
    result_name: str        # Variable name for result
    notes: str = ""         # Optional explanation

    def format_result(self, precision: int = 6) -> str:
        """Format result for display."""
        if isinstance(self.result, (list, np.ndarray)):
            arr = np.asarray(self.result)
            if len(arr) <= 5:
                return "[" + ", ".join(f"{v:.{precision}f}" for v in arr) + "]"
            else:
                return f"[{arr[0]:.{precision}f}, {arr[1]:.{precision}f}, ..., {arr[-1]:.{precision}f}] (n={len(arr)})"
        elif isinstance(self.result, float):
            return f"{self.result:.{precision}f}"
        else:
            return str(self.result)


@dataclass
class Derivation:
    """Complete derivation for one engine run."""
    engine_name: str
    method_name: str
    signal_id: str
    window_id: str
    window_start: Optional[str] = None
    window_end: Optional[str] = None
    sample_size: int = 0
    raw_data_sample: List[float] = field(default_factory=list)
    steps: List[DerivationStep] = field(default_factory=list)
    final_result: Optional[float] = None
    prism_output: Optional[float] = None
    interpretation: str = ""
    data_path: str = ""
    generated_at: datetime = field(default_factory=datetime.now)

    # Engine-specific metadata
    parameters: dict = field(default_factory=dict)

    def add_step(self, title: str, equation: str, calculation: str,
                 result: Union[float, List[float], np.ndarray],
                 result_name: str, notes: str = "") -> Union[float, List[float], np.ndarray]:
        """Add a derivation step and return result for chaining."""
        step = DerivationStep(
            step_number=len(self.steps) + 1,
            title=title,
            equation=equation,
            calculation=calculation,
            result=result,
            result_name=result_name,
            notes=notes
        )
        self.steps.append(step)
        return result

    def validate(self, rtol: float = 1e-5) -> bool:
        """Check if derived result matches PRISM output."""
        if self.final_result is None or self.prism_output is None:
            return False
        return np.isclose(self.final_result, self.prism_output, rtol=rtol)

    def to_markdown(self) -> str:
        """Render complete derivation as markdown."""
        lines = []

        # Header
        lines.append(f"# PRISM Mathematical Derivation: {self.method_name}")
        lines.append("")
        lines.append(f"**Engine:** `{self.engine_name}`  ")
        lines.append(f"**Method:** {self.method_name}  ")
        lines.append(f"**Generated:** {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}  ")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Purpose section (engine-specific, can be overridden)
        lines.append("## 1. Purpose")
        lines.append("")
        lines.append(self._get_purpose())
        lines.append("")
        lines.append("---")
        lines.append("")

        # Mathematical Definition
        lines.append("## 2. Mathematical Definition")
        lines.append("")
        lines.append(self._get_definition())
        lines.append("")

        # Interpretation table
        lines.append("### Interpretation")
        lines.append("")
        lines.append(self._get_interpretation_table())
        lines.append("")
        lines.append("---")
        lines.append("")

        # Input Data
        lines.append("## 3. Input Data")
        lines.append("")
        lines.append(f"**Signal:** `{self.signal_id}`  ")
        lines.append(f"**Window:** {self.window_id}  ")
        if self.window_start and self.window_end:
            lines.append(f"**Date Range:** {self.window_start} to {self.window_end}  ")
        lines.append(f"**Sample Size:** n = {self.sample_size}  ")
        lines.append("")

        # Parameters if any
        if self.parameters:
            lines.append("### Parameters")
            lines.append("")
            lines.append("| Parameter | Value |")
            lines.append("|-----------|-------|")
            for k, v in self.parameters.items():
                lines.append(f"| {k} | {v} |")
            lines.append("")

        # Raw data sample
        if self.raw_data_sample:
            lines.append("### Raw Data Sample (first 10 values)")
            lines.append("")
            lines.append("```")
            for i, v in enumerate(self.raw_data_sample[:10]):
                lines.append(f"x[{i}] = {v:.6f}")
            if len(self.raw_data_sample) > 10:
                lines.append("...")
            lines.append("```")
            lines.append("")

        lines.append("---")
        lines.append("")

        # Step-by-step calculation
        lines.append("## 4. Step-by-Step Calculation")
        lines.append("")

        for step in self.steps:
            lines.append(f"### Step {step.step_number}: {step.title}")
            lines.append("")
            lines.append("**Equation:**")
            lines.append(f"```")
            lines.append(step.equation)
            lines.append("```")
            lines.append("")
            lines.append("**Calculation:**")
            lines.append("```")
            lines.append(step.calculation)
            lines.append("```")
            lines.append("")
            lines.append(f"**Result:** {step.result_name} = {step.format_result()}")
            lines.append("")
            if step.notes:
                lines.append(f"*Note: {step.notes}*")
                lines.append("")

        lines.append("---")
        lines.append("")

        # Result Validation
        lines.append("## 5. Result Validation")
        lines.append("")
        lines.append("| Metric | Derived Value | PRISM Output | Match |")
        lines.append("|--------|---------------|--------------|-------|")

        derived_str = f"{self.final_result:.6f}" if self.final_result is not None else "N/A"
        prism_str = f"{self.prism_output:.6f}" if self.prism_output is not None else "N/A"
        match_str = "Yes" if self.validate() else "No"

        lines.append(f"| {self.engine_name} | {derived_str} | {prism_str} | {match_str} |")
        lines.append("")

        if self.validate():
            lines.append("**Validation: PASSED** - Derived value matches PRISM output.")
        else:
            lines.append("**Validation: CHECK** - Values may differ due to numerical precision or algorithm variants.")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Physical Interpretation
        lines.append("## 6. Physical Interpretation")
        lines.append("")
        if self.interpretation:
            lines.append(self.interpretation)
        else:
            lines.append("*No interpretation provided.*")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Reproducibility
        lines.append("## 7. Reproducibility")
        lines.append("")
        if self.data_path:
            lines.append(f"**Data Source:** `{self.data_path}`  ")
        lines.append("")
        lines.append("**To reproduce:**")
        lines.append("```bash")
        lines.append(f"python -m prism.derivations.generate --engine {self.engine_name} --signal {self.signal_id} --window {self.window_id}")
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("*PRISM Framework - Compute once, query forever*")

        return "\n".join(lines)

    def _get_purpose(self) -> str:
        """Get purpose description for this engine. Override in subclasses."""
        purposes = {
            'hurst_exponent': "The Hurst exponent measures long-range dependence (memory) in a signal topology. It distinguishes between persistent processes (trending), random walks, and anti-persistent (mean-reverting) processes.",
            'lyapunov_exponent': "The Lyapunov exponent quantifies the rate of separation of infinitesimally close trajectories. A positive value confirms deterministic chaos.",
            'sample_entropy': "Sample entropy measures the complexity/regularity of a signal topology. Lower values indicate more regular, predictable patterns.",
            'permutation_entropy': "Permutation entropy measures complexity based on the distribution of ordinal patterns in the data.",
            'spectral_entropy': "Spectral entropy measures the flatness of the power spectrum. High values indicate broadband noise; low values indicate concentrated periodic components.",
            'dfa': "Detrended Fluctuation Analysis (DFA) measures self-similarity and long-range correlations, robust to non-stationarity.",
            'garch': "GARCH models conditional heteroskedasticity (volatility clustering) - periods of high volatility tend to cluster together.",
            'cohesion': "Cohesion measures the average pairwise similarity of signal vectors within a cohort.",
            'effective_dimension': "Effective dimension (participation ratio) measures the number of independent behavioral modes in the cohort.",
            'divergence': "Divergence (Laplacian) measures the net flow at each point in behavioral space - sources (expanding) vs sinks (contracting).",
        }
        return purposes.get(self.engine_name, f"Computes the {self.engine_name} metric.")

    def _get_definition(self) -> str:
        """Get mathematical definition. Override in subclasses."""
        definitions = {
            'hurst_exponent': """
The Hurst exponent H is estimated from the scaling of the rescaled range:

```
E[R(n)/S(n)] ~ C · n^H
```

Where:
- R(n) = range of cumulative deviations from mean
- S(n) = standard deviation
- n = sample size
- H = Hurst exponent (0 < H < 1)
""",
            'lyapunov_exponent': """
The largest Lyapunov exponent λ measures exponential divergence:

```
|δZ(t)| ~ |δZ(0)| · e^(λt)
```

Where:
- δZ(t) = separation between nearby trajectories at time t
- λ > 0 indicates chaos (exponential divergence)
- λ < 0 indicates stability (convergence to attractor)
""",
            'sample_entropy': """
Sample entropy is defined as:

```
SampEn(m, r, N) = -ln[A/B]
```

Where:
- A = number of template matches of length m+1
- B = number of template matches of length m
- m = embedding dimension
- r = tolerance threshold
""",
            'dfa': """
DFA scaling exponent α is estimated from:

```
F(s) ~ s^α
```

Where:
- F(s) = RMS fluctuation at scale s
- α = DFA exponent
- α > 0.5: persistent (trending)
- α = 0.5: random walk
- α < 0.5: anti-persistent (mean-reverting)
""",
        }
        return definitions.get(self.engine_name, f"```\n{self.engine_name}\n```")

    def _get_interpretation_table(self) -> str:
        """Get interpretation table. Override in subclasses."""
        tables = {
            'hurst_exponent': """
| Value Range | Interpretation |
|-------------|----------------|
| H < 0.5 | Anti-persistent (mean-reverting) |
| H = 0.5 | Random walk (no memory) |
| H > 0.5 | Persistent (trending) |
| H > 1.0 | Indicates strong determinism (use DFA) |
""",
            'lyapunov_exponent': """
| Value | Interpretation |
|-------|----------------|
| λ > 0 | Chaos (exponential divergence) |
| λ = 0 | Edge of chaos / periodic |
| λ < 0 | Stable fixed point / limit cycle |
""",
            'sample_entropy': """
| Value Range | Interpretation |
|-------------|----------------|
| Low (< 0.5) | Highly regular/deterministic |
| Medium (0.5-1.5) | Moderate complexity |
| High (> 1.5) | High complexity/randomness |
""",
            'dfa': """
| Value | Interpretation |
|-------|----------------|
| α < 0.5 | Anti-persistent |
| α = 0.5 | White noise |
| 0.5 < α < 1.0 | Persistent long-range correlations |
| α = 1.0 | 1/f noise (pink noise) |
| α > 1.0 | Non-stationary, unbounded |
""",
        }
        return tables.get(self.engine_name, "| Value | Interpretation |\n|-------|----------------|\n| - | See documentation |")


class DerivableEngine(ABC):
    """Mixin for engines that support derivation output."""

    @abstractmethod
    def compute_with_derivation(self, data: np.ndarray,
                                signal_id: str,
                                window_id: str,
                                window_start: str = None,
                                window_end: str = None) -> tuple[Any, Derivation]:
        """
        Compute the metric AND capture all intermediate steps.

        Returns:
            tuple: (result_dict, Derivation object)
        """
        pass
