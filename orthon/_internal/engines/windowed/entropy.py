"""
PRISM Entropy Engine

Measures complexity and predictability of signal topology.

Measures:
- Permutation entropy (complexity)
- Sample entropy (regularity)

Phase: Unbound
Normalization: Varies by method
"""

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to use antropy (fast), fall back to manual (slow)
try:
    import antropy as ant
    HAS_ANTROPY = True
except ImportError:
    HAS_ANTROPY = False
    logger.warning("antropy not installed. Using slow entropy calculation.")


def compute_sample_entropy_with_derivation(
    values: np.ndarray,
    signal_id: str = "unknown",
    window_id: str = "0",
    window_start: str = None,
    window_end: str = None,
    m: int = 2,
    r: float = None,
) -> tuple:
    """
    Compute Sample Entropy with full mathematical derivation.

    Returns:
        tuple: (result_dict, Derivation object)
    """
    from orthon._internal.entry_points.derivations.base import Derivation

    deriv = Derivation(
        engine_name="sample_entropy",
        method_name="Sample Entropy (SampEn)",
        signal_id=signal_id,
        window_id=window_id,
        window_start=window_start,
        window_end=window_end,
        sample_size=len(values),
        raw_data_sample=values[:10].tolist() if len(values) >= 10 else values.tolist(),
    )

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)

    if n < 20:
        deriv.final_result = None
        deriv.interpretation = "Insufficient data (n < 20)"
        return {"sample_entropy": None}, deriv

    # Set tolerance
    if r is None:
        r = 0.2 * np.std(values)

    deriv.parameters = {'m': m, 'r': r}

    # Step 1: Data summary
    deriv.add_step(
        title="Input Data Summary",
        equation="X = {x₁, x₂, ..., xₙ}",
        calculation=f"n = {n}\nMean: {np.mean(values):.4f}\nStd: {np.std(values):.4f}",
        result=n,
        result_name="n"
    )

    # Step 2: Set parameters
    deriv.add_step(
        title="Set Parameters",
        equation="m = embedding dimension, r = tolerance threshold",
        calculation=f"m = {m} (pattern length)\nr = 0.2 × σ = 0.2 × {np.std(values):.4f} = {r:.4f}",
        result=r,
        result_name="r",
        notes="r = 0.2σ is a common choice (Richman & Moorman, 2000)"
    )

    # Step 3: Build templates of length m
    templates_m = np.array([values[i:i + m] for i in range(n - m)])
    n_templates_m = len(templates_m)

    deriv.add_step(
        title=f"Build Templates of Length m={m}",
        equation=f"Template tᵢ = [xᵢ, xᵢ₊₁, ..., xᵢ₊{m-1}]",
        calculation=f"Number of templates: {n_templates_m}\n\nExample templates:\nt₀ = [{values[0]:.4f}, {values[1]:.4f}]\nt₁ = [{values[1]:.4f}, {values[2]:.4f}]\nt₂ = [{values[2]:.4f}, {values[3]:.4f}]\n⋮",
        result=n_templates_m,
        result_name="N_m"
    )

    # Step 4: Count matches for m
    from scipy.spatial.distance import cdist
    dist_m = cdist(templates_m, templates_m, metric='chebyshev')
    np.fill_diagonal(dist_m, np.inf)
    B = np.sum(dist_m < r) // 2

    # Show example distance calculation
    d01 = np.max(np.abs(templates_m[0] - templates_m[1]))

    deriv.add_step(
        title=f"Count Template Matches (length m={m})",
        equation="Match if d(tᵢ, tⱼ) = max|tᵢₖ - tⱼₖ| < r",
        calculation=f"Distance metric: Chebyshev (max absolute difference)\n\nExample: d(t₀, t₁) = max(|{templates_m[0][0]:.4f} - {templates_m[1][0]:.4f}|, |{templates_m[0][1]:.4f} - {templates_m[1][1]:.4f}|)\n       = max({abs(templates_m[0][0] - templates_m[1][0]):.4f}, {abs(templates_m[0][1] - templates_m[1][1]):.4f})\n       = {d01:.4f}\n       {'< ' + f'{r:.4f} = MATCH' if d01 < r else '>= ' + f'{r:.4f} = NO MATCH'}\n\nTotal matches B = {B}",
        result=B,
        result_name="B",
        notes="B = number of template pairs matching within tolerance r"
    )

    # Step 5: Build templates of length m+1
    templates_m1 = np.array([values[i:i + m + 1] for i in range(n - m - 1)])
    n_templates_m1 = len(templates_m1)

    deriv.add_step(
        title=f"Build Templates of Length m+1={m+1}",
        equation=f"Template tᵢ = [xᵢ, xᵢ₊₁, ..., xᵢ₊{m}]",
        calculation=f"Number of templates: {n_templates_m1}\n\nExample templates:\nt₀ = [{values[0]:.4f}, {values[1]:.4f}, {values[2]:.4f}]\nt₁ = [{values[1]:.4f}, {values[2]:.4f}, {values[3]:.4f}]\n⋮",
        result=n_templates_m1,
        result_name="N_m+1"
    )

    # Step 6: Count matches for m+1
    dist_m1 = cdist(templates_m1, templates_m1, metric='chebyshev')
    np.fill_diagonal(dist_m1, np.inf)
    A = np.sum(dist_m1 < r) // 2

    deriv.add_step(
        title=f"Count Template Matches (length m+1={m+1})",
        equation="Match if d(tᵢ, tⱼ) = max|tᵢₖ - tⱼₖ| < r",
        calculation=f"Total matches A = {A}",
        result=A,
        result_name="A",
        notes="A = number of (m+1)-length template pairs matching"
    )

    # Step 7: Compute sample entropy
    if B == 0:
        samp_ent = np.nan
        deriv.add_step(
            title="Compute Sample Entropy",
            equation="SampEn = -ln(A/B)",
            calculation="B = 0, Sample Entropy undefined",
            result=np.nan,
            result_name="SampEn"
        )
    else:
        ratio = A / B
        samp_ent = -np.log(ratio) if A > 0 else np.nan

        deriv.add_step(
            title="Compute Sample Entropy",
            equation="SampEn = -ln(A/B)",
            calculation=f"A/B = {A}/{B} = {ratio:.6f}\nSampEn = -ln({ratio:.6f}) = {samp_ent:.6f}" if not np.isnan(samp_ent) else f"A = 0, SampEn = undefined",
            result=samp_ent if not np.isnan(samp_ent) else None,
            result_name="SampEn",
            notes="Lower entropy = more regular/predictable"
        )

    deriv.final_result = samp_ent if not np.isnan(samp_ent) else None
    deriv.prism_output = samp_ent if not np.isnan(samp_ent) else None

    # Interpretation
    if np.isnan(samp_ent):
        interp = "Sample entropy undefined (no matches found)."
    elif samp_ent < 0.5:
        interp = f"SampEn = {samp_ent:.4f} < 0.5 indicates **highly regular** dynamics. The signal topology is highly predictable."
    elif samp_ent < 1.5:
        interp = f"SampEn = {samp_ent:.4f} indicates **moderate complexity**. Some predictability exists."
    else:
        interp = f"SampEn = {samp_ent:.4f} > 1.5 indicates **high complexity/randomness**. The signal topology is unpredictable."

    deriv.interpretation = interp

    return {"sample_entropy": samp_ent if not np.isnan(samp_ent) else None}, deriv


def compute_permutation_entropy_with_derivation(
    values: np.ndarray,
    signal_id: str = "unknown",
    window_id: str = "0",
    window_start: str = None,
    window_end: str = None,
    order: int = 3,
    tau: int = 1,
) -> tuple:
    """
    Compute Permutation Entropy with full mathematical derivation.

    Returns:
        tuple: (result_dict, Derivation object)
    """
    from orthon._internal.entry_points.derivations.base import Derivation
    from collections import Counter
    from math import factorial

    deriv = Derivation(
        engine_name="permutation_entropy",
        method_name="Permutation Entropy (PE)",
        signal_id=signal_id,
        window_id=window_id,
        window_start=window_start,
        window_end=window_end,
        sample_size=len(values),
        raw_data_sample=values[:10].tolist() if len(values) >= 10 else values.tolist(),
        parameters={'order': order, 'tau': tau}
    )

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)

    if n < order * tau:
        deriv.final_result = None
        deriv.interpretation = f"Insufficient data (n < order × tau = {order * tau})"
        return {"permutation_entropy": None}, deriv

    # Step 1: Data summary
    deriv.add_step(
        title="Input Data Summary",
        equation="X = {x₁, x₂, ..., xₙ}",
        calculation=f"n = {n}\nOrder m = {order}\nTime delay τ = {tau}",
        result=n,
        result_name="n"
    )

    # Step 2: Extract ordinal patterns
    patterns = []
    for i in range(n - (order - 1) * tau):
        window = values[i:i + order * tau:tau]
        pattern = tuple(np.argsort(window))
        patterns.append(pattern)

    n_patterns = len(patterns)

    # Show example pattern extraction
    ex_window = values[:order]
    ex_pattern = tuple(np.argsort(ex_window))

    deriv.add_step(
        title="Extract Ordinal Patterns",
        equation="πᵢ = rank permutation of [xᵢ, xᵢ₊τ, ..., xᵢ₊(m-1)τ]",
        calculation=f"Example (i=0):\n  Window: [{ex_window[0]:.4f}, {ex_window[1]:.4f}, {ex_window[2]:.4f}]\n  Sorted indices: {ex_pattern}\n  Pattern π₀ = {ex_pattern}\n\nTotal patterns extracted: {n_patterns}",
        result=n_patterns,
        result_name="N",
        notes=f"Possible patterns: {order}! = {factorial(order)}"
    )

    # Step 3: Count pattern frequencies
    counts = Counter(patterns)
    n_unique = len(counts)

    # Show top patterns
    top_patterns = counts.most_common(5)
    pattern_str = "\n".join([f"  {p}: count = {c}, prob = {c/n_patterns:.4f}" for p, c in top_patterns])
    if len(counts) > 5:
        pattern_str += "\n  ⋮"

    deriv.add_step(
        title="Count Pattern Frequencies",
        equation="p(π) = count(π) / N",
        calculation=f"Unique patterns observed: {n_unique} / {factorial(order)} possible\n\nTop patterns:\n{pattern_str}",
        result=n_unique,
        result_name="n_unique"
    )

    # Step 4: Compute Shannon entropy
    probs = [count / n_patterns for count in counts.values()]
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    max_entropy = np.log2(factorial(order))

    deriv.add_step(
        title="Compute Shannon Entropy",
        equation="H = -Σ p(π) × log₂(p(π))",
        calculation=f"H = -({probs[0]:.4f} × log₂({probs[0]:.4f}) + {probs[1]:.4f} × log₂({probs[1]:.4f}) + ...)\nH = {entropy:.6f} bits",
        result=entropy,
        result_name="H"
    )

    # Step 5: Normalize
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    deriv.add_step(
        title="Normalize by Maximum Entropy",
        equation="PE = H / H_max = H / log₂(m!)",
        calculation=f"H_max = log₂({order}!) = log₂({factorial(order)}) = {max_entropy:.6f}\nPE = {entropy:.6f} / {max_entropy:.6f} = {normalized_entropy:.6f}",
        result=normalized_entropy,
        result_name="PE",
        notes="Normalized PE ∈ [0, 1]: 0 = deterministic, 1 = random"
    )

    deriv.final_result = normalized_entropy
    deriv.prism_output = normalized_entropy

    # Interpretation
    if normalized_entropy < 0.3:
        interp = f"PE = {normalized_entropy:.4f} < 0.3 indicates **highly deterministic** dynamics. Strong ordinal structure."
    elif normalized_entropy < 0.7:
        interp = f"PE = {normalized_entropy:.4f} indicates **moderate complexity**. Mix of deterministic and stochastic components."
    else:
        interp = f"PE = {normalized_entropy:.4f} > 0.7 indicates **high randomness**. Ordinal patterns are nearly uniform."

    deriv.interpretation = interp

    return {"permutation_entropy": normalized_entropy}, deriv


def compute_entropy(values: np.ndarray) -> dict:
    """
    Measure entropy of a single signal.

    Args:
        values: Array of observed values (native sampling)

    Returns:
        Dict of metric_name -> metric_value
    """
    if len(values) < 20:
        return {}

    # Remove NaNs
    values = values[~np.isnan(values)]
    if len(values) < 20:
        return {}

    result = {}

    if HAS_ANTROPY:
        # Fast path using antropy
        try:
            perm_ent = ant.perm_entropy(values, order=3, normalize=True)
            if not np.isnan(perm_ent):
                result['permutation_entropy'] = float(perm_ent)
        except Exception:
            pass

        try:
            samp_ent = ant.sample_entropy(values, order=2)
            if not np.isnan(samp_ent):
                result['sample_entropy'] = float(samp_ent)
        except Exception:
            pass
    else:
        # Slow fallback
        try:
            perm_entropy = _permutation_entropy(values, dim=3)
            if not np.isnan(perm_entropy):
                result['permutation_entropy'] = float(perm_entropy)
        except Exception:
            pass

        try:
            samp_entropy = _sample_entropy_vectorized(values, dim=2)
            if not np.isnan(samp_entropy):
                result['sample_entropy'] = float(samp_entropy)
        except Exception:
            pass

    return result


def _permutation_entropy(series: np.ndarray, dim: int, tau: int = 1) -> float:
    """Permutation entropy based on ordinal patterns."""
    from collections import Counter
    from math import factorial

    n = len(series)
    if n < dim * tau:
        return np.nan

    patterns = []
    for i in range(n - (dim - 1) * tau):
        window = series[i:i + dim * tau:tau]
        pattern = tuple(np.argsort(window))
        patterns.append(pattern)

    counts = Counter(patterns)
    n_patterns = len(patterns)
    probs = [count / n_patterns for count in counts.values()]

    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    max_entropy = np.log2(factorial(dim))

    return entropy / max_entropy if max_entropy > 0 else 0


def _sample_entropy_vectorized(series: np.ndarray, dim: int, r: float = None) -> float:
    """Sample entropy using vectorized distance calculation."""
    from scipy.spatial.distance import cdist

    n = len(series)
    if r is None:
        r = 0.2 * np.std(series)

    if r == 0 or n < dim + 1:
        return np.nan

    # NO DOWNSAMPLING - academic research grade
    # Use ALL templates for proper entropy calculation
    # Memory: 252² × 8 bytes = 496 KB (trivial for modern systems)
    indices = np.arange(n - dim)

    # Build templates
    templates_m = np.array([series[i:i + dim] for i in indices])
    
    indices_m1 = indices[indices < n - dim - 1]
    templates_m1 = np.array([series[i:i + dim + 1] for i in indices_m1])

    # Vectorized distance calculation (Chebyshev = max absolute diff)
    dist_m = cdist(templates_m, templates_m, metric='chebyshev')
    dist_m1 = cdist(templates_m1, templates_m1, metric='chebyshev')

    # Count matches (excluding diagonal / self-matches)
    np.fill_diagonal(dist_m, np.inf)
    np.fill_diagonal(dist_m1, np.inf)

    B = np.sum(dist_m < r) // 2  # divide by 2 for symmetric
    A = np.sum(dist_m1 < r) // 2

    if B == 0:
        return np.nan

    return -np.log(A / B) if A > 0 else np.nan