"""
PRISM Derivation Framework
==========================

Generate step-by-step mathematical derivations with actual data values.
Creates institutional-grade documentation proving every calculation.

Usage:
    from orthon._internal.entry_points.derivations import Derivation, DerivationStep

    deriv = Derivation(
        engine_name="hurst_exponent",
        method_name="Rescaled Range (R/S) Analysis",
        signal_id="lorenz_x",
        window_id="47",
        sample_size=252
    )
    deriv.add_step(
        title="Compute Mean",
        equation="x̄ = (1/n) Σᵢ xᵢ",
        calculation="x̄ = (1.234 + 2.345 + ...) / 252",
        result=1.567,
        result_name="x̄"
    )
"""

from orthon._internal.entry_points.derivations.base import Derivation, DerivationStep

__all__ = ['Derivation', 'DerivationStep']
