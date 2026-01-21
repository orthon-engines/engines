"""
Test Laplace Module Imports
===========================

Quick verification that the reorganized laplace modules import correctly.
"""

import pytest
import numpy as np


class TestLaplaceImports:
    """Test all laplace module imports work correctly."""

    def test_inline_laplace_imports(self):
        """Test prism.modules.laplace_transform (discrete operators)."""
        from orthon._internal.engines.laplace.transform import (
            compute_laplace_for_series,
            compute_gradient,
            compute_laplacian,
        )
        assert callable(compute_laplace_for_series)
        assert callable(compute_gradient)
        assert callable(compute_laplacian)

    def test_laplace_transform_imports(self):
        """Test prism.modules.laplace_transform (RunningLaplace)."""
        from orthon._internal.engines.laplace.transform import (
            RunningLaplace,
            compute_laplace_field,
            laplace_gradient,
            laplace_divergence,
            laplace_energy,
            decompose_by_scale,
        )
        assert RunningLaplace is not None
        assert callable(compute_laplace_field)
        assert callable(laplace_gradient)
        assert callable(laplace_divergence)
        assert callable(laplace_energy)
        assert callable(decompose_by_scale)

    def test_laplace_compute_imports(self):
        """Test prism.entry_points.laplace (CLI utilities)."""
        from orthon._internal.entry_points.laplace import WindowConfig
        assert WindowConfig is not None

    def test_laplace_pairwise_imports(self):
        """Test prism.modules.laplace_pairwise (vectorized pairwise)."""
        from orthon._internal.engines.laplace.pairwise import (
            run_laplace_pairwise_vectorized,
            run_laplace_pairwise_windowed,
        )
        assert callable(run_laplace_pairwise_vectorized)
        assert callable(run_laplace_pairwise_windowed)

    def test_modules_package_exports(self):
        """Test prism.modules package exports laplace functions."""
        from orthon._internal.engines import (
            RunningLaplace,
            compute_laplace_field,
            compute_laplace_for_series,
            compute_gradient,
            compute_laplacian,
        )
        assert RunningLaplace is not None
        assert callable(compute_laplace_field)
        assert callable(compute_laplace_for_series)

    def test_entry_points_exports(self):
        """Test prism.entry_points.laplace exports."""
        from orthon._internal.entry_points.laplace import WindowConfig, compute_laplace_field
        assert callable(compute_laplace_field)
        assert WindowConfig is not None


class TestLaplaceComputation:
    """Test basic laplace computations work."""

    def test_compute_gradient(self):
        """Test gradient computation."""
        from orthon._internal.engines.laplace.transform import compute_gradient

        values = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
        gradient = compute_gradient(values)

        assert len(gradient) == len(values)
        assert not np.all(np.isnan(gradient))
        # Interior points should have central difference
        assert np.isclose(gradient[2], (7.0 - 2.0) / 2.0)

    def test_compute_laplacian(self):
        """Test laplacian computation."""
        from orthon._internal.engines.laplace.transform import compute_laplacian

        values = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
        laplacian = compute_laplacian(values)

        assert len(laplacian) == len(values)
        assert not np.all(np.isnan(laplacian))
        # Interior: f(t+1) - 2*f(t) + f(t-1)
        assert np.isclose(laplacian[2], 7.0 - 2*4.0 + 2.0)

    def test_running_laplace(self):
        """Test RunningLaplace incremental computation."""
        from orthon._internal.engines.laplace.transform import RunningLaplace

        laplace = RunningLaplace(s_values=np.array([0.1, 1.0]))
        laplace.reset("test_signal")

        # Feed some observations
        for i, v in enumerate([1.0, 2.0, 3.0, 2.5, 2.0]):
            laplace.update(float(i), v)

        field = laplace.get_field()
        assert field.signal_id == "test_signal"
        assert len(field.timestamps) == 5
        assert field.field.shape == (5, 2)  # 5 timestamps, 2 s-values


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
