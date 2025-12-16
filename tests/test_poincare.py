"""Tests for Poincaré ball geometry operations."""

import pytest
import torch

from hyperbolicbo.geometry.poincare import (
    poincare_distance,
    poincare_distance_matrix,
    mobius_add,
    mobius_scalar_mul,
    exp_map,
    log_map,
    project_to_ball,
    geodesic,
    hyperbolic_midpoint,
    adaptive_dimension,
    MAX_NORM,
)


class TestPoincareDistance:
    """Tests for poincare_distance function."""
    
    def test_same_point_zero_distance(self):
        """Distance from point to itself is zero."""
        x = torch.tensor([0.3, 0.4])
        d = poincare_distance(x, x)
        assert d.item() == pytest.approx(0.0, abs=1e-5)
    
    def test_origin_distance(self):
        """Distance from origin follows simpler formula."""
        origin = torch.tensor([0.0, 0.0])
        x = torch.tensor([0.5, 0.0])
        d = poincare_distance(origin, x)
        
        # For origin: d(0, x) = 2 * artanh(||x||)
        expected = 2 * torch.atanh(torch.tensor(0.5))
        assert d.item() == pytest.approx(expected.item(), abs=1e-5)
    
    def test_symmetry(self):
        """Distance is symmetric."""
        x = torch.tensor([0.3, 0.2])
        y = torch.tensor([-0.1, 0.4])
        assert poincare_distance(x, y).item() == pytest.approx(
            poincare_distance(y, x).item(), abs=1e-6
        )
    
    def test_triangle_inequality(self):
        """Triangle inequality holds."""
        x = torch.tensor([0.1, 0.2])
        y = torch.tensor([0.3, -0.1])
        z = torch.tensor([-0.2, 0.3])
        
        d_xy = poincare_distance(x, y)
        d_yz = poincare_distance(y, z)
        d_xz = poincare_distance(x, z)
        
        assert d_xz.item() <= d_xy.item() + d_yz.item() + 1e-5
    
    def test_boundary_large_distance(self):
        """Points near boundary have large distance."""
        x = torch.tensor([0.99, 0.0])
        y = torch.tensor([-0.99, 0.0])
        d = poincare_distance(x, y)
        
        # Should be much larger than Euclidean distance (~2)
        assert d.item() > 5.0


class TestDistanceMatrix:
    """Tests for pairwise distance matrix."""
    
    def test_matrix_shape(self):
        """Matrix has correct shape."""
        X = torch.randn(5, 3) * 0.3
        Y = torch.randn(7, 3) * 0.3
        
        D = poincare_distance_matrix(X, Y)
        assert D.shape == (5, 7)
    
    def test_self_distance_diagonal_zero(self):
        """Diagonal is zero for self-distance matrix."""
        X = torch.randn(10, 2) * 0.3
        D = poincare_distance_matrix(X)
        
        assert torch.allclose(D.diag(), torch.zeros(10), atol=1e-5)


class TestMobiusOperations:
    """Tests for Möbius addition and scalar multiplication."""
    
    def test_mobius_add_identity(self):
        """Adding zero is identity."""
        x = torch.tensor([0.3, 0.4])
        zero = torch.tensor([0.0, 0.0])
        
        result = mobius_add(x, zero)
        assert torch.allclose(result, x, atol=1e-5)
    
    def test_mobius_add_stays_in_ball(self):
        """Result stays inside ball."""
        x = torch.tensor([0.8, 0.0])
        y = torch.tensor([0.0, 0.8])
        
        result = mobius_add(x, y)
        assert result.norm().item() < 1.0
    
    def test_mobius_scalar_identity(self):
        """Scalar 1 is identity."""
        x = torch.tensor([0.3, 0.4])
        result = mobius_scalar_mul(1.0, x)
        assert torch.allclose(result, x, atol=1e-4)
    
    def test_mobius_scalar_zero(self):
        """Scalar 0 gives origin."""
        x = torch.tensor([0.3, 0.4])
        result = mobius_scalar_mul(0.0, x)
        assert torch.allclose(result, torch.zeros(2), atol=1e-5)


class TestExpLogMaps:
    """Tests for exponential and logarithmic maps."""
    
    def test_exp_log_inverse(self):
        """Exp and log are inverses."""
        x = torch.tensor([0.2, 0.1])
        y = torch.tensor([0.4, -0.2])
        
        v = log_map(y, x)
        y_recovered = exp_map(v, x)
        
        assert torch.allclose(y_recovered, y, atol=1e-4)
    
    def test_exp_from_origin(self):
        """Exp from origin for tangent vector."""
        v = torch.tensor([0.5, 0.0])
        result = exp_map(v)
        
        # Should project along v direction
        assert result[0].item() > 0
        assert result.norm().item() < 1.0


class TestGeodesic:
    """Tests for geodesic computation."""
    
    def test_geodesic_endpoints(self):
        """Geodesic at t=0 and t=1 gives endpoints."""
        x = torch.tensor([0.1, 0.2])
        y = torch.tensor([0.4, -0.1])
        
        g0 = geodesic(x, y, 0.0)
        g1 = geodesic(x, y, 1.0)
        
        assert torch.allclose(g0, x, atol=1e-4)
        assert torch.allclose(g1, y, atol=1e-4)
    
    def test_midpoint_equidistant(self):
        """Midpoint is equidistant from endpoints."""
        x = torch.tensor([0.1, 0.3])
        y = torch.tensor([0.5, -0.2])
        
        mid = hyperbolic_midpoint(x, y)
        
        d1 = poincare_distance(x, mid)
        d2 = poincare_distance(mid, y)
        
        assert d1.item() == pytest.approx(d2.item(), abs=1e-4)


class TestProjection:
    """Tests for projection to ball."""
    
    def test_inside_unchanged(self):
        """Points inside ball are unchanged."""
        x = torch.tensor([0.3, 0.4])
        result = project_to_ball(x)
        assert torch.allclose(result, x, atol=1e-6)
    
    def test_outside_projected(self):
        """Points outside ball are projected in."""
        x = torch.tensor([2.0, 3.0])
        result = project_to_ball(x)
        assert result.norm().item() < 1.0


class TestAdaptiveDimension:
    """Tests for adaptive dimension selection."""
    
    def test_small_degree(self):
        """Small degree gets 2D."""
        assert adaptive_dimension(2) == 2
    
    def test_medium_degree(self):
        """Medium degree gets appropriate dim."""
        assert adaptive_dimension(8) == 3
        assert adaptive_dimension(16) == 4
    
    def test_large_degree_capped(self):
        """Large degree is capped at 8."""
        assert adaptive_dimension(1000) == 8
