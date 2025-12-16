"""Tests for HyperbolicGP."""

import pytest
import torch

from hyperbolicbo.gp.hyperbolic_gp import HyperbolicGP
from hyperbolicbo.gp.kernels import HyperbolicRBFKernel


class TestHyperbolicGP:
    """Tests for HyperbolicGP class."""
    
    @pytest.fixture
    def simple_data(self):
        """Simple 2D training data."""
        X = torch.tensor([
            [0.1, 0.1],
            [0.3, 0.0],
            [-0.2, 0.2],
            [0.0, -0.3],
        ])
        y = torch.tensor([1.0, 2.0, 0.5, 1.5])
        return X, y
    
    def test_fit_stores_data(self, simple_data):
        """Fit stores training data."""
        X, y = simple_data
        gp = HyperbolicGP(dim=2)
        gp.fit(X, y)
        
        assert gp.n_observations == 4
        assert gp.X is not None
        assert gp.y is not None
    
    def test_predict_shape(self, simple_data):
        """Predict returns correct shapes."""
        X, y = simple_data
        gp = HyperbolicGP(dim=2)
        gp.fit(X, y)
        
        X_test = torch.randn(5, 2) * 0.3
        mean, std = gp.predict(X_test)
        
        assert mean.shape == (5,)
        assert std.shape == (5,)
    
    def test_predict_interpolation(self, simple_data):
        """Predictions at training points are reasonable."""
        X, y = simple_data
        gp = HyperbolicGP(dim=2, n_train_iters=100)
        gp.fit(X, y)
        
        mean, _ = gp.predict(X)
        
        # Should be reasonably close (GP may not interpolate exactly)
        # Check mean is in expected range
        assert mean.mean().item() > 0.5
        assert mean.std().item() < 1.0
    
    def test_update_adds_observations(self, simple_data):
        """Update adds new observations."""
        X, y = simple_data
        gp = HyperbolicGP(dim=2)
        gp.fit(X, y)
        
        X_new = torch.tensor([[0.5, 0.0]])
        y_new = torch.tensor([3.0])
        gp.update(X_new, y_new)
        
        assert gp.n_observations == 5
    
    def test_acquisition_variance(self, simple_data):
        """Acquisition variance is computed correctly."""
        X, y = simple_data
        gp = HyperbolicGP(dim=2)
        gp.fit(X, y)
        
        candidates = torch.randn(10, 2) * 0.3
        var = gp.acquisition_variance(candidates)
        
        assert var.item() >= 0


class TestHyperbolicRBFKernel:
    """Tests for HyperbolicRBFKernel."""
    
    def test_kernel_positive(self):
        """Kernel values are positive."""
        kernel = HyperbolicRBFKernel()
        
        X = torch.randn(5, 2) * 0.3
        K = kernel(X, X).evaluate()
        
        assert (K >= 0).all()
    
    def test_kernel_diagonal_one(self):
        """Diagonal elements are 1 (before scaling)."""
        kernel = HyperbolicRBFKernel()
        
        X = torch.randn(5, 2) * 0.3
        K = kernel(X, X).evaluate()
        
        # Diagonal should be close to 1 (same point → distance 0 → exp(0) = 1)
        assert torch.allclose(K.diag(), torch.ones(5), atol=0.1)
    
    def test_kernel_symmetric(self):
        """Kernel matrix is symmetric."""
        kernel = HyperbolicRBFKernel()
        
        X = torch.randn(5, 2) * 0.3
        K = kernel(X, X).evaluate()
        
        assert torch.allclose(K, K.t(), atol=1e-5)
