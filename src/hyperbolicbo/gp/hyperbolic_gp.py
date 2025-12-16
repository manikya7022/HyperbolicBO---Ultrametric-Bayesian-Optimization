"""Hyperbolic Gaussian Process implementation.

A GP with hyperbolic kernel for tree-structured optimization.
Uses horosphere clustering for O(1) prediction approximation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from torch import Tensor
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from .kernels import HyperbolicRBFKernel
from ..geometry.poincare import project_to_ball
from ..geometry.horosphere import HorosphereIndex


class HyperbolicGPModel(ExactGP):
    """Exact GP with hyperbolic RBF kernel.
    
    This is a GPyTorch model that uses hyperbolic distance
    in its kernel computation.
    """
    
    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        likelihood: GaussianLikelihood,
        kernel_type: str = "rbf",
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        
        if kernel_type == "rbf":
            self.covar_module = gpytorch.kernels.ScaleKernel(
                HyperbolicRBFKernel()
            )
        else:
            raise ValueError(f"Unsupported kernel: {kernel_type}")
    
    def forward(self, x: Tensor) -> MultivariateNormal:
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)


@dataclass
class HyperbolicGP:
    """High-level Hyperbolic Gaussian Process.
    
    Wraps GPyTorch model with:
    - Automatic training (hyperparameter optimization)
    - Horosphere index for O(1) approximate prediction
    - Convenient fit/predict API
    
    Attributes:
        dim: Poincaré ball dimension
        n_train_iters: Training iterations for hyperparameters
        use_horosphere: Whether to use O(1) approximate predictions
        device: Torch device (cpu/cuda)
    """
    
    dim: int = 2
    n_train_iters: int = 50
    use_horosphere: bool = True
    device: str = "cpu"
    
    # Internal state
    _model: Optional[HyperbolicGPModel] = field(default=None, repr=False)
    _likelihood: Optional[GaussianLikelihood] = field(default=None, repr=False)
    _X: Optional[Tensor] = field(default=None, repr=False)
    _y: Optional[Tensor] = field(default=None, repr=False)
    _horosphere_index: Optional[HorosphereIndex] = field(default=None, repr=False)
    
    def fit(
        self,
        X: Tensor,
        y: Tensor,
        verbose: bool = False,
    ) -> "HyperbolicGP":
        """Fit GP to observations.
        
        Args:
            X: Observed points in Poincaré ball, shape (n, dim)
            y: Observed values, shape (n,)
            verbose: Print training progress
            
        Returns:
            Self for chaining
        """
        X = project_to_ball(X.to(self.device))
        y = y.to(self.device)
        
        self._X = X
        self._y = y
        
        # Initialize model
        self._likelihood = GaussianLikelihood().to(self.device)
        self._model = HyperbolicGPModel(X, y, self._likelihood).to(self.device)
        
        # Train hyperparameters
        self._model.train()
        self._likelihood.train()
        
        optimizer = torch.optim.Adam(self._model.parameters(), lr=0.1)
        mll = ExactMarginalLogLikelihood(self._likelihood, self._model)
        
        for i in range(self.n_train_iters):
            optimizer.zero_grad()
            output = self._model(X)
            loss = -mll(output, y)
            loss.backward()
            optimizer.step()
            
            if verbose and (i + 1) % 10 == 0:
                print(f"Iter {i+1}/{self.n_train_iters}, Loss: {loss.item():.4f}")
        
        # Build horosphere index for fast queries
        if self.use_horosphere:
            self._horosphere_index = HorosphereIndex.build(X)
        
        return self
    
    def predict(
        self,
        X: Tensor,
        return_std: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Predict mean and variance at new points.
        
        Args:
            X: Query points in Poincaré ball, shape (m, dim)
            return_std: Whether to return standard deviation
            
        Returns:
            Tuple of (mean, std) each of shape (m,), or just mean
        """
        if self._model is None:
            raise RuntimeError("Must call fit() before predict()")
        
        X = project_to_ball(X.to(self.device))
        
        self._model.eval()
        self._likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self._likelihood(self._model(X))
            mean = pred.mean
            
            if return_std:
                std = pred.stddev
                return mean, std
            return mean, None
    
    def predict_approximate(
        self,
        X: Tensor,
        k_neighbors: int = 5,
    ) -> Tuple[Tensor, Tensor]:
        """O(1) approximate prediction using horosphere index.
        
        Uses only k nearest neighbors for local GP prediction.
        Much faster than full GP for large datasets.
        
        Args:
            X: Query points, shape (m, dim)
            k_neighbors: Number of neighbors to use
            
        Returns:
            Tuple of (mean, std)
        """
        if self._horosphere_index is None:
            return self.predict(X)
        
        X = project_to_ball(X.to(self.device))
        m = X.shape[0]
        
        means = torch.zeros(m, device=self.device)
        stds = torch.zeros(m, device=self.device)
        
        # Query nearest neighbors via horosphere
        indices, _ = self._horosphere_index.query(X, k=k_neighbors)
        
        for i in range(m):
            # Local GP with only nearest neighbors
            idx = indices[i]
            local_X = self._X[idx]
            local_y = self._y[idx]
            
            # Simple local regression
            local_gp = HyperbolicGP(dim=self.dim, n_train_iters=10, use_horosphere=False)
            local_gp.fit(local_X, local_y)
            mean, std = local_gp.predict(X[i:i+1])
            
            means[i] = mean.squeeze()
            stds[i] = std.squeeze() if std is not None else 0.0
        
        return means, stds
    
    def update(
        self,
        X_new: Tensor,
        y_new: Tensor,
        refit: bool = True,
    ) -> "HyperbolicGP":
        """Add new observations and optionally refit.
        
        Args:
            X_new: New observed points, shape (m, dim)
            y_new: New observed values, shape (m,)
            refit: Whether to retrain hyperparameters
            
        Returns:
            Self for chaining
        """
        X_new = project_to_ball(X_new.to(self.device))
        y_new = y_new.to(self.device)
        
        if self._X is None:
            self._X = X_new
            self._y = y_new
        else:
            self._X = torch.cat([self._X, X_new], dim=0)
            self._y = torch.cat([self._y, y_new], dim=0)
        
        if refit:
            self.fit(self._X, self._y)
        else:
            # Just update horosphere index
            if self.use_horosphere:
                self._horosphere_index = HorosphereIndex.build(self._X)
        
        return self
    
    @property
    def X(self) -> Optional[Tensor]:
        """Observed points."""
        return self._X
    
    @property
    def y(self) -> Optional[Tensor]:
        """Observed values."""
        return self._y
    
    @property
    def n_observations(self) -> int:
        """Number of observations."""
        return 0 if self._X is None else len(self._X)
    
    def acquisition_variance(self, candidates: Tensor) -> Tensor:
        """Compute variance for adaptive batch sizing.
        
        Higher variance = more uncertainty = larger batch beneficial.
        
        Args:
            candidates: Candidate points, shape (m, dim)
            
        Returns:
            Mean variance across candidates
        """
        _, stds = self.predict(candidates)
        return stds.pow(2).mean() if stds is not None else torch.tensor(0.0)
