"""Parallelized Thompson Sampling for hyperbolic space.

Thompson Sampling: sample from posterior, optimize sample, propose point.

Key innovation: use hyperbolic Fourier features to avoid O(n³) matrix inversion.
Random Fourier Features approximate the kernel:
    K(x, x') ≈ φ(x)ᵀφ(x')
    
For hyperbolic kernel, we use the logarithmic map to compute features.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
from torch import Tensor
import numpy as np

from ..gp.hyperbolic_gp import HyperbolicGP
from ..geometry.poincare import log_map, project_to_ball, poincare_distance


class HyperbolicFourierFeatures:
    """Random Fourier Features for hyperbolic RBF kernel.
    
    Approximates K(x, x') = exp(-d_H(x,x')²/2l²) via:
        K(x, x') ≈ φ(x)ᵀφ(x') / n_features
        
    where φ(x) = [cos(ωᵢᵀ log_0(x) + bᵢ)]_{i=1}^n_features
    
    and ω ~ N(0, 1/l²), b ~ Uniform(0, 2π)
    """
    
    def __init__(
        self,
        n_features: int = 100,
        lengthscale: float = 1.0,
        dim: int = 2,
        device: str = "cpu",
    ):
        """Initialize random Fourier features.
        
        Args:
            n_features: Number of random features
            lengthscale: Kernel lengthscale
            dim: Input dimension
            device: Torch device
        """
        self.n_features = n_features
        self.lengthscale = lengthscale
        self.dim = dim
        self.device = device
        
        # Sample random frequencies
        self.omega = torch.randn(dim, n_features, device=device) / lengthscale
        self.bias = torch.rand(n_features, device=device) * 2 * np.pi
    
    def transform(self, X: Tensor) -> Tensor:
        """Transform points to feature space.
        
        Args:
            X: Points in Poincaré ball, shape (n, d)
            
        Returns:
            Features, shape (n, n_features)
        """
        X = project_to_ball(X.to(self.device))
        
        # Use log map from origin to get tangent vectors
        tangent = log_map(X)  # Shape: (n, d)
        
        # Ensure consistent dtype
        tangent = tangent.float()
        omega = self.omega.float()
        bias = self.bias.float()
        
        # Random Fourier features: cos(ωᵀx + b)
        proj = tangent @ omega  # Shape: (n, n_features)
        features = torch.cos(proj + bias)
        
        return features * np.sqrt(2.0 / self.n_features)
    
    def approximate_kernel(self, X1: Tensor, X2: Tensor) -> Tensor:
        """Approximate kernel matrix via dot product of features.
        
        Args:
            X1: First set of points, shape (n, d)
            X2: Second set of points, shape (m, d)
            
        Returns:
            Approximate kernel matrix, shape (n, m)
        """
        phi1 = self.transform(X1)
        phi2 = self.transform(X2)
        return phi1 @ phi2.t()


def adaptive_batch_size(
    gp: HyperbolicGP,
    candidates: Tensor,
    max_batch: int = 8,
    min_batch: int = 2,
) -> int:
    """Compute adaptive batch size based on GP variance.
    
    Higher uncertainty → larger batch for more exploration.
    
    batch = min(max_batch, min_batch + var/var_median * max_batch)
    
    Args:
        gp: Fitted HyperbolicGP
        candidates: Candidate points
        max_batch: Maximum batch size
        min_batch: Minimum batch size
        
    Returns:
        Recommended batch size
    """
    if gp.n_observations < 5:
        return max_batch  # Early exploration
    
    var = gp.acquisition_variance(candidates)
    
    # Estimate median variance from random samples
    n_samples = min(100, len(candidates))
    idx = torch.randperm(len(candidates))[:n_samples]
    var_median = gp.acquisition_variance(candidates[idx])
    
    if var_median < 1e-8:
        return min_batch
    
    batch = int(min(max_batch, min_batch + (var / var_median) * max_batch))
    return max(min_batch, min(max_batch, batch))


def parallel_hyperbolic_ts(
    gp: HyperbolicGP,
    candidates: Tensor,
    n_samples: int,
    n_features: int = 100,
) -> Tensor:
    """Parallelized Thompson Sampling via hyperbolic Fourier features.
    
    Algorithm:
    1. Compute Fourier features φ(X) for training data
    2. Solve for weights: w = (ΦᵀΦ)⁻¹Φᵀy (O(n) if n_features << n)
    3. Sample random functions: f ~ GP via perturbed weights
    4. Return argmax for each sample
    
    Args:
        gp: Fitted HyperbolicGP with observations
        candidates: Candidate points, shape (m, d)
        n_samples: Number of parallel samples (batch size)
        n_features: Number of random Fourier features
        
    Returns:
        Indices of selected candidates, shape (n_samples,)
    """
    if gp.X is None or gp.y is None:
        # No observations: random selection
        return torch.randperm(len(candidates))[:n_samples]
    
    device = gp.device if hasattr(gp, 'device') else 'cpu'
    candidates = project_to_ball(candidates.to(device))
    
    # Get training data
    X_train = gp.X
    y_train = gp.y
    
    # Estimate lengthscale from GP (or use default)
    lengthscale = 1.0
    if gp._model is not None:
        try:
            lengthscale = gp._model.covar_module.base_kernel.lengthscale.item()
        except:
            pass
    
    # Build Fourier features
    rff = HyperbolicFourierFeatures(
        n_features=n_features,
        lengthscale=lengthscale,
        dim=gp.dim,
        device=device,
    )
    
    # Transform training data
    Phi_train = rff.transform(X_train)  # (n, n_features)
    
    # Solve for base weights: w = (ΦᵀΦ + λI)⁻¹Φᵀy
    # Using regularized least squares for stability
    lambda_reg = 1e-4
    PhiTPhi = Phi_train.t() @ Phi_train  # (n_features, n_features)
    PhiTPhi += lambda_reg * torch.eye(n_features, device=device)
    
    PhiTy = Phi_train.t() @ y_train  # (n_features,)
    
    # Cholesky solve (O(n_features³) which is O(1) since n_features is fixed)
    L = torch.linalg.cholesky(PhiTPhi)
    w = torch.cholesky_solve(PhiTy.unsqueeze(-1), L).squeeze(-1)
    
    # Transform candidates
    Phi_cand = rff.transform(candidates)  # (m, n_features)
    
    # Base predictions
    f_base = Phi_cand @ w  # (m,)
    
    # Sample perturbations for Thompson sampling
    # Each sample is a different random function from the posterior
    noise_scale = 0.1  # Controls exploration
    
    # Sample noise for weights
    weight_noise = torch.randn(n_samples, n_features, device=device) * noise_scale
    
    # Perturbed predictions for each sample
    f_samples = f_base.unsqueeze(0) + (weight_noise @ Phi_cand.t())  # (n_samples, m)
    
    # Select argmax for each sample
    selected_indices = f_samples.argmax(dim=1)  # (n_samples,)
    
    # Remove duplicates by using unique indices
    unique_indices = torch.unique(selected_indices)
    
    if len(unique_indices) < n_samples:
        # Not enough unique: add random ones
        remaining = n_samples - len(unique_indices)
        mask = torch.ones(len(candidates), dtype=torch.bool, device=device)
        mask[unique_indices] = False
        available = mask.nonzero().squeeze(-1)
        
        if len(available) > 0:
            extra = available[torch.randperm(len(available))[:remaining]]
            unique_indices = torch.cat([unique_indices, extra])
    
    return unique_indices[:n_samples]


def thompson_sample_single(
    gp: HyperbolicGP,
    candidates: Tensor,
    n_features: int = 100,
) -> int:
    """Sample a single point via Thompson Sampling.
    
    Args:
        gp: Fitted HyperbolicGP
        candidates: Candidate points
        n_features: Fourier feature dimension
        
    Returns:
        Index of selected candidate
    """
    indices = parallel_hyperbolic_ts(gp, candidates, n_samples=1, n_features=n_features)
    return indices[0].item()


def diverse_thompson_sampling(
    gp: HyperbolicGP,
    candidates: Tensor,
    n_samples: int,
    diversity_weight: float = 0.5,
) -> Tensor:
    """Thompson Sampling with diversity bonus.
    
    Penalizes selecting points too close (in hyperbolic distance)
    to already-selected points.
    
    Args:
        gp: Fitted HyperbolicGP
        candidates: Candidate points
        n_samples: Batch size
        diversity_weight: How much to weight diversity (0-1)
        
    Returns:
        Indices of selected candidates
    """
    device = gp.device if hasattr(gp, 'device') else 'cpu'
    candidates = project_to_ball(candidates.to(device))
    m = len(candidates)
    
    # Get base Thompson samples
    base_samples = parallel_hyperbolic_ts(gp, candidates, n_samples * 2)
    
    selected = []
    mask = torch.ones(m, dtype=torch.bool, device=device)
    
    for idx in base_samples:
        if not mask[idx]:
            continue
        
        selected.append(idx.item())
        
        if len(selected) >= n_samples:
            break
        
        # Penalize nearby candidates
        dists = poincare_distance(candidates[idx:idx+1], candidates)
        nearby = dists < 0.5  # Threshold
        mask = mask & ~nearby.squeeze()
    
    # Fill remaining if needed
    while len(selected) < n_samples:
        remaining = mask.nonzero().squeeze(-1)
        if len(remaining) == 0:
            remaining = torch.arange(m, device=device)
        idx = remaining[torch.randint(len(remaining), (1,))].item()
        if idx not in selected:
            selected.append(idx)
    
    return torch.tensor(selected[:n_samples], device=device)
