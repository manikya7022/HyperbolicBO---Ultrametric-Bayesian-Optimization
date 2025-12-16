"""Hyperbolic kernels for Gaussian Processes.

Kernels define covariance between points in hyperbolic space.
Key insight: standard RBF kernel K(x,y) = exp(-||x-y||²/2l²) becomes
hyperbolic RBF: K(x,y) = exp(-d_H(x,y)²/2l²) where d_H is Poincaré distance.

This makes the GP naturally suited for tree-structured data.
"""

from __future__ import annotations

import torch
from torch import Tensor
import gpytorch
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive

from ..geometry.poincare import poincare_distance, poincare_distance_matrix


class HyperbolicRBFKernel(Kernel):
    """RBF kernel using hyperbolic (Poincaré) distance.
    
    K(x, x') = σ² × exp(-d_H(x, x')² / (2l²))
    
    where d_H is the Poincaré ball distance:
    d_H(u,v) = arcosh(1 + 2||u-v||² / ((1-||u||²)(1-||v||²)))
    
    This kernel captures tree structure naturally: points deep in the
    Poincaré ball (near boundary) have larger mutual distances,
    reflecting their position in the tree hierarchy.
    
    Attributes:
        lengthscale: Controls kernel bandwidth (how quickly covariance decays)
        outputscale: Variance multiplier σ²
    """
    
    has_lengthscale = True
    
    def __init__(
        self,
        ard_num_dims: int | None = None,
        batch_shape: torch.Size = torch.Size([]),
        **kwargs,
    ):
        """Initialize hyperbolic RBF kernel.
        
        Args:
            ard_num_dims: Number of ARD dimensions (None = isotropic)
            batch_shape: Batch dimensions for multiple kernels
        """
        super().__init__(
            has_lengthscale=True,
            ard_num_dims=ard_num_dims,
            batch_shape=batch_shape,
            **kwargs,
        )
    
    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **kwargs,
    ) -> Tensor:
        """Compute kernel matrix.
        
        Args:
            x1: First set of points (..., n, d)
            x2: Second set of points (..., m, d)
            diag: If True, return only diagonal of kernel matrix
            last_dim_is_batch: If True, last dim is batch not feature
            
        Returns:
            Kernel matrix of shape (..., n, m) or (..., n) if diag=True
        """
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2)
            x2 = x2.transpose(-1, -2)
        
        if diag:
            # Diagonal: self-covariance is always 1 (before outputscale)
            dist = poincare_distance(x1, x2)
        else:
            # Full matrix
            dist = poincare_distance_matrix(x1, x2)
        
        # RBF: exp(-d² / 2l²)
        scaled_dist_sq = dist.pow(2) / (2.0 * self.lengthscale.pow(2))
        
        return torch.exp(-scaled_dist_sq)


class HyperbolicMaternKernel(Kernel):
    """Matérn kernel using hyperbolic distance.
    
    The Matérn kernel with smoothness ν:
    K(x, x') = σ² × (2^(1-ν)/Γ(ν)) × (√(2ν)d/l)^ν × K_ν(√(2ν)d/l)
    
    For ν = 1/2: K = exp(-d/l) (Laplace)
    For ν = 3/2: K = (1 + √3 d/l) exp(-√3 d/l)
    For ν = 5/2: K = (1 + √5 d/l + 5d²/3l²) exp(-√5 d/l)
    
    We implement ν = 5/2 for smooth functions (common in BO).
    """
    
    has_lengthscale = True
    
    def __init__(
        self,
        nu: float = 2.5,
        batch_shape: torch.Size = torch.Size([]),
        **kwargs,
    ):
        """Initialize hyperbolic Matérn kernel.
        
        Args:
            nu: Smoothness parameter (0.5, 1.5, or 2.5)
            batch_shape: Batch dimensions
        """
        super().__init__(
            has_lengthscale=True,
            batch_shape=batch_shape,
            **kwargs,
        )
        self.nu = nu
    
    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        **kwargs,
    ) -> Tensor:
        """Compute Matérn kernel matrix using hyperbolic distance."""
        if diag:
            dist = poincare_distance(x1, x2)
        else:
            dist = poincare_distance_matrix(x1, x2)
        
        scaled_dist = dist / self.lengthscale
        
        if self.nu == 0.5:
            # Laplace
            return torch.exp(-scaled_dist)
        elif self.nu == 1.5:
            # Matérn 3/2
            sqrt3_d = 1.7320508075688772 * scaled_dist  # √3
            return (1.0 + sqrt3_d) * torch.exp(-sqrt3_d)
        elif self.nu == 2.5:
            # Matérn 5/2
            sqrt5_d = 2.23606797749979 * scaled_dist  # √5
            return (1.0 + sqrt5_d + sqrt5_d.pow(2) / 3.0) * torch.exp(-sqrt5_d)
        else:
            raise ValueError(f"Unsupported nu={self.nu}. Use 0.5, 1.5, or 2.5")


class HyperbolicPeriodicKernel(Kernel):
    """Periodic kernel on the Poincaré disk.
    
    Useful when the search space has circular/periodic structure
    (e.g., rotations in architecture search).
    
    K(x, x') = exp(-2 sin²(π d_H / p) / l²)
    
    where p is the period.
    """
    
    has_lengthscale = True
    
    def __init__(
        self,
        period_length: float = 1.0,
        batch_shape: torch.Size = torch.Size([]),
        **kwargs,
    ):
        super().__init__(
            has_lengthscale=True,
            batch_shape=batch_shape,
            **kwargs,
        )
        self.register_parameter(
            "raw_period_length",
            torch.nn.Parameter(torch.tensor(period_length).log()),
        )
        self.register_constraint("raw_period_length", Positive())
    
    @property
    def period_length(self) -> Tensor:
        return self.raw_period_length_constraint.transform(self.raw_period_length)
    
    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        **kwargs,
    ) -> Tensor:
        if diag:
            dist = poincare_distance(x1, x2)
        else:
            dist = poincare_distance_matrix(x1, x2)
        
        sin_term = torch.sin(3.14159265359 * dist / self.period_length)
        
        return torch.exp(-2.0 * sin_term.pow(2) / self.lengthscale.pow(2))


class AdditiveHyperbolicKernel(Kernel):
    """Additive kernel over hyperbolic dimensions.
    
    K(x, x') = Σ_i K_i(x_i, x_i')
    
    Useful when different dimensions have different scales/importance.
    Implements sum of 1D hyperbolic RBF kernels with separate lengthscales.
    """
    
    def __init__(
        self,
        num_dims: int,
        batch_shape: torch.Size = torch.Size([]),
        **kwargs,
    ):
        super().__init__(batch_shape=batch_shape, **kwargs)
        self.num_dims = num_dims
        
        self.register_parameter(
            "raw_lengthscales",
            torch.nn.Parameter(torch.zeros(num_dims)),
        )
        self.register_constraint("raw_lengthscales", Positive())
    
    @property
    def lengthscales(self) -> Tensor:
        return self.raw_lengthscales_constraint.transform(self.raw_lengthscales)
    
    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        **kwargs,
    ) -> Tensor:
        result = None
        
        for i in range(self.num_dims):
            # Extract i-th dimension and extend to 2D for poincare_distance
            x1_i = x1[..., i:i+1]
            x2_i = x2[..., i:i+1]
            
            if diag:
                dist_i = (x1_i - x2_i).abs().squeeze(-1)
            else:
                dist_i = torch.cdist(x1_i, x2_i).squeeze(-1)
            
            k_i = torch.exp(-dist_i.pow(2) / (2.0 * self.lengthscales[i].pow(2)))
            
            if result is None:
                result = k_i
            else:
                result = result + k_i
        
        return result / self.num_dims  # Normalize


def create_hyperbolic_kernel(
    kernel_type: str = "rbf",
    **kwargs,
) -> Kernel:
    """Factory function for creating hyperbolic kernels.
    
    Args:
        kernel_type: One of "rbf", "matern", "periodic", "additive"
        **kwargs: Passed to kernel constructor
        
    Returns:
        Configured kernel instance
    """
    kernels = {
        "rbf": HyperbolicRBFKernel,
        "matern": HyperbolicMaternKernel,
        "periodic": HyperbolicPeriodicKernel,
        "additive": AdditiveHyperbolicKernel,
    }
    
    if kernel_type not in kernels:
        raise ValueError(f"Unknown kernel type: {kernel_type}. Choose from {list(kernels)}")
    
    return kernels[kernel_type](**kwargs)
