"""Poincaré ball operations for hyperbolic geometry.

This module implements core operations on the Poincaré ball model of hyperbolic space.
The Poincaré ball B^n = {x ∈ ℝⁿ : ||x|| < 1} with the metric:

    ds² = 4 / (1 - ||x||²)² × ||dx||²

Key operations:
- Distance: d(u,v) = arcosh(1 + 2||u-v||² / ((1-||u||²)(1-||v||²)))
- Möbius addition: u ⊕ v = ((1+2⟨u,v⟩+||v||²)u + (1-||u||²)v) / (1+2⟨u,v⟩+||u||²||v||²)
- Exponential map: exp_x(v) projects tangent vector v at x onto the manifold
- Logarithmic map: log_x(y) gives tangent vector at x pointing toward y
"""

from __future__ import annotations

import torch
from torch import Tensor

# Numerical stability constants
EPS = 1e-15
MAX_NORM = 1.0 - 1e-5  # Clip to avoid boundary singularities


def _clamp_norm(x: Tensor, max_norm: float = MAX_NORM) -> Tensor:
    """Clamp tensor norm to stay within Poincaré ball.
    
    Args:
        x: Points in ℝⁿ
        max_norm: Maximum allowed norm (default 1-1e-5 for stability)
        
    Returns:
        Points with ||x|| < max_norm
    """
    norm = x.norm(dim=-1, keepdim=True)
    clamped = x * (max_norm / norm.clamp(min=max_norm))
    return torch.where(norm > max_norm, clamped, x)


def _lambda_x(x: Tensor) -> Tensor:
    """Conformal factor λ_x = 2 / (1 - ||x||²).
    
    This is the key metric scaling factor in Poincaré geometry.
    """
    return 2.0 / (1.0 - x.pow(2).sum(dim=-1, keepdim=True).clamp(max=1.0 - EPS))


def poincare_distance(u: Tensor, v: Tensor) -> Tensor:
    """Compute hyperbolic distance in Poincaré ball.
    
    d(u,v) = arcosh(1 + 2||u-v||² / ((1-||u||²)(1-||v||²)))
    
    This is the core distance function that makes hyperbolic space natural
    for tree-structured data: distances grow exponentially near the boundary.
    
    Args:
        u: Points of shape (..., d)
        v: Points of shape (..., d)
        
    Returns:
        Distances of shape (...)
    """
    u = _clamp_norm(u)
    v = _clamp_norm(v)
    
    diff_norm_sq = (u - v).pow(2).sum(dim=-1)
    u_norm_sq = u.pow(2).sum(dim=-1)
    v_norm_sq = v.pow(2).sum(dim=-1)
    
    # Numerical stability: clamp denominator
    denom = (1.0 - u_norm_sq) * (1.0 - v_norm_sq)
    denom = denom.clamp(min=EPS)
    
    # arcosh(x) = log(x + sqrt(x² - 1))
    x = 1.0 + 2.0 * diff_norm_sq / denom
    x = x.clamp(min=1.0 + EPS)  # arcosh domain: x >= 1
    
    return torch.acosh(x)


def poincare_distance_matrix(X: Tensor, Y: Tensor | None = None) -> Tensor:
    """Compute pairwise hyperbolic distance matrix.
    
    Args:
        X: Points of shape (n, d)
        Y: Points of shape (m, d), defaults to X
        
    Returns:
        Distance matrix of shape (n, m)
    """
    if Y is None:
        Y = X
    
    # Expand for broadcasting: (n, 1, d) and (1, m, d)
    X_exp = X.unsqueeze(1)
    Y_exp = Y.unsqueeze(0)
    
    return poincare_distance(X_exp, Y_exp)


def mobius_add(u: Tensor, v: Tensor) -> Tensor:
    """Möbius addition in Poincaré ball.
    
    u ⊕ v = ((1 + 2⟨u,v⟩ + ||v||²)u + (1 - ||u||²)v) / (1 + 2⟨u,v⟩ + ||u||²||v||²)
    
    This is the hyperbolic analog of vector addition. Key property:
    d(0, u ⊕ v) = d(u, v) when one point is at origin.
    
    Args:
        u: Points of shape (..., d)
        v: Points of shape (..., d)
        
    Returns:
        Möbius sum of shape (..., d)
    """
    u = _clamp_norm(u)
    v = _clamp_norm(v)
    
    u_norm_sq = u.pow(2).sum(dim=-1, keepdim=True)
    v_norm_sq = v.pow(2).sum(dim=-1, keepdim=True)
    uv_dot = (u * v).sum(dim=-1, keepdim=True)
    
    numerator = (1.0 + 2.0 * uv_dot + v_norm_sq) * u + (1.0 - u_norm_sq) * v
    denominator = 1.0 + 2.0 * uv_dot + u_norm_sq * v_norm_sq
    denominator = denominator.clamp(min=EPS)
    
    result = numerator / denominator
    return _clamp_norm(result)


def mobius_scalar_mul(r: Tensor | float, x: Tensor) -> Tensor:
    """Möbius scalar multiplication.
    
    r ⊗ x = tanh(r × artanh(||x||)) × (x / ||x||)
    
    Scales a point along the geodesic from origin.
    
    Args:
        r: Scalar or tensor of scalars
        x: Points of shape (..., d)
        
    Returns:
        Scaled points of shape (..., d)
    """
    x = _clamp_norm(x)
    x_norm = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    
    # artanh(x) = 0.5 * log((1+x)/(1-x))
    artanh_norm = torch.atanh(x_norm.clamp(max=1.0 - EPS))
    
    new_norm = torch.tanh(r * artanh_norm)
    
    result = new_norm * (x / x_norm)
    return _clamp_norm(result)


def exp_map(v: Tensor, x: Tensor | None = None) -> Tensor:
    """Exponential map: tangent space → Poincaré ball.
    
    exp_x(v) = x ⊕ (tanh(λ_x × ||v|| / 2) × v / ||v||)
    
    Maps a tangent vector v at point x to a point on the manifold.
    Used for Riemannian gradient descent.
    
    Args:
        v: Tangent vectors of shape (..., d)
        x: Base points of shape (..., d), defaults to origin
        
    Returns:
        Points on manifold of shape (..., d)
    """
    if x is None:
        x = torch.zeros_like(v)
    x = _clamp_norm(x)
    
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=EPS)
    lambda_x = _lambda_x(x)
    
    # Direction normalized, scaled by tanh factor
    direction = v / v_norm
    scale = torch.tanh(lambda_x * v_norm / 2.0)
    
    projected = scale * direction
    
    return mobius_add(x, _clamp_norm(projected))


def log_map(y: Tensor, x: Tensor | None = None) -> Tensor:
    """Logarithmic map: Poincaré ball → tangent space.
    
    log_x(y) = (2 / λ_x) × artanh(||-x ⊕ y||) × (-x ⊕ y) / ||-x ⊕ y||
    
    Maps a point y to the tangent vector at x pointing toward y.
    Inverse of exp_map.
    
    Args:
        y: Target points of shape (..., d)
        x: Base points of shape (..., d), defaults to origin
        
    Returns:
        Tangent vectors of shape (..., d)
    """
    if x is None:
        x = torch.zeros_like(y)
    x = _clamp_norm(x)
    y = _clamp_norm(y)
    
    # -x ⊕ y using Möbius addition with negated x
    neg_x = -x
    diff = mobius_add(neg_x, y)
    
    diff_norm = diff.norm(dim=-1, keepdim=True).clamp(min=EPS, max=1.0 - EPS)
    lambda_x = _lambda_x(x)
    
    # artanh for scaling
    artanh_norm = torch.atanh(diff_norm)
    
    direction = diff / diff_norm
    scale = (2.0 / lambda_x) * artanh_norm
    
    return scale * direction


def project_to_ball(x: Tensor, max_norm: float = MAX_NORM) -> Tensor:
    """Project points to interior of Poincaré ball.
    
    Ensures ||x|| < max_norm for numerical stability.
    Points outside the ball are scaled to lie just inside.
    
    Args:
        x: Points of shape (..., d)
        max_norm: Maximum allowed norm (default 1-1e-5)
        
    Returns:
        Points with ||x|| < max_norm
    """
    return _clamp_norm(x, max_norm)


def geodesic(x: Tensor, y: Tensor, t: Tensor | float) -> Tensor:
    """Point on geodesic from x to y at parameter t.
    
    γ(t) = x ⊕ (t ⊗ (-x ⊕ y))
    
    Args:
        x: Start points of shape (..., d)
        y: End points of shape (..., d)
        t: Parameter in [0, 1], can be tensor for batch
        
    Returns:
        Points along geodesic of shape (..., d)
    """
    x = _clamp_norm(x)
    y = _clamp_norm(y)
    
    # Direction from x to y in Möbius sense
    neg_x = -x
    direction = mobius_add(neg_x, y)
    
    # Scale by t
    scaled = mobius_scalar_mul(t, direction)
    
    return mobius_add(x, scaled)


def hyperbolic_midpoint(x: Tensor, y: Tensor) -> Tensor:
    """Compute hyperbolic midpoint of two points.
    
    Args:
        x: Points of shape (..., d)
        y: Points of shape (..., d)
        
    Returns:
        Midpoints of shape (..., d)
    """
    return geodesic(x, y, 0.5)


def hyperbolic_centroid(points: Tensor, weights: Tensor | None = None) -> Tensor:
    """Compute weighted hyperbolic centroid (Fréchet mean).
    
    Uses iterative algorithm since closed-form doesn't exist.
    
    Args:
        points: Points of shape (n, d)
        weights: Optional weights of shape (n,), defaults to uniform
        
    Returns:
        Centroid of shape (d,)
    """
    if weights is None:
        weights = torch.ones(points.shape[0], device=points.device)
    weights = weights / weights.sum()
    
    # Initialize at Euclidean weighted mean (projected)
    centroid = _clamp_norm((weights.unsqueeze(-1) * points).sum(dim=0))
    
    # Gradient descent in hyperbolic space
    for _ in range(100):
        # Gradient: sum of log maps from centroid to points
        logs = log_map(points, centroid.unsqueeze(0))
        grad = (weights.unsqueeze(-1) * logs).sum(dim=0)
        
        if grad.norm() < 1e-8:
            break
            
        # Step along gradient (learning rate 0.5)
        centroid = exp_map(0.5 * grad, centroid)
    
    return centroid


def adaptive_dimension(max_degree: int) -> int:
    """Compute optimal Poincaré ball dimension for given tree degree.
    
    Uses formula: dim = min(8, ceil(log2(max_degree)))
    
    Higher degree trees need more dimensions to avoid distortion.
    
    Args:
        max_degree: Maximum branching factor in tree
        
    Returns:
        Recommended embedding dimension
    """
    import math
    return min(8, max(2, math.ceil(math.log2(max(2, max_degree)))))
