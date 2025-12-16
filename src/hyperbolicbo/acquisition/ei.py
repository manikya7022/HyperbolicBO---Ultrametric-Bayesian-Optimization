"""Hyperbolic Expected Improvement acquisition function.

EI(x) = E[max(f(x) - f(x*), 0)]

where f(x*) is the current best observation.

In hyperbolic space, we use O(1) distance lookups via horospheres
and the hyperbolic kernel for covariance computation.
"""

from __future__ import annotations

import torch
from torch import Tensor
from scipy.stats import norm as scipy_norm
import numpy as np

from ..gp.hyperbolic_gp import HyperbolicGP
from ..geometry.poincare import project_to_ball


def hyperbolic_ei(
    x: Tensor,
    gp: HyperbolicGP,
    y_best: float | Tensor,
    xi: float = 0.01,
) -> Tensor:
    """Compute Expected Improvement at points x.
    
    EI(x) = (μ - y_best - ξ) Φ(Z) + σ φ(Z)
    
    where Z = (μ - y_best - ξ) / σ
          Φ = standard normal CDF
          φ = standard normal PDF
    
    Uses hyperbolic GP for μ and σ computation.
    
    Args:
        x: Query points, shape (n, d) or (d,)
        gp: Fitted HyperbolicGP
        y_best: Best observed value so far
        xi: Exploration-exploitation trade-off (higher = more exploration)
        
    Returns:
        EI values, shape (n,) or scalar
    """
    single = x.dim() == 1
    if single:
        x = x.unsqueeze(0)
    
    x = project_to_ball(x)
    
    # Get GP predictions (O(1) with horosphere approximation)
    if gp.use_horosphere and gp.n_observations > 50:
        mu, sigma = gp.predict_approximate(x)
    else:
        mu, sigma = gp.predict(x)
    
    if sigma is None:
        sigma = torch.ones_like(mu) * 1e-6
    
    # Avoid division by zero
    sigma = sigma.clamp(min=1e-8)
    
    y_best = torch.as_tensor(y_best, device=x.device, dtype=x.dtype)
    
    # Improvement
    improvement = mu - y_best - xi
    
    # Z-score
    Z = improvement / sigma
    
    # EI formula using PyTorch
    # Φ(z) and φ(z)
    Z_np = Z.detach().cpu().numpy()
    cdf = torch.tensor(scipy_norm.cdf(Z_np), device=x.device, dtype=x.dtype)
    pdf = torch.tensor(scipy_norm.pdf(Z_np), device=x.device, dtype=x.dtype)
    
    ei = improvement * cdf + sigma * pdf
    
    # EI is always non-negative
    ei = ei.clamp(min=0.0)
    
    if single:
        return ei.squeeze(0)
    return ei


def hyperbolic_ei_grad(
    x: Tensor,
    gp: HyperbolicGP,
    y_best: float,
    xi: float = 0.01,
) -> Tuple[Tensor, Tensor]:
    """Compute EI and its gradient (for Riemannian optimization).
    
    Args:
        x: Single query point, shape (d,)
        gp: Fitted HyperbolicGP
        y_best: Best observed value
        xi: Exploration parameter
        
    Returns:
        Tuple of (ei_value, gradient) both shape (d,) or scalar
    """
    x = x.clone().requires_grad_(True)
    ei = hyperbolic_ei(x, gp, y_best, xi)
    ei.backward()
    
    return ei.detach(), x.grad.detach()


def batch_hyperbolic_ei(
    candidates: Tensor,
    gp: HyperbolicGP,
    y_best: float,
    batch_size: int,
    xi: float = 0.01,
) -> Tensor:
    """Select top batch_size points by EI.
    
    Simple greedy selection (not q-EI, but fast).
    
    Args:
        candidates: Candidate points, shape (n, d)
        gp: Fitted HyperbolicGP
        y_best: Best observed value
        batch_size: Number of points to select
        xi: Exploration parameter
        
    Returns:
        Indices of selected points, shape (batch_size,)
    """
    ei_values = hyperbolic_ei(candidates, gp, y_best, xi)
    
    # Select top-k
    _, indices = ei_values.topk(min(batch_size, len(candidates)))
    
    return indices


def optimize_hyperbolic_ei(
    gp: HyperbolicGP,
    y_best: float,
    n_restarts: int = 10,
    n_iters: int = 50,
    xi: float = 0.01,
) -> Tensor:
    """Find point that maximizes EI via Riemannian gradient ascent.
    
    Uses Geoopt's Riemannian Adam optimizer on the Poincaré ball.
    
    Args:
        gp: Fitted HyperbolicGP
        y_best: Best observed value
        n_restarts: Number of random restarts
        n_iters: Iterations per restart
        xi: Exploration parameter
        
    Returns:
        Best point found, shape (d,)
    """
    import geoopt
    
    device = gp.device if hasattr(gp, 'device') else 'cpu'
    dim = gp.dim
    
    best_ei = float('-inf')
    best_x = None
    
    manifold = geoopt.PoincareBall()
    
    for _ in range(n_restarts):
        # Random initialization inside ball
        x = torch.randn(dim, device=device) * 0.3
        x = project_to_ball(x)
        x = geoopt.ManifoldParameter(x, manifold=manifold)
        
        optimizer = geoopt.optim.RiemannianAdam([x], lr=0.1)
        
        for _ in range(n_iters):
            optimizer.zero_grad()
            
            # Negative EI for minimization
            ei = -hyperbolic_ei(x, gp, y_best, xi)
            ei.backward()
            
            optimizer.step()
        
        final_ei = -hyperbolic_ei(x.detach(), gp, y_best, xi).item()
        
        if final_ei > best_ei:
            best_ei = final_ei
            best_x = x.detach().clone()
    
    return best_x
