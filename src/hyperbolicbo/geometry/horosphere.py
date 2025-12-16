"""Horosphere clustering for O(1) nearest neighbor lookup.

Horospheres are hyperbolic analogs of hyperplanes. In the Poincaré ball,
they appear as spheres tangent to the boundary. Key property:

    All points on a horosphere are equidistant from the boundary point.

This enables O(1) nearest neighbor queries by pre-clustering points
based on their "height" (distance from a reference boundary direction).

Algorithm:
1. Choose k reference directions (boundary points)
2. For each point, compute Busemann function b(x) = lim[t→∞] d(x,γ(t)) - t
3. Cluster by Busemann values (discretized into buckets)
4. Query: compute Busemann for query, lookup bucket, linear search within
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
import numpy as np

from .poincare import poincare_distance, _clamp_norm, EPS


def busemann_function(x: Tensor, direction: Tensor) -> Tensor:
    """Compute Busemann function for points x toward boundary direction.
    
    b_ξ(x) = log((1 - ||x||²) / ||ξ - x||²)
    
    where ξ is a unit vector (point on boundary).
    
    The Busemann function gives the "height" of a point relative to
    a horosphere defined by direction ξ. Points with same b value
    lie on the same horosphere.
    
    Args:
        x: Points of shape (..., d)
        direction: Unit vector(s) of shape (..., d) or (d,)
        
    Returns:
        Busemann values of shape (...)
    """
    x = _clamp_norm(x)
    
    # Ensure direction is on boundary (unit norm)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp(min=EPS)
    
    x_norm_sq = x.pow(2).sum(dim=-1)
    diff_norm_sq = (direction - x).pow(2).sum(dim=-1)
    
    # Avoid log of zero
    numerator = (1.0 - x_norm_sq).clamp(min=EPS)
    denominator = diff_norm_sq.clamp(min=EPS)
    
    return torch.log(numerator / denominator)


@dataclass
class HorosphereCluster:
    """A cluster defined by a horosphere level."""
    level: float  # Busemann value range center
    indices: list[int]  # Point indices in this cluster
    points: Tensor  # Actual points for linear search


@dataclass
class HorosphereIndex:
    """Index structure for O(1) approximate nearest neighbor via horospheres.
    
    Pre-computes Busemann values and clusters points by horosphere level.
    Query time is O(bucket_size) ≈ O(1) for well-distributed points.
    """
    
    points: Tensor  # All indexed points (n, d)
    directions: Tensor  # Reference boundary directions (k, d)
    busemann_values: Tensor  # Pre-computed (n, k)
    buckets: dict[tuple[int, ...], list[int]]  # Discretized Busemann → indices
    n_buckets: int  # Discretization granularity
    
    @classmethod
    def build(
        cls,
        points: Tensor,
        n_directions: int = 4,
        n_buckets: int = 10,
    ) -> "HorosphereIndex":
        """Build horosphere index from points.
        
        Args:
            points: Points to index of shape (n, d)
            n_directions: Number of reference boundary directions
            n_buckets: Discretization granularity per direction
            
        Returns:
            HorosphereIndex ready for queries
        """
        n, d = points.shape
        points = _clamp_norm(points)
        
        # Choose evenly-spaced directions on boundary (unit sphere)
        # Using Fibonacci lattice for good coverage
        directions = _fibonacci_sphere(n_directions, d, points.device)
        
        # Compute Busemann for all points and directions
        # Shape: (n, k) where n = #points, k = #directions
        busemann_values = torch.stack([
            busemann_function(points, directions[i])
            for i in range(n_directions)
        ], dim=1)
        
        # Normalize to [0, 1] range for bucketing
        bmin = busemann_values.min(dim=0).values
        bmax = busemann_values.max(dim=0).values
        brange = (bmax - bmin).clamp(min=EPS)
        normalized = (busemann_values - bmin) / brange
        
        # Discretize into bucket indices
        bucket_indices = (normalized * (n_buckets - 1)).long().clamp(0, n_buckets - 1)
        
        # Build bucket dictionary
        buckets: dict[tuple[int, ...], list[int]] = {}
        for i in range(n):
            key = tuple(bucket_indices[i].tolist())
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(i)
        
        return cls(
            points=points,
            directions=directions,
            busemann_values=busemann_values,
            buckets=buckets,
            n_buckets=n_buckets,
        )
    
    def query(
        self,
        query_points: Tensor,
        k: int = 1,
        search_radius: int = 1,
    ) -> tuple[Tensor, Tensor]:
        """Find k nearest neighbors for query points.
        
        Args:
            query_points: Query points of shape (m, d) or (d,)
            k: Number of neighbors to return
            search_radius: How many adjacent buckets to search
            
        Returns:
            Tuple of (indices, distances) each of shape (m, k) or (k,)
        """
        single = query_points.dim() == 1
        if single:
            query_points = query_points.unsqueeze(0)
        
        m = query_points.shape[0]
        query_points = _clamp_norm(query_points)
        
        # Compute Busemann for queries
        query_busemann = torch.stack([
            busemann_function(query_points, self.directions[i])
            for i in range(len(self.directions))
        ], dim=1)
        
        # Normalize using same scale as index
        bmin = self.busemann_values.min(dim=0).values
        bmax = self.busemann_values.max(dim=0).values
        brange = (bmax - bmin).clamp(min=EPS)
        normalized = (query_busemann - bmin) / brange
        
        bucket_indices = (normalized * (self.n_buckets - 1)).long().clamp(0, self.n_buckets - 1)
        
        all_indices = []
        all_distances = []
        
        for i in range(m):
            base_key = bucket_indices[i].tolist()
            
            # Gather candidates from nearby buckets
            candidate_indices = []
            for offset in _nearby_buckets(len(self.directions), search_radius):
                key = tuple(
                    max(0, min(self.n_buckets - 1, base_key[j] + offset[j]))
                    for j in range(len(self.directions))
                )
                candidate_indices.extend(self.buckets.get(key, []))
            
            if not candidate_indices:
                # Fallback: search all points
                candidate_indices = list(range(len(self.points)))
            
            # Remove duplicates, compute distances
            candidate_indices = list(set(candidate_indices))
            candidates = self.points[candidate_indices]
            
            dists = poincare_distance(
                query_points[i:i+1].expand(len(candidates), -1),
                candidates
            )
            
            # Get top-k
            topk = min(k, len(candidate_indices))
            top_dists, top_local = dists.topk(topk, largest=False)
            top_indices = torch.tensor(
                [candidate_indices[j] for j in top_local.tolist()],
                device=query_points.device
            )
            
            # Pad if needed
            if topk < k:
                pad_size = k - topk
                top_indices = torch.cat([
                    top_indices,
                    torch.zeros(pad_size, dtype=torch.long, device=query_points.device)
                ])
                top_dists = torch.cat([
                    top_dists,
                    torch.full((pad_size,), float('inf'), device=query_points.device)
                ])
            
            all_indices.append(top_indices)
            all_distances.append(top_dists)
        
        indices = torch.stack(all_indices)
        distances = torch.stack(all_distances)
        
        if single:
            return indices.squeeze(0), distances.squeeze(0)
        return indices, distances
    
    def update(self, new_points: Tensor) -> "HorosphereIndex":
        """Add new points to index.
        
        For efficiency, rebuilds index with all points.
        For truly incremental updates, would need LSH-style approach.
        
        Args:
            new_points: New points of shape (m, d)
            
        Returns:
            New HorosphereIndex with all points
        """
        all_points = torch.cat([self.points, new_points], dim=0)
        return HorosphereIndex.build(
            all_points,
            n_directions=len(self.directions),
            n_buckets=self.n_buckets,
        )


def _fibonacci_sphere(n: int, dim: int, device: torch.device) -> Tensor:
    """Generate n evenly-spaced points on unit sphere using Fibonacci lattice.
    
    Works for any dimension, but optimized for 2D and 3D.
    """
    if dim == 2:
        # 2D: just evenly space on circle
        angles = torch.linspace(0, 2 * np.pi, n + 1, device=device)[:-1]
        return torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    
    # Higher dimensions: use Fibonacci-like spiral
    points = torch.zeros(n, dim, device=device)
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    for i in range(n):
        # Generate point on hypersphere
        coords = []
        for d in range(dim - 1):
            theta = 2 * np.pi * ((i * phi) % 1) * (d + 1) / dim
            coords.append(np.cos(theta) * np.sin(np.pi * (i + 0.5) / n))
        coords.append(np.cos(np.pi * (i + 0.5) / n))
        
        point = torch.tensor(coords, device=device)
        points[i] = point / point.norm().clamp(min=EPS)
    
    return points


def _nearby_buckets(n_dims: int, radius: int) -> list[tuple[int, ...]]:
    """Generate all offset tuples within L∞ radius.
    
    For radius=1 and n_dims=2: [(-1,-1), (-1,0), ..., (1,1)] = 9 offsets
    """
    from itertools import product
    offsets = list(range(-radius, radius + 1))
    return list(product(offsets, repeat=n_dims))
