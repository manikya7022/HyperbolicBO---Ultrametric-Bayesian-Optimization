"""Example: Visualize Poincaré disk embeddings.

This example shows how to visualize pipeline embeddings
on the Poincaré disk.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.insert(0, "../src")

from hyperbolicbo import HyperbolicBO
from hyperbolicbo.geometry.poincare import poincare_distance


def plot_poincare_disk(ax, title="Poincaré Disk"):
    """Draw Poincaré disk boundary."""
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def main():
    """Visualize pipeline embeddings on Poincaré disk."""
    print("HyperbolicBO Visualization Example")
    print("=" * 40)
    
    # Initialize optimizer with 2D for visualization
    optimizer = HyperbolicBO(
        dim=2,
        acquisition="thompson",
        pipeline_type="automl",
    )
    
    # Generate some random pipelines and scores
    np.random.seed(42)
    n_pipelines = 30
    
    for i in range(n_pipelines):
        # Generate random pipeline
        suggestions = optimizer.suggest(n_suggestions=1)
        
        # Fake score (based on pipeline complexity)
        pipeline = suggestions[0]
        n_stages = len(pipeline.get("stages", []))
        score = 0.5 + 0.4 * np.random.random() - 0.1 * n_stages / 5
        
        optimizer.observe(suggestions, [score])
    
    print(f"Generated {optimizer.n_observations} pipelines")
    
    # Extract embeddings and scores
    embeddings = optimizer._embeddings.numpy()
    scores = optimizer._scores.numpy()
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Embeddings colored by score
    ax1 = axes[0]
    plot_poincare_disk(ax1, "Pipeline Embeddings (colored by score)")
    
    scatter = ax1.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=scores,
        cmap='viridis',
        s=100,
        edgecolors='white',
        linewidth=1,
        zorder=10,
    )
    plt.colorbar(scatter, ax=ax1, label='Score')
    
    # Mark best point
    best_idx = scores.argmax()
    ax1.scatter(
        embeddings[best_idx, 0],
        embeddings[best_idx, 1],
        c='red',
        s=300,
        marker='*',
        edgecolors='black',
        linewidth=2,
        zorder=20,
        label=f'Best (score={scores[best_idx]:.3f})',
    )
    ax1.legend()
    
    # Plot 2: Hyperbolic distance heatmap from best point
    ax2 = axes[1]
    plot_poincare_disk(ax2, "Hyperbolic Distance from Best")
    
    # Create grid
    xx, yy = np.meshgrid(
        np.linspace(-0.95, 0.95, 50),
        np.linspace(-0.95, 0.95, 50),
    )
    
    grid_points = torch.tensor(
        np.stack([xx.ravel(), yy.ravel()], axis=1),
        dtype=torch.float32
    )
    
    # Filter to inside disk
    mask = (grid_points.norm(dim=1) < 0.98).numpy()
    
    # Compute distances from best
    best_emb = torch.tensor(embeddings[best_idx], dtype=torch.float32)
    distances = poincare_distance(
        grid_points,
        best_emb.unsqueeze(0).expand(len(grid_points), -1)
    ).numpy()
    
    # Plot heatmap
    distances_grid = np.full(xx.shape, np.nan)
    distances_grid.ravel()[mask] = distances[mask]
    
    contour = ax2.contourf(
        xx, yy, distances_grid,
        levels=20,
        cmap='plasma',
        alpha=0.7,
    )
    plt.colorbar(contour, ax=ax2, label='Hyperbolic Distance')
    
    # Add observation points
    ax2.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c='white',
        s=50,
        edgecolors='black',
        alpha=0.8,
        zorder=10,
    )
    ax2.scatter(
        embeddings[best_idx, 0],
        embeddings[best_idx, 1],
        c='red',
        s=200,
        marker='*',
        edgecolors='white',
        linewidth=2,
        zorder=20,
    )
    
    plt.tight_layout()
    plt.savefig('poincare_visualization.png', dpi=150)
    print("Saved: poincare_visualization.png")
    plt.show()


if __name__ == "__main__":
    main()
