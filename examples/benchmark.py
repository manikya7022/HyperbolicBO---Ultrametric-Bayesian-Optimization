"""Benchmark: HyperbolicBO vs Standard BO vs Random Search.

Compares optimization methods on synthetic tree-structured objective.
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict

import sys
sys.path.insert(0, "src")

from hyperbolicbo import HyperbolicBO
from hyperbolicbo.geometry.poincare import poincare_distance


# =============================================================================
# Synthetic Tree-Structured Objective
# =============================================================================

class TreeStructuredObjective:
    """Synthetic objective with tree structure.
    
    The objective has multiple local optima arranged in a tree.
    Hyperbolic methods should find the global optimum faster
    because similar pipelines (in tree sense) have similar scores.
    """
    
    def __init__(self, n_branches: int = 3, depth: int = 4, noise: float = 0.05):
        self.n_branches = n_branches
        self.depth = depth
        self.noise = noise
        
        # Create tree of optima (each leaf is a potential optimum)
        self.n_leaves = n_branches ** depth
        np.random.seed(42)
        
        # One global optimum, rest are local
        self.global_optimum_leaf = np.random.randint(self.n_leaves)
        self.leaf_values = np.random.uniform(0.3, 0.7, self.n_leaves)
        self.leaf_values[self.global_optimum_leaf] = 1.0
        
    def pipeline_to_leaf(self, pipeline: Dict) -> int:
        """Map pipeline to leaf index based on structure."""
        # Hash pipeline to get deterministic leaf assignment
        stages = pipeline.get("stages", [])
        
        # Use stage types to determine path in tree
        hash_val = 0
        for i, stage in enumerate(stages[:self.depth]):
            stage_type = stage.get("type", "")
            type_hash = hash(stage_type) % self.n_branches
            hash_val = hash_val * self.n_branches + type_hash
        
        return hash_val % self.n_leaves
    
    def __call__(self, pipeline: Dict) -> float:
        """Evaluate pipeline."""
        leaf = self.pipeline_to_leaf(pipeline)
        
        # Base value from tree position
        base_value = self.leaf_values[leaf]
        
        # Add some structure: deeper pipelines slightly penalized
        n_stages = len(pipeline.get("stages", []))
        depth_penalty = 0.02 * max(0, n_stages - 3)
        
        # Noise
        noise = np.random.normal(0, self.noise)
        
        return max(0, min(1, base_value - depth_penalty + noise))


# =============================================================================
# Random Search Baseline
# =============================================================================

class RandomSearch:
    """Random search baseline."""
    
    def __init__(self, pipeline_type: str = "automl"):
        self.pipeline_type = pipeline_type
        self.history = []
        
    def suggest(self, n_suggestions: int) -> List[Dict]:
        """Generate random pipelines."""
        pipelines = []
        for _ in range(n_suggestions):
            stages = []
            # Random number of stages
            n_stages = np.random.randint(1, 5)
            
            if np.random.random() > 0.3:
                stages.append({
                    "type": np.random.choice(["scaler", "normalizer"]),
                    "method": np.random.choice(["StandardScaler", "MinMaxScaler"])
                })
            
            if np.random.random() > 0.5:
                stages.append({
                    "type": "pca",
                    "n_components": int(np.random.choice([10, 20, 50]))
                })
            
            stages.append({
                "type": np.random.choice(["xgb", "rf", "lr", "svm"]),
            })
            
            pipelines.append({"stages": stages})
        
        return pipelines
    
    def observe(self, pipelines: List[Dict], scores: List[float]):
        """Record observations."""
        for p, s in zip(pipelines, scores):
            self.history.append({"pipeline": p, "score": s})
    
    def best(self) -> Tuple[Dict, float]:
        """Get best observed."""
        if not self.history:
            return {}, 0.0
        best_entry = max(self.history, key=lambda x: x["score"])
        return best_entry["pipeline"], best_entry["score"]


# =============================================================================
# Simple Euclidean BO (using sklearn-style embedding)
# =============================================================================

class EuclideanBO:
    """Simple Euclidean BO using feature hashing."""
    
    def __init__(self, dim: int = 10):
        self.dim = dim
        self.history = []
        self.X = []
        self.y = []
        
    def _embed(self, pipeline: Dict) -> np.ndarray:
        """Simple feature hashing."""
        stages = pipeline.get("stages", [])
        features = np.zeros(self.dim)
        
        for i, stage in enumerate(stages):
            stage_type = stage.get("type", "")
            # Hash to feature index
            idx = hash(stage_type) % self.dim
            features[idx] += 1.0 / (i + 1)
        
        return features / (np.linalg.norm(features) + 1e-8)
    
    def suggest(self, n_suggestions: int) -> List[Dict]:
        """Suggest using simple UCB."""
        # Generate candidates
        candidates = []
        for _ in range(n_suggestions * 10):
            stages = []
            if np.random.random() > 0.3:
                stages.append({"type": np.random.choice(["scaler", "normalizer"])})
            if np.random.random() > 0.5:
                stages.append({"type": "pca", "n_components": np.random.choice([10, 20, 50])})
            stages.append({"type": np.random.choice(["xgb", "rf", "lr", "svm"])})
            candidates.append({"stages": stages})
        
        if len(self.X) < 3:
            # Random selection initially
            return candidates[:n_suggestions]
        
        # Simple UCB: mean + exploration bonus
        X_train = np.array(self.X)
        y_train = np.array(self.y)
        
        scores = []
        for c in candidates:
            x = self._embed(c)
            
            # Distance to training points
            dists = np.linalg.norm(X_train - x, axis=1)
            
            # Nearest neighbor prediction
            nearest_idx = dists.argmin()
            mean = y_train[nearest_idx]
            
            # Exploration bonus (higher for points far from training)
            exploration = 0.1 * dists.min()
            
            scores.append(mean + exploration)
        
        # Select top
        top_idx = np.argsort(scores)[-n_suggestions:]
        return [candidates[i] for i in top_idx]
    
    def observe(self, pipelines: List[Dict], scores: List[float]):
        """Record observations."""
        for p, s in zip(pipelines, scores):
            self.history.append({"pipeline": p, "score": s})
            self.X.append(self._embed(p))
            self.y.append(s)
    
    def best(self) -> Tuple[Dict, float]:
        """Get best observed."""
        if not self.history:
            return {}, 0.0
        best_entry = max(self.history, key=lambda x: x["score"])
        return best_entry["pipeline"], best_entry["score"]


# =============================================================================
# Benchmark Runner
# =============================================================================

@dataclass
class BenchmarkResult:
    method: str
    best_scores: List[float]  # Best score at each iteration
    times: List[float]  # Cumulative time
    final_best: float
    total_time: float


def run_benchmark(
    objective,
    n_iterations: int = 20,
    batch_size: int = 4,
    n_runs: int = 3,
) -> Dict[str, BenchmarkResult]:
    """Run benchmark comparing methods."""
    
    results = {}
    
    methods = {
        "HyperbolicBO": lambda: HyperbolicBO(dim=4, acquisition="thompson", n_parallel=batch_size),
        "EuclideanBO": lambda: EuclideanBO(dim=10),
        "RandomSearch": lambda: RandomSearch(),
    }
    
    for method_name, create_optimizer in methods.items():
        print(f"\n{'='*50}")
        print(f"Running: {method_name}")
        print(f"{'='*50}")
        
        all_best_scores = []
        all_times = []
        
        for run in range(n_runs):
            print(f"  Run {run + 1}/{n_runs}...", end=" ")
            
            np.random.seed(run * 100)
            torch.manual_seed(run * 100)
            
            optimizer = create_optimizer()
            best_scores = []
            times = []
            current_best = 0.0
            
            start_time = time.time()
            
            for i in range(n_iterations):
                # Suggest
                suggestions = optimizer.suggest(n_suggestions=batch_size)
                
                # Evaluate
                scores = [objective(p) for p in suggestions]
                
                # Observe
                optimizer.observe(suggestions, scores)
                
                # Track best
                current_best = max(current_best, max(scores))
                best_scores.append(current_best)
                times.append(time.time() - start_time)
            
            _, final = optimizer.best()
            print(f"Final best: {final:.4f}")
            
            all_best_scores.append(best_scores)
            all_times.append(times)
        
        # Average across runs
        avg_best_scores = np.mean(all_best_scores, axis=0).tolist()
        avg_times = np.mean(all_times, axis=0).tolist()
        
        results[method_name] = BenchmarkResult(
            method=method_name,
            best_scores=avg_best_scores,
            times=avg_times,
            final_best=avg_best_scores[-1],
            total_time=avg_times[-1],
        )
    
    return results


def plot_results(results: Dict[str, BenchmarkResult], save_path: str = "benchmark_results.png"):
    """Plot benchmark comparison."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {
        "HyperbolicBO": "#e63946",
        "EuclideanBO": "#457b9d", 
        "RandomSearch": "#2a9d8f",
    }
    
    # Plot 1: Best score vs iterations
    ax1 = axes[0]
    for method_name, result in results.items():
        iterations = list(range(1, len(result.best_scores) + 1))
        ax1.plot(
            iterations, 
            result.best_scores, 
            label=f"{method_name} (final: {result.final_best:.3f})",
            color=colors.get(method_name, "gray"),
            linewidth=2,
            marker='o',
            markersize=4,
        )
    
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Best Score Found", fontsize=12)
    ax1.set_title("Optimization Progress", fontsize=14)
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Plot 2: Best score vs time
    ax2 = axes[1]
    for method_name, result in results.items():
        ax2.plot(
            result.times,
            result.best_scores,
            label=f"{method_name} ({result.total_time:.1f}s)",
            color=colors.get(method_name, "gray"),
            linewidth=2,
            marker='o',
            markersize=4,
        )
    
    ax2.set_xlabel("Time (seconds)", fontsize=12)
    ax2.set_ylabel("Best Score Found", fontsize=12)
    ax2.set_title("Optimization Efficiency", fontsize=14)
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {save_path}")
    plt.show()


def print_summary(results: Dict[str, BenchmarkResult]):
    """Print summary table."""
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"{'Method':<20} {'Final Best':<15} {'Time (s)':<15} {'Speedup':<15}")
    print("-"*70)
    
    random_best = results["RandomSearch"].final_best
    random_time = results["RandomSearch"].total_time
    
    for method_name, result in results.items():
        speedup = "-"
        if method_name != "RandomSearch" and result.final_best > random_best:
            # How many iterations did random need to reach this score?
            random_scores = results["RandomSearch"].best_scores
            for i, s in enumerate(random_scores):
                if s >= result.final_best * 0.95:
                    speedup = f"{(i+1) / len(result.best_scores):.1f}x"
                    break
            else:
                speedup = ">N/A"
        
        print(f"{method_name:<20} {result.final_best:<15.4f} {result.total_time:<15.2f} {speedup:<15}")
    
    print("="*70)


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*70)
    print("HyperbolicBO BENCHMARK COMPARISON")
    print("="*70)
    print("""
    Comparing:
    - HyperbolicBO (Poincaré ball geometry + Thompson Sampling)
    - EuclideanBO (Feature hashing + UCB)
    - RandomSearch (Baseline)
    
    Objective: Synthetic tree-structured function
    """)
    
    # Create objective
    objective = TreeStructuredObjective(n_branches=3, depth=4, noise=0.05)
    print(f"Objective: {objective.n_leaves} leaves, global optimum at leaf {objective.global_optimum_leaf}")
    
    # Run benchmark
    results = run_benchmark(
        objective,
        n_iterations=15,
        batch_size=4,
        n_runs=3,
    )
    
    # Print summary
    print_summary(results)
    
    # Plot
    plot_results(results, "benchmark_results.png")
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
