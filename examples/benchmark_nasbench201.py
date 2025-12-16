"""NAS-Bench-201 Benchmark: HyperbolicBO vs Baselines.

Compares optimization methods on the NAS-Bench-201 neural architecture
search benchmark, demonstrating HyperbolicBO's advantages on true
tree-structured architecture spaces.
"""

import time
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hyperbolicbo import HyperbolicBO
from hyperbolicbo.benchmarks import NASBench201, NASBench201Objective, OPERATIONS, NUM_EDGES


# =============================================================================
# Baseline Methods
# =============================================================================

class RandomSearchNAS:
    """Random search baseline for NAS."""
    
    def __init__(self):
        self.history = []
        
    def suggest(self, n_suggestions: int) -> List[Dict]:
        archs = []
        for _ in range(n_suggestions):
            ops = [np.random.choice(OPERATIONS) for _ in range(NUM_EDGES)]
            cells = [{"op": op, "input": [i]} for i, op in enumerate(ops)]
            archs.append({"cells": cells})
        return archs
    
    def observe(self, archs: List[Dict], scores: List[float]):
        for a, s in zip(archs, scores):
            self.history.append({"arch": a, "score": s})
    
    def best(self):
        if not self.history:
            return {}, 0.0
        best_entry = max(self.history, key=lambda x: x["score"])
        return best_entry["arch"], best_entry["score"]


class RegularizedEvolution:
    """Regularized Evolution (AgingEvolution) for NAS.
    
    Based on Real et al., "Regularized Evolution for Image Classifier 
    Architecture Search", AAAI 2019.
    """
    
    def __init__(self, population_size: int = 20, sample_size: int = 5):
        self.population_size = population_size
        self.sample_size = sample_size
        self.population = []  # List of (arch, score)
        
    def suggest(self, n_suggestions: int) -> List[Dict]:
        archs = []
        
        for _ in range(n_suggestions):
            if len(self.population) < self.population_size:
                # Random init
                ops = [np.random.choice(OPERATIONS) for _ in range(NUM_EDGES)]
            else:
                # Tournament selection + mutation
                sample = np.random.choice(len(self.population), self.sample_size, replace=False)
                parent_idx = max(sample, key=lambda i: self.population[i][1])
                parent = self.population[parent_idx][0]
                
                # Mutate: change one random operation
                ops = [c["op"] for c in parent["cells"]]
                mutate_idx = np.random.randint(NUM_EDGES)
                ops[mutate_idx] = np.random.choice(OPERATIONS)
            
            cells = [{"op": op, "input": [i]} for i, op in enumerate(ops)]
            archs.append({"cells": cells})
        
        return archs
    
    def observe(self, archs: List[Dict], scores: List[float]):
        for a, s in zip(archs, scores):
            self.population.append((a, s))
            # Age-based removal
            if len(self.population) > self.population_size * 2:
                self.population.pop(0)  # Remove oldest
    
    def best(self):
        if not self.population:
            return {}, 0.0
        best_entry = max(self.population, key=lambda x: x[1])
        return best_entry[0], best_entry[1]


# =============================================================================
# Benchmark Runner
# =============================================================================

@dataclass
class BenchmarkResult:
    method: str
    best_scores: List[float]  # Best score at each iteration
    query_counts: List[int]   # Queries at each iteration
    final_best: float
    final_queries: int
    total_time: float
    

def run_nas_benchmark(
    n_iterations: int = 30,
    batch_size: int = 4,
    n_runs: int = 5,
    save_results: bool = True,
) -> Dict[str, BenchmarkResult]:
    """Run NAS-Bench-201 benchmark comparison."""
    
    print("=" * 70)
    print("NAS-BENCH-201 BENCHMARK")
    print("=" * 70)
    print(f"Iterations: {n_iterations}, Batch size: {batch_size}, Runs: {n_runs}")
    print()
    
    # Initialize benchmark
    benchmark = NASBench201(use_synthetic=True)  # Use synthetic for demo
    print(f"Search space: {len(benchmark):,} architectures")
    
    # Get optimal for reference
    optimal = benchmark.get_optimal_architecture()
    optimal_score = benchmark.query(optimal).valid_acc
    print(f"Optimal validation accuracy: {optimal_score:.4f}")
    print()
    
    results = {}
    
    methods = {
        "HyperbolicBO": lambda: HyperbolicBO(
            dim=4, 
            acquisition="thompson", 
            pipeline_type="nas",
            n_parallel=batch_size,
        ),
        "RegularizedEvolution": lambda: RegularizedEvolution(
            population_size=20, 
            sample_size=5,
        ),
        "RandomSearch": lambda: RandomSearchNAS(),
    }
    
    for method_name, create_optimizer in methods.items():
        print(f"{'='*50}")
        print(f"Running: {method_name}")
        print(f"{'='*50}")
        
        all_best_scores = []
        all_query_counts = []
        all_times = []
        
        for run in range(n_runs):
            print(f"  Run {run + 1}/{n_runs}...", end=" ", flush=True)
            
            np.random.seed(run * 100)
            torch.manual_seed(run * 100)
            
            optimizer = create_optimizer()
            objective = NASBench201Objective(benchmark)
            
            best_scores = []
            query_counts = []
            current_best = 0.0
            
            start_time = time.time()
            
            for i in range(n_iterations):
                # Suggest architectures
                archs = optimizer.suggest(n_suggestions=batch_size)
                
                # Evaluate
                scores = [objective(a) for a in archs]
                
                # Observe
                optimizer.observe(archs, scores)
                
                # Track best
                current_best = max(current_best, max(scores))
                best_scores.append(current_best)
                query_counts.append(objective.query_count)
            
            _, final = optimizer.best()
            elapsed = time.time() - start_time
            
            # Compute regret (gap to optimal)
            regret = optimal_score - final
            print(f"Best: {final:.4f} (regret: {regret:.4f}), Time: {elapsed:.1f}s")
            
            all_best_scores.append(best_scores)
            all_query_counts.append(query_counts)
            all_times.append(elapsed)
        
        # Average across runs
        avg_best_scores = np.mean(all_best_scores, axis=0).tolist()
        avg_query_counts = np.mean(all_query_counts, axis=0).tolist()
        
        results[method_name] = BenchmarkResult(
            method=method_name,
            best_scores=avg_best_scores,
            query_counts=avg_query_counts,
            final_best=avg_best_scores[-1],
            final_queries=int(avg_query_counts[-1]),
            total_time=np.mean(all_times),
        )
    
    # Print summary
    print_summary(results, optimal_score)
    
    # Plot
    plot_results(results, optimal_score, "nasbench201_results.png")
    
    # Save JSON
    if save_results:
        save_path = Path("nasbench201_results.json")
        with open(save_path, "w") as f:
            json.dump({k: asdict(v) for k, v in results.items()}, f, indent=2)
        print(f"\nSaved results to: {save_path}")
    
    return results


def print_summary(results: Dict[str, BenchmarkResult], optimal: float):
    """Print summary table."""
    print("\n" + "=" * 80)
    print("NAS-BENCH-201 BENCHMARK SUMMARY")
    print(f"Optimal validation accuracy: {optimal:.4f}")
    print("=" * 80)
    print(f"{'Method':<25} {'Best Acc':<12} {'Regret':<12} {'Queries':<12} {'Time (s)':<12}")
    print("-" * 80)
    
    for method_name, result in results.items():
        regret = optimal - result.final_best
        print(f"{method_name:<25} {result.final_best:<12.4f} {regret:<12.4f} "
              f"{result.final_queries:<12d} {result.total_time:<12.2f}")
    
    print("=" * 80)
    
    # Determine winner
    best_method = min(results.items(), key=lambda x: optimal - x[1].final_best)
    print(f"\n🏆 Winner: {best_method[0]} with {best_method[1].final_best:.4f} accuracy")


def plot_results(
    results: Dict[str, BenchmarkResult], 
    optimal: float,
    save_path: str = "nasbench201_results.png"
):
    """Plot benchmark results."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {
        "HyperbolicBO": "#e63946",
        "RegularizedEvolution": "#457b9d",
        "RandomSearch": "#2a9d8f",
    }
    
    markers = {
        "HyperbolicBO": "o",
        "RegularizedEvolution": "s",
        "RandomSearch": "^",
    }
    
    # Plot 1: Best accuracy vs iterations
    ax1 = axes[0]
    for method_name, result in results.items():
        iterations = list(range(1, len(result.best_scores) + 1))
        ax1.plot(
            iterations,
            result.best_scores,
            label=method_name,
            color=colors.get(method_name, "gray"),
            marker=markers.get(method_name, "o"),
            markersize=4,
            linewidth=2,
            markevery=5,
        )
    
    ax1.axhline(y=optimal, color='gold', linestyle='--', linewidth=2, label=f'Optimal ({optimal:.3f})')
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Validation Accuracy", fontsize=12)
    ax1.set_title("NAS-Bench-201: Convergence by Iteration", fontsize=14)
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.3, 1.0)
    
    # Plot 2: Best accuracy vs queries (sample efficiency)
    ax2 = axes[1]
    for method_name, result in results.items():
        ax2.plot(
            result.query_counts,
            result.best_scores,
            label=f"{method_name}",
            color=colors.get(method_name, "gray"),
            marker=markers.get(method_name, "o"),
            markersize=4,
            linewidth=2,
            markevery=5,
        )
    
    ax2.axhline(y=optimal, color='gold', linestyle='--', linewidth=2, label='Optimal')
    ax2.set_xlabel("Number of Architecture Evaluations", fontsize=12)
    ax2.set_ylabel("Validation Accuracy", fontsize=12)
    ax2.set_title("NAS-Bench-201: Sample Efficiency", fontsize=14)
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.3, 1.0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {save_path}")
    plt.show()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║           HyperbolicBO vs Baselines on NAS-Bench-201                ║
    ║                                                                      ║
    ║  Comparing:                                                          ║
    ║  • HyperbolicBO (Poincaré geometry + Thompson Sampling)             ║
    ║  • Regularized Evolution (AgingEvolution)                           ║
    ║  • Random Search (baseline)                                         ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    results = run_nas_benchmark(
        n_iterations=30,
        batch_size=4,
        n_runs=3,
    )
    
    print("\n✅ NAS-Bench-201 benchmark complete!")
