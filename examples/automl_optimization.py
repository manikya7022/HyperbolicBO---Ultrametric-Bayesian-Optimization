"""Example: AutoML pipeline optimization with HyperbolicBO.

This example demonstrates how to use HyperbolicBO to optimize
sklearn-style machine learning pipelines.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Import HyperbolicBO
import sys
sys.path.insert(0, "../src")

from hyperbolicbo import HyperbolicBO


def build_sklearn_pipeline(config: dict) -> Pipeline:
    """Convert HyperbolicBO config to sklearn Pipeline."""
    steps = []
    
    for stage in config.get("stages", []):
        stage_type = stage.get("type", "")
        
        if stage_type == "scaler":
            method = stage.get("method", "StandardScaler")
            if method == "StandardScaler":
                steps.append(("scaler", StandardScaler()))
            else:
                steps.append(("scaler", MinMaxScaler()))
        
        elif stage_type == "pca":
            n_components = min(stage.get("n_components", 10), 20)
            steps.append(("pca", PCA(n_components=n_components)))
        
        elif stage_type == "lr":
            steps.append(("model", LogisticRegression(max_iter=200)))
        
        elif stage_type == "rf":
            n_estimators = stage.get("n_estimators", 100)
            max_depth = stage.get("max_depth", 5)
            steps.append(("model", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
            )))
        
        elif stage_type == "svm":
            steps.append(("model", SVC(kernel="rbf")))
    
    # Ensure we have a model
    if not any(step[0] == "model" for step in steps):
        steps.append(("model", LogisticRegression(max_iter=200)))
    
    return Pipeline(steps)


def evaluate_pipeline(config: dict, X: np.ndarray, y: np.ndarray) -> float:
    """Evaluate a pipeline config using cross-validation."""
    try:
        pipeline = build_sklearn_pipeline(config)
        scores = cross_val_score(pipeline, X, y, cv=3, scoring="accuracy")
        return scores.mean()
    except Exception as e:
        print(f"Pipeline failed: {e}")
        return 0.0


def main():
    """Run AutoML optimization example."""
    print("=" * 60)
    print("HyperbolicBO AutoML Example")
    print("=" * 60)
    
    # Create synthetic classification dataset
    print("\n1. Creating dataset...")
    X, y = make_classification(
        n_samples=500,
        n_features=30,
        n_informative=15,
        n_redundant=5,
        random_state=42,
    )
    print(f"   Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Initialize HyperbolicBO
    print("\n2. Initializing HyperbolicBO...")
    optimizer = HyperbolicBO(
        dim=4,                      # 4D Poincaré ball
        acquisition="thompson",     # Parallel Thompson Sampling
        pipeline_type="automl",     # AutoML pipeline optimization
        n_parallel=4,               # 4 parallel suggestions
    )
    print(f"   Dimension: {optimizer.dim}")
    print(f"   Acquisition: {optimizer.acquisition}")
    
    # Create objective function
    def objective(config):
        return evaluate_pipeline(config, X, y)
    
    # Run optimization
    print("\n3. Running optimization...")
    n_iterations = 10
    
    for i in range(n_iterations):
        # Get suggestions
        suggestions = optimizer.suggest(n_suggestions=4)
        
        # Evaluate each suggestion
        scores = [objective(s) for s in suggestions]
        
        # Report observations
        optimizer.observe(suggestions, scores)
        
        # Print progress
        best_pipeline, best_score = optimizer.best()
        print(f"   Iter {i+1:2d}: batch_best={max(scores):.4f}, overall_best={best_score:.4f}")
    
    # Final results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    
    best_pipeline, best_score = optimizer.best()
    print(f"\nBest Score: {best_score:.4f}")
    print(f"Best Pipeline: {best_pipeline}")
    
    # Show all pipelines in history
    print(f"\nTotal pipelines evaluated: {optimizer.n_observations}")
    
    # Visualize (if 2D)
    if optimizer.dim == 2:
        print("\nPoincaré disk embeddings (first few):")
        for h in optimizer.history[:5]:
            emb = h["embedding"]
            print(f"  Score {h['score']:.3f}: ({emb[0]:.3f}, {emb[1]:.3f})")


if __name__ == "__main__":
    main()
