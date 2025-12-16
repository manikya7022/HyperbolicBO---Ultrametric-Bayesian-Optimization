# HyperbolicBO: Ultrametric Bayesian Optimization

A novel Bayesian Optimization framework that leverages hyperbolic geometry for efficient optimization over tree-structured and hierarchical search spaces.

## Table of Contents

1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Technical Approach](#technical-approach)
4. [Mathematical Foundation](#mathematical-foundation)
5. [Architecture](#architecture)
6. [Installation](#installation)
7. [Usage](#usage)
8. [API Reference](#api-reference)
9. [Benchmarks](#benchmarks)
10. [Technical Details](#technical-details)

---

## Introduction

HyperbolicBO is a research implementation of Bayesian Optimization that uses the Poincare ball model of hyperbolic geometry to efficiently search tree-structured configuration spaces. Traditional Bayesian Optimization methods use Euclidean distance metrics, which fail to capture the hierarchical relationships inherent in many machine learning search spaces such as neural architecture search, AutoML pipeline optimization, and healthcare data processing workflows.

This implementation provides a complete framework including hyperbolic geometry operations, Gaussian Process regression with hyperbolic kernels, acquisition functions adapted for Riemannian manifolds, and a REST API for integration into existing workflows.

---

## Problem Statement

### Limitations of Traditional Bayesian Optimization

Standard Bayesian Optimization faces two fundamental challenges when applied to hierarchical search spaces:

1. **Euclidean Distance Failure**: Tree-structured configurations do not naturally embed into Euclidean space. Two neural architectures that share a common parent structure may be geometrically distant in Euclidean space despite being functionally similar.

2. **Cubic Scaling**: Gaussian Process inference requires O(n^3) matrix operations, making it computationally prohibitive for large-scale architecture search with thousands of evaluated configurations.

### Target Applications

HyperbolicBO is designed for optimization problems with inherent tree or hierarchical structure:

- **Neural Architecture Search (NAS)**: Cell-based architecture representations where operations form directed acyclic graphs
- **AutoML Pipelines**: Sequential preprocessing, feature engineering, and model selection choices
- **Healthcare FHIR Pipelines**: Clinical data processing chains following hierarchical medical ontologies

---

## Technical Approach

### Hyperbolic Geometry for Tree Embedding

The Poincare ball model provides a natural embedding space for tree-structured data. In this model, the n-dimensional unit ball B^n = {x in R^n : ||x|| < 1} is equipped with a Riemannian metric that expands exponentially toward the boundary.

Key properties that make hyperbolic space suitable for tree embedding:

1. **Exponential Volume Growth**: The volume of a hyperbolic ball grows exponentially with radius, matching the exponential growth of tree node counts with depth.

2. **Natural Hierarchy Encoding**: Points near the center represent root-level configurations; points near the boundary represent deeply nested configurations.

3. **Preserved Tree Distances**: The hyperbolic distance between two points approximates their tree distance (path length through lowest common ancestor).

### Horosphere Clustering for O(1) Lookup

Horospheres are hyperbolic analogs of hyperplanes. By clustering observed configurations based on their Busemann function values (distance to a reference boundary direction), HyperbolicBO achieves constant-time nearest neighbor queries for acquisition function evaluation.

### Hyperbolic Gaussian Process

The covariance function of the Gaussian Process uses hyperbolic distance instead of Euclidean distance:

K(x, y) = sigma^2 * exp(-d_H(x, y)^2 / (2 * l^2))

where d_H is the Poincare ball distance and l is the lengthscale hyperparameter.

---

## Mathematical Foundation

### Poincare Ball Distance

The distance between two points u and v in the Poincare ball is defined as:

```
d(u, v) = arcosh(1 + 2 * ||u - v||^2 / ((1 - ||u||^2) * (1 - ||v||^2)))
```

This distance has the following properties:
- d(u, v) = 0 if and only if u = v
- d(u, v) = d(v, u) (symmetry)
- d(u, v) + d(v, w) >= d(u, w) (triangle inequality)
- Distance grows without bound as points approach the boundary

### Mobius Addition

Vector addition in hyperbolic space is defined by Mobius addition:

```
u + v = ((1 + 2<u,v> + ||v||^2) * u + (1 - ||u||^2) * v) / (1 + 2<u,v> + ||u||^2 * ||v||^2)
```

This operation respects the Riemannian metric of the Poincare ball.

### Exponential and Logarithmic Maps

The exponential map projects tangent vectors onto the manifold:

```
exp_x(v) = x + tanh(lambda_x * ||v|| / 2) * v / ||v||
```

where lambda_x = 2 / (1 - ||x||^2) is the conformal factor.

The logarithmic map performs the inverse operation, used for computing Riemannian gradients.

### Busemann Function

The Busemann function measures the "height" of a point relative to a boundary direction:

```
b_xi(x) = log((1 - ||x||^2) / ||xi - x||^2)
```

Points with equal Busemann values lie on the same horosphere.

---

## Architecture

### Module Structure

```
src/hyperbolicbo/
    geometry/
        poincare.py         # Poincare ball operations
        horosphere.py       # O(1) nearest neighbor indexing
    gp/
        kernels.py          # Hyperbolic covariance functions
        hyperbolic_gp.py    # Gaussian Process implementation
    acquisition/
        ei.py               # Expected Improvement
        thompson.py         # Parallelized Thompson Sampling
    embeddings/
        pipeline_encoder.py # Configuration to embedding mapping
    optimizer/
        hbo.py              # Main HyperbolicBO class
    api/
        schemas.py          # Pydantic request/response models
        routes.py           # FastAPI endpoints
    benchmarks/
        __init__.py         # NAS-Bench-201 adapter
```

### Component Interactions

1. **PipelineEncoder** converts configuration dictionaries to Poincare ball points
2. **HyperbolicGP** maintains a Gaussian Process model with hyperbolic kernel
3. **HorosphereIndex** provides O(1) approximate nearest neighbor queries
4. **Acquisition Functions** suggest next configurations using GP predictions
5. **HyperbolicBO** orchestrates the optimization loop

---

## Installation

### Requirements

- Python 3.10 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU acceleration)

### Installation from Source

```bash
git clone https://github.com/manikya7022/HyperbolicBO---Ultrametric-Bayesian-Optimization.git
cd HyperbolicBO---Ultrametric-Bayesian-Optimization
pip install -e .
```

### Installation with Development Dependencies

```bash
pip install -e ".[dev]"
```

### Dependencies

Core dependencies:
- geoopt: Riemannian optimization in PyTorch
- gpytorch: Gaussian Process inference
- torch: Neural network backend
- fastapi: REST API framework
- uvicorn: ASGI server
- mlflow: Experiment tracking
- networkx: Graph operations
- numpy, scipy: Numerical computing
- matplotlib: Visualization

---

## Usage

### Basic Optimization Loop

```python
from hyperbolicbo import HyperbolicBO

# Initialize optimizer
optimizer = HyperbolicBO(
    dim=8,                      # Poincare ball dimension
    acquisition="thompson",     # Acquisition function
    pipeline_type="automl",     # Configuration type
    n_parallel=4,               # Batch size
)

# Define objective function
def objective(config):
    # Evaluate configuration and return score
    return evaluate(config)

# Optimization loop
for iteration in range(20):
    # Get suggested configurations
    suggestions = optimizer.suggest(n_suggestions=4)
    
    # Evaluate each suggestion
    scores = [objective(config) for config in suggestions]
    
    # Report observations
    optimizer.observe(suggestions, scores)

# Retrieve best configuration
best_config, best_score = optimizer.best()
```

### Running the Full Optimization

```python
from hyperbolicbo import HyperbolicBO

optimizer = HyperbolicBO(dim=4, acquisition="thompson")

result = optimizer.run(
    objective=my_objective_function,
    n_iterations=30,
    verbose=True,
)

print(f"Best score: {result.best_score}")
print(f"Best configuration: {result.best_pipeline}")
print(f"Total evaluations: {len(result.history)}")
```

### Using the REST API

Start the server:

```bash
uvicorn hyperbolicbo.api.routes:app --host 0.0.0.0 --port 8000
```

Request next suggestions:

```bash
curl -X POST http://localhost:8000/acquire \
    -H "Content-Type: application/json" \
    -d '{"n_suggestions": 4}'
```

Report observations:

```bash
curl -X POST http://localhost:8000/observe \
    -H "Content-Type: application/json" \
    -d '{
        "pipelines": [{"stages": [{"type": "xgb"}]}],
        "scores": [0.85]
    }'
```

### Neural Architecture Search Example

```python
from hyperbolicbo import HyperbolicBO
from hyperbolicbo.benchmarks import NASBench201, NASBench201Objective

# Initialize NAS-Bench-201
benchmark = NASBench201(use_synthetic=True)
objective = NASBench201Objective(benchmark, metric="valid_acc")

# Initialize optimizer for NAS
optimizer = HyperbolicBO(
    dim=4,
    acquisition="thompson",
    pipeline_type="nas",
)

# Run optimization
for i in range(30):
    archs = optimizer.suggest(n_suggestions=4)
    scores = [objective(arch) for arch in archs]
    optimizer.observe(archs, scores)

best_arch, best_acc = optimizer.best()
print(f"Best architecture accuracy: {best_acc}")
```

---

## API Reference

### HyperbolicBO Class

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| dim | int | 2 | Poincare ball dimension |
| acquisition | str | "thompson" | Acquisition function: "thompson" or "ei" |
| pipeline_type | str | "automl" | Configuration type: "nas", "automl", or "fhir" |
| n_parallel | int | 4 | Default batch size |
| xi | float | 0.01 | Exploration parameter for Expected Improvement |
| device | str | "cpu" | Computation device |

**Methods:**

| Method | Arguments | Returns | Description |
|--------|-----------|---------|-------------|
| suggest | n_suggestions, candidates | List[Dict] | Get next configurations to evaluate |
| observe | pipelines, scores | self | Record evaluation results |
| best | - | Tuple[Dict, float] | Get best observed configuration |
| run | objective, n_iterations, verbose | OptimizationResult | Execute full optimization loop |
| auto_dim | max_degree | HyperbolicBO | Create optimizer with automatic dimension selection |

### HyperbolicGP Class

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| dim | int | 2 | Input dimension |
| n_train_iters | int | 50 | Hyperparameter optimization iterations |
| use_horosphere | bool | True | Enable O(1) approximate predictions |
| device | str | "cpu" | Computation device |

**Methods:**

| Method | Arguments | Returns | Description |
|--------|-----------|---------|-------------|
| fit | X, y, verbose | self | Fit GP to observations |
| predict | X, return_std | Tuple[Tensor, Tensor] | Predict mean and standard deviation |
| predict_approximate | X, k_neighbors | Tuple[Tensor, Tensor] | O(1) prediction using horosphere index |
| update | X_new, y_new, refit | self | Add new observations |

### Geometry Functions

**poincare.py:**

| Function | Description |
|----------|-------------|
| poincare_distance(u, v) | Compute hyperbolic distance between points |
| poincare_distance_matrix(X, Y) | Compute pairwise distance matrix |
| mobius_add(u, v) | Mobius addition of two points |
| exp_map(v, x) | Exponential map at point x |
| log_map(y, x) | Logarithmic map at point x |
| project_to_ball(x) | Project point into valid ball region |
| geodesic(x, y, t) | Point on geodesic at parameter t |
| adaptive_dimension(max_degree) | Compute optimal dimension for tree degree |

---

## Benchmarks

### Synthetic Benchmark

```bash
python examples/benchmark.py
```

Compares HyperbolicBO, EuclideanBO, and RandomSearch on a synthetic tree-structured objective.

### NAS-Bench-201 Benchmark

```bash
python examples/benchmark_nasbench201.py
```

Compares HyperbolicBO, Regularized Evolution, and RandomSearch on the NAS-Bench-201 architecture search benchmark with 15,625 architectures.

### Running with Real NAS-Bench-201 Data

1. Install the NAS-Bench-201 API:
```bash
pip install nas-bench-201
```

2. Download the benchmark data from the official repository

3. Run with data path:
```bash
python examples/benchmark_nasbench201.py --data_path /path/to/NAS-Bench-201-v1_0.pth
```

---

## Technical Details

### Dimension Selection

The optimal Poincare ball dimension depends on the maximum branching factor of the tree structure:

```
dimension = min(8, ceil(log2(max_degree)))
```

For typical AutoML pipelines with branching factor less than 8, dimension 4 is sufficient. For NAS with larger cell structures, dimension 8 may be required.

### Numerical Stability

Points are clipped to ||x|| < 0.99999 to avoid numerical issues near the boundary. All operations use float32 precision with explicit dtype handling.

### Thompson Sampling Implementation

Parallelized Thompson Sampling uses Random Fourier Features to approximate the GP kernel without O(n^3) matrix inversion:

1. Compute Fourier features phi(X) for training data
2. Solve for weights: w = (Phi^T Phi + lambda I)^-1 Phi^T y
3. Sample random functions via weight perturbation
4. Select argmax for each sample

This achieves O(n) complexity with respect to the number of observations.

### Horosphere Index Implementation

The horosphere index uses the following algorithm:

1. Choose k reference boundary directions using Fibonacci lattice
2. Compute Busemann function values for all observed points
3. Discretize Busemann values into buckets
4. For queries, compute Busemann values and look up corresponding bucket
5. Perform linear search within bucket for exact nearest neighbors

With well-distributed points, bucket size is approximately constant, yielding O(1) query time.

### Configuration Encoding

Pipeline configurations are encoded to Poincare ball points using:

1. Convert configuration to directed graph
2. Compute topological ordering and node depths
3. Generate operation embeddings (pretrained or hash-based)
4. Apply depth-dependent radial scaling
5. Aggregate node embeddings with importance weighting
6. Project result to valid ball region

---

## Running Tests

Execute the test suite:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=hyperbolicbo --cov-report=html
```

Current test status: 39 tests passing with 57% code coverage.

---

## Project Structure

```
HyperbolicBO/
    src/
        hyperbolicbo/
            __init__.py
            geometry/
            gp/
            acquisition/
            embeddings/
            optimizer/
            api/
            benchmarks/
    tests/
        test_poincare.py
        test_gp.py
        test_optimizer.py
    examples/
        automl_optimization.py
        visualize_embedding.py
        benchmark.py
        benchmark_nasbench201.py
    pyproject.toml
    README.md
```

---

## References

1. Nickel, M., & Kiela, D. (2017). Poincare Embeddings for Learning Hierarchical Representations. NeurIPS.

2. Ganea, O., Becigneul, G., & Hofmann, T. (2018). Hyperbolic Neural Networks. NeurIPS.

3. Dong, X., & Yang, Y. (2020). NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search. ICLR.

4. Real, E., et al. (2019). Regularized Evolution for Image Classifier Architecture Search. AAAI.

5. Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian Optimization of Machine Learning Algorithms. NeurIPS.
