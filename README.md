# HyperbolicBO: Ultrametric Bayesian Optimization

A Bayesian Optimization framework that uses the Poincare ball model of hyperbolic geometry for optimization over tree-structured configuration spaces.

## Table of Contents

1. [Introduction](#introduction)
2. [Background](#background)
3. [Mathematical Foundation](#mathematical-foundation)
4. [Architecture](#architecture)
5. [Installation](#installation)
6. [Usage](#usage)
7. [API Reference](#api-reference)
8. [Benchmarks](#benchmarks)
9. [Limitations](#limitations)
10. [References](#references)

---

## Introduction

HyperbolicBO is an implementation of Bayesian Optimization that uses hyperbolic geometry to represent tree-structured search spaces. The implementation uses the Poincare ball model, where configurations are embedded as points inside the unit ball, and distances are computed using the hyperbolic metric.

This project provides:
- Poincare ball geometry operations (distance, Mobius addition, exponential/logarithmic maps)
- Gaussian Process regression with hyperbolic distance-based kernels
- Thompson Sampling and Expected Improvement acquisition functions
- Pipeline encoders for NAS, AutoML, and FHIR configuration types
- A REST API for integration with external systems

---

## Background

### Motivation

Tree-structured search spaces appear in several machine learning optimization problems:

- **Neural Architecture Search**: Cell-based architectures form directed acyclic graphs
- **AutoML Pipeline Selection**: Sequential choices of preprocessing and model components
- **Healthcare Data Processing**: Clinical resource chains following medical ontologies

Standard Bayesian Optimization uses Euclidean distance in its kernel functions. Hyperbolic spaces have the geometric property that tree structures can be embedded with lower distortion than in Euclidean spaces, as shown by Nickel and Kiela (2017).

### Approach

This implementation:
1. Embeds pipeline configurations into the Poincare ball
2. Uses hyperbolic distance in the Gaussian Process kernel
3. Applies acquisition functions adapted for the Riemannian manifold

---

## Mathematical Foundation

### Poincare Ball Model

The Poincare ball is the open unit ball B^n = {x in R^n : ||x|| < 1} equipped with the Riemannian metric tensor:

```
g_x = (2 / (1 - ||x||^2))^2 * I
```

### Hyperbolic Distance

The geodesic distance between two points u and v in the Poincare ball is:

```
d(u, v) = arcosh(1 + 2 * ||u - v||^2 / ((1 - ||u||^2) * (1 - ||v||^2)))
```

where arcosh is the inverse hyperbolic cosine function.

### Mobius Addition

The Mobius addition of two points u and v is defined as:

```
u (+) v = ((1 + 2<u,v> + ||v||^2) * u + (1 - ||u||^2) * v) / (1 + 2<u,v> + ||u||^2 * ||v||^2)
```

This operation is the hyperbolic analog of vector addition.

### Exponential Map

The exponential map at point x for tangent vector v is:

```
exp_x(v) = x (+) (tanh(lambda_x * ||v|| / 2) * v / ||v||)
```

where lambda_x = 2 / (1 - ||x||^2) is the conformal factor and (+) denotes Mobius addition.

### Logarithmic Map

The logarithmic map at point x for target point y is:

```
log_x(y) = (2 / lambda_x) * arctanh(||(-x) (+) y||) * ((-x) (+) y) / ||(-x) (+) y||
```

### Hyperbolic Kernel

The RBF kernel using hyperbolic distance:

```
K(x, y) = sigma^2 * exp(-d(x, y)^2 / (2 * l^2))
```

where d(x, y) is the Poincare distance, sigma^2 is the output variance, and l is the lengthscale.

---

## Architecture

### Module Structure

```
src/hyperbolicbo/
    geometry/
        poincare.py         # Poincare ball distance and operations
        horosphere.py       # Approximate nearest neighbor indexing
    gp/
        kernels.py          # Hyperbolic covariance functions
        hyperbolic_gp.py    # GPyTorch-based Gaussian Process
    acquisition/
        ei.py               # Expected Improvement
        thompson.py         # Thompson Sampling with Fourier features
    embeddings/
        pipeline_encoder.py # Configuration to embedding conversion
    optimizer/
        hbo.py              # Main optimization loop
    api/
        schemas.py          # Pydantic models
        routes.py           # FastAPI endpoints
    benchmarks/
        __init__.py         # NAS-Bench-201 adapter
```

---

## Installation

### Requirements

- Python 3.10 or higher
- PyTorch 2.0 or higher

### Install from Source

```bash
git clone https://github.com/manikya7022/HyperbolicBO---Ultrametric-Bayesian-Optimization.git
cd HyperbolicBO---Ultrametric-Bayesian-Optimization
pip install -e .
```

### Install with Development Dependencies

```bash
pip install -e ".[dev]"
```

### Core Dependencies

- geoopt: Riemannian optimization
- gpytorch: Gaussian Process inference
- torch: Tensor computation
- fastapi: REST API
- networkx: Graph operations
- numpy, scipy: Numerical computation

---

## Usage

### Basic Example

```python
from hyperbolicbo import HyperbolicBO

# Initialize optimizer
optimizer = HyperbolicBO(
    dim=4,                      # Poincare ball dimension
    acquisition="thompson",     # Acquisition function
    pipeline_type="automl",     # Configuration type
)

# Define objective function
def objective(config):
    # Your evaluation logic here
    return score

# Optimization loop
for iteration in range(20):
    suggestions = optimizer.suggest(n_suggestions=4)
    scores = [objective(config) for config in suggestions]
    optimizer.observe(suggestions, scores)

best_config, best_score = optimizer.best()
```

### REST API

Start the server:

```bash
uvicorn hyperbolicbo.api.routes:app --host 0.0.0.0 --port 8000
```

Request suggestions:

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

---

## API Reference

### HyperbolicBO

**Constructor:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| dim | int | 2 | Embedding dimension |
| acquisition | str | "thompson" | "thompson" or "ei" |
| pipeline_type | str | "automl" | "nas", "automl", or "fhir" |
| n_parallel | int | 4 | Default batch size |
| device | str | "cpu" | Computation device |

**Methods:**

| Method | Description |
|--------|-------------|
| suggest(n_suggestions) | Return list of suggested configurations |
| observe(pipelines, scores) | Record evaluation results |
| best() | Return (best_config, best_score) tuple |

### Geometry Functions

| Function | Description |
|----------|-------------|
| poincare_distance(u, v) | Hyperbolic distance between points |
| mobius_add(u, v) | Mobius addition |
| exp_map(v, x) | Exponential map at x |
| log_map(y, x) | Logarithmic map at x |
| project_to_ball(x) | Project to valid ball region |

---

## Benchmarks

### Synthetic Benchmark

```bash
python examples/benchmark.py
```

This runs a comparison on a synthetic objective function. Results use synthetic data and should not be interpreted as representative of real-world performance.

### NAS-Bench-201 Benchmark

```bash
python examples/benchmark_nasbench201.py
```

By default, this uses synthetic accuracy values for testing. The synthetic mode does not reflect actual neural network performance.

To run with real NAS-Bench-201 data:

1. Install the API: `pip install nas-bench-201`
2. Download benchmark data from the official NAS-Bench-201 repository
3. Run with data path: `python examples/benchmark_nasbench201.py --data_path /path/to/data.pth`

### Test Suite

```bash
pytest tests/ -v
```

Current status: 39 tests passing.

---

## Limitations

1. **Gaussian Process Scaling**: The underlying GP uses exact inference with O(n^3) complexity for n observations. The horosphere indexing provides approximate predictions but does not eliminate this cost during model fitting.

2. **Kernel Validity**: The hyperbolic RBF kernel is positive semi-definite, but numerical precision issues may arise for points very close to the ball boundary.

3. **Benchmark Results**: The included synthetic benchmarks do not represent real-world optimization performance. Validation on actual tasks requires integration with real evaluation systems.

4. **Dimension Selection**: The heuristic `dim = min(8, ceil(log2(max_degree)))` is based on theoretical tree embedding results but has not been validated empirically for optimization performance.

5. **Thompson Sampling Approximation**: The Fourier features approximation introduces error in the posterior samples. The quality of this approximation depends on the number of features used.

---

## References

Nickel, M., & Kiela, D. (2017). Poincare Embeddings for Learning Hierarchical Representations. Advances in Neural Information Processing Systems (NeurIPS).

Ganea, O., Becigneul, G., & Hofmann, T. (2018). Hyperbolic Neural Networks. Advances in Neural Information Processing Systems (NeurIPS).

Dong, X., & Yang, Y. (2020). NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search. International Conference on Learning Representations (ICLR).

Real, E., Aggarwal, A., Huang, Y., & Le, Q. V. (2019). Regularized Evolution for Image Classifier Architecture Search. AAAI Conference on Artificial Intelligence.

Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian Optimization of Machine Learning Algorithms. Advances in Neural Information Processing Systems (NeurIPS).

Becigneul, G., & Ganea, O. (2019). Riemannian Adaptive Optimization Methods. International Conference on Learning Representations (ICLR).
