"""FastAPI routes for HyperbolicBO service."""

from __future__ import annotations

from typing import Optional
import hashlib
import json
import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from .schemas import (
    AcquireRequest, AcquireResponse,
    ObserveRequest, ObserveResponse,
    BestResponse, VisualizeResponse, VisualizationPoint,
    HealthResponse, OptimizerConfig, InitializeResponse,
    PipelineConfig,
)
from ..optimizer.hbo import HyperbolicBO


# Create FastAPI app
app = FastAPI(
    title="HyperbolicBO API",
    description="Ultrametric Bayesian Optimization using Poincaré ball geometry",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global optimizer instance (will be initialized)
_optimizer: Optional[HyperbolicBO] = None
_config: Optional[OptimizerConfig] = None


def get_optimizer() -> HyperbolicBO:
    """Get or create optimizer instance."""
    global _optimizer, _config
    if _optimizer is None:
        _optimizer = HyperbolicBO()
        _config = OptimizerConfig()
    return _optimizer


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "HyperbolicBO API",
        "version": "0.1.0",
        "docs_url": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    opt = get_optimizer()
    return HealthResponse(
        status="healthy",
        n_observations=opt.n_observations,
        dimension=opt.dim,
        acquisition_type=opt.acquisition,
    )


@app.post("/initialize", response_model=InitializeResponse)
async def initialize(config: OptimizerConfig):
    """Initialize optimizer with custom configuration."""
    global _optimizer, _config
    
    _optimizer = HyperbolicBO(
        dim=config.dim,
        acquisition=config.acquisition,
        pipeline_type=config.pipeline_type,
        n_parallel=config.n_parallel,
        xi=config.xi,
    )
    _config = config
    
    return InitializeResponse(status="initialized", config=config)


@app.post("/acquire", response_model=AcquireResponse)
async def acquire(request: AcquireRequest):
    """Get next pipeline suggestions.
    
    Uses Thompson Sampling or Expected Improvement to select
    promising pipeline configurations to evaluate next.
    """
    opt = get_optimizer()
    
    # Convert candidates if provided
    candidates = None
    if request.candidates:
        candidates = [c.model_dump(exclude_none=True) for c in request.candidates]
    
    # Get suggestions (async-compatible by running in executor)
    loop = asyncio.get_event_loop()
    suggestions = await loop.run_in_executor(
        None,
        lambda: opt.suggest(request.n_suggestions, candidates)
    )
    
    # Get embeddings
    embeddings = opt._encoder.encode_batch(suggestions)
    
    # Get acquisition values if using EI and have observations
    acq_values = None
    if opt.acquisition == "ei" and opt.n_observations > 0:
        from ..acquisition.ei import hyperbolic_ei
        y_best = opt._scores.max().item()
        acq_values = hyperbolic_ei(embeddings, opt._gp, y_best).tolist()
    
    return AcquireResponse(
        suggestions=[PipelineConfig(**s) for s in suggestions],
        embeddings=embeddings.tolist(),
        acquisition_values=acq_values,
    )


@app.post("/observe", response_model=ObserveResponse)
async def observe(request: ObserveRequest):
    """Report pipeline evaluation results.
    
    Updates the GP model with new observations.
    """
    opt = get_optimizer()
    
    if len(request.pipelines) != len(request.scores):
        raise HTTPException(
            status_code=400,
            detail=f"Length mismatch: {len(request.pipelines)} pipelines vs {len(request.scores)} scores"
        )
    
    # Convert pipelines
    pipelines = [p.model_dump(exclude_none=True) for p in request.pipelines]
    
    # Observe (run in executor to not block)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: opt.observe(pipelines, request.scores)
    )
    
    best_pipeline, best_score = opt.best()
    
    return ObserveResponse(
        n_total_observations=opt.n_observations,
        best_score=best_score,
        best_pipeline=PipelineConfig(**best_pipeline),
    )


@app.get("/best", response_model=BestResponse)
async def best():
    """Get the best pipeline observed so far."""
    opt = get_optimizer()
    
    if opt.n_observations == 0:
        raise HTTPException(
            status_code=400,
            detail="No observations yet. Call /observe first."
        )
    
    best_pipeline, best_score = opt.best()
    embedding = opt._encoder.encode(best_pipeline)
    
    return BestResponse(
        pipeline=PipelineConfig(**best_pipeline),
        score=best_score,
        embedding=embedding.tolist(),
        n_observations=opt.n_observations,
    )


@app.get("/visualize", response_model=VisualizeResponse)
async def visualize():
    """Get Poincaré disk visualization data.
    
    Returns all observations projected to 2D for visualization.
    """
    opt = get_optimizer()
    
    if opt.n_observations == 0:
        return VisualizeResponse(
            observations=[],
            best_point=VisualizationPoint(x=0, y=0),
        )
    
    # Get embeddings and scores
    embeddings = opt._embeddings
    scores = opt._scores
    
    # Project to 2D if higher dimension
    if opt.dim > 2:
        # Simple projection: take first 2 dimensions
        # Could use Lorentz PCA for better projection
        embeddings_2d = embeddings[:, :2]
    else:
        embeddings_2d = embeddings
    
    # Create visualization points
    observations = []
    for i in range(len(embeddings_2d)):
        pipeline_hash = hashlib.md5(
            json.dumps(opt._pipelines[i], sort_keys=True).encode()
        ).hexdigest()[:8]
        
        observations.append(VisualizationPoint(
            x=float(embeddings_2d[i, 0]),
            y=float(embeddings_2d[i, 1]),
            score=float(scores[i]),
            pipeline_hash=pipeline_hash,
        ))
    
    # Best point
    best_idx = scores.argmax().item()
    best_point = VisualizationPoint(
        x=float(embeddings_2d[best_idx, 0]),
        y=float(embeddings_2d[best_idx, 1]),
        score=float(scores[best_idx]),
    )
    
    # Unit disk boundary (for plotting)
    n_boundary = 100
    angles = np.linspace(0, 2 * np.pi, n_boundary)
    boundary = [[float(np.cos(a)), float(np.sin(a))] for a in angles]
    
    return VisualizeResponse(
        observations=observations,
        best_point=best_point,
        disk_boundary=boundary,
    )


@app.delete("/reset")
async def reset():
    """Reset optimizer to initial state."""
    global _optimizer
    _optimizer = None
    return {"status": "reset", "message": "Optimizer cleared"}


# Entry point for uvicorn
def create_app() -> FastAPI:
    """Create and configure the FastAPI app."""
    return app
