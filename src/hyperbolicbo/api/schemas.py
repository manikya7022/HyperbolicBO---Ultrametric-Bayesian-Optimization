"""Pydantic schemas for API requests/responses."""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    """A pipeline configuration."""
    cells: Optional[List[Dict[str, Any]]] = Field(
        None, description="NAS cell operations"
    )
    stages: Optional[List[Dict[str, Any]]] = Field(
        None, description="AutoML pipeline stages"
    )
    resource_chain: Optional[List[str]] = Field(
        None, description="FHIR resource chain"
    )
    features: Optional[Dict[str, Any]] = Field(
        None, description="FHIR feature specifications"
    )
    
    class Config:
        extra = "allow"  # Allow additional fields


class AcquireRequest(BaseModel):
    """Request for next pipeline suggestions."""
    n_suggestions: int = Field(
        default=1, ge=1, le=100,
        description="Number of suggestions to return"
    )
    candidates: Optional[List[PipelineConfig]] = Field(
        None, description="Optional candidate pool to select from"
    )


class AcquireResponse(BaseModel):
    """Response with suggested pipelines."""
    suggestions: List[PipelineConfig]
    embeddings: List[List[float]] = Field(
        description="Poincaré ball embeddings of suggestions"
    )
    acquisition_values: Optional[List[float]] = Field(
        None, description="Acquisition function values"
    )


class ObserveRequest(BaseModel):
    """Request to report evaluations."""
    pipelines: List[PipelineConfig]
    scores: List[float] = Field(description="Evaluation scores (higher is better)")


class ObserveResponse(BaseModel):
    """Response after observing new data."""
    n_total_observations: int
    best_score: float
    best_pipeline: PipelineConfig


class BestResponse(BaseModel):
    """Response with best pipeline so far."""
    pipeline: PipelineConfig
    score: float
    embedding: List[float]
    n_observations: int


class VisualizationPoint(BaseModel):
    """A point for Poincaré disk visualization."""
    x: float
    y: float
    score: Optional[float] = None
    pipeline_hash: Optional[str] = None


class VisualizeResponse(BaseModel):
    """Response with visualization data."""
    observations: List[VisualizationPoint]
    best_point: VisualizationPoint
    disk_boundary: List[List[float]] = Field(
        default_factory=lambda: [[1.0 * cos, 1.0 * sin] 
                                  for cos, sin in zip([1, 0, -1, 0], [0, 1, 0, -1])],
        description="Points defining unit disk boundary"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["healthy", "unhealthy"]
    n_observations: int
    dimension: int
    acquisition_type: str


class OptimizerConfig(BaseModel):
    """Configuration for optimizer initialization."""
    dim: int = Field(default=2, ge=2, le=16)
    acquisition: Literal["ei", "thompson"] = "thompson"
    pipeline_type: Literal["nas", "automl", "fhir"] = "automl"
    n_parallel: int = Field(default=4, ge=1, le=32)
    xi: float = Field(default=0.01, ge=0.0, le=1.0)


class InitializeResponse(BaseModel):
    """Response after initializing optimizer."""
    status: str = "initialized"
    config: OptimizerConfig
