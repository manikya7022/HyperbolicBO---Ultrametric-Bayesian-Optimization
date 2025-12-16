"""HyperbolicBO: Main Bayesian Optimization class.

High-level API for hyperbolic Bayesian optimization.
Combines all components: GP, acquisition, embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal, Tuple, Callable
import time

import torch
from torch import Tensor
import numpy as np

from ..gp.hyperbolic_gp import HyperbolicGP
from ..acquisition.ei import hyperbolic_ei, optimize_hyperbolic_ei
from ..acquisition.thompson import parallel_hyperbolic_ts, adaptive_batch_size
from ..embeddings.pipeline_encoder import PipelineEncoder, PipelineType
from ..geometry.poincare import project_to_ball, adaptive_dimension


AcquisitionType = Literal["ei", "thompson"]


@dataclass
class OptimizationResult:
    """Result from an optimization run."""
    best_pipeline: Dict[str, Any]
    best_score: float
    best_embedding: Tensor
    n_iterations: int
    history: List[Dict[str, Any]]
    total_time: float


@dataclass
class HyperbolicBO:
    """Hyperbolic Bayesian Optimization for tree-structured spaces.
    
    Main entry point for the HyperbolicBO library.
    
    Example:
        >>> optimizer = HyperbolicBO(dim=8, acquisition="thompson")
        >>> for _ in range(20):
        ...     candidates = optimizer.suggest(n_suggestions=4)
        ...     scores = [objective(c) for c in candidates]
        ...     optimizer.observe(candidates, scores)
        >>> best = optimizer.best()
    """
    
    # Configuration
    dim: int = 2
    acquisition: AcquisitionType = "thompson"
    pipeline_type: PipelineType = "automl"
    n_parallel: int = 4
    xi: float = 0.01  # Exploration parameter for EI
    device: str = "cpu"
    
    # Internal state (private)
    _gp: HyperbolicGP = field(init=False, repr=False)
    _encoder: PipelineEncoder = field(init=False, repr=False)
    _pipelines: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    _embeddings: Optional[Tensor] = field(default=None, repr=False)
    _scores: Optional[Tensor] = field(default=None, repr=False)
    _history: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    _iteration: int = field(default=0, repr=False)
    
    def __post_init__(self):
        """Initialize components after dataclass creation."""
        self._gp = HyperbolicGP(
            dim=self.dim,
            use_horosphere=True,
            device=self.device,
        )
        self._encoder = PipelineEncoder(
            dim=self.dim,
            pipeline_type=self.pipeline_type,
        )
    
    @classmethod
    def auto_dim(cls, max_degree: int = 10, **kwargs) -> "HyperbolicBO":
        """Create optimizer with automatically chosen dimension.
        
        Uses formula: dim = min(8, ceil(log2(max_degree)))
        
        Args:
            max_degree: Expected maximum branching factor in pipelines
            **kwargs: Other HyperbolicBO parameters
            
        Returns:
            Configured HyperbolicBO instance
        """
        dim = adaptive_dimension(max_degree)
        return cls(dim=dim, **kwargs)
    
    def suggest(
        self,
        n_suggestions: int = 1,
        candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Suggest next pipelines to evaluate.
        
        Args:
            n_suggestions: Number of suggestions to return
            candidates: Optional pool of candidates to select from
            
        Returns:
            List of suggested pipeline configurations
        """
        self._iteration += 1
        
        # Adaptive batch size if using Thompson
        if self.acquisition == "thompson" and self._gp.n_observations > 0:
            if candidates is not None:
                cand_emb = self._encoder.encode_batch(candidates)
                n_suggestions = min(
                    n_suggestions,
                    adaptive_batch_size(self._gp, cand_emb)
                )
        
        # If no observations yet, return random candidates
        if self._gp.n_observations == 0:
            if candidates is not None:
                idx = np.random.choice(len(candidates), min(n_suggestions, len(candidates)), replace=False)
                return [candidates[i] for i in idx]
            else:
                # Generate random pipelines (simple case)
                return self._generate_random_pipelines(n_suggestions)
        
        # Encode candidates
        if candidates is None:
            # Generate candidate pool
            candidates = self._generate_random_pipelines(n_suggestions * 10)
        
        cand_embeddings = self._encoder.encode_batch(candidates)
        
        # Select using acquisition function
        if self.acquisition == "ei":
            y_best = self._scores.max().item()
            ei_values = hyperbolic_ei(cand_embeddings, self._gp, y_best, self.xi)
            _, indices = ei_values.topk(min(n_suggestions, len(candidates)))
            selected = [candidates[i.item()] for i in indices]
        
        elif self.acquisition == "thompson":
            indices = parallel_hyperbolic_ts(
                self._gp,
                cand_embeddings,
                n_samples=n_suggestions,
            )
            selected = [candidates[i.item()] for i in indices]
        
        else:
            raise ValueError(f"Unknown acquisition: {self.acquisition}")
        
        return selected
    
    def observe(
        self,
        pipelines: List[Dict[str, Any]],
        scores: List[float],
    ) -> "HyperbolicBO":
        """Report observed evaluations.
        
        Args:
            pipelines: Evaluated pipeline configurations
            scores: Observed scores (higher is better)
            
        Returns:
            Self for chaining
        """
        embeddings = self._encoder.encode_batch(pipelines)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        
        # Store history
        for p, s, e in zip(pipelines, scores, embeddings):
            self._history.append({
                "iteration": self._iteration,
                "pipeline": p,
                "score": s,
                "embedding": e.tolist(),
            })
        
        # Accumulate data
        self._pipelines.extend(pipelines)
        
        if self._embeddings is None:
            self._embeddings = embeddings
            self._scores = scores_tensor
        else:
            self._embeddings = torch.cat([self._embeddings, embeddings], dim=0)
            self._scores = torch.cat([self._scores, scores_tensor], dim=0)
        
        # Refit GP
        self._gp.fit(self._embeddings, self._scores)
        
        return self
    
    def best(self) -> Tuple[Dict[str, Any], float]:
        """Get best observed pipeline.
        
        Returns:
            Tuple of (best_pipeline, best_score)
        """
        if self._scores is None or len(self._scores) == 0:
            raise RuntimeError("No observations yet. Call observe() first.")
        
        best_idx = self._scores.argmax().item()
        return self._pipelines[best_idx], self._scores[best_idx].item()
    
    def best_predicted(
        self,
        candidates: Optional[List[Dict[str, Any]]] = None,
        n_candidates: int = 1000,
    ) -> Tuple[Dict[str, Any], float]:
        """Get pipeline with best predicted score.
        
        Uses GP to predict on candidates and returns highest mean.
        
        Args:
            candidates: Optional candidate pool
            n_candidates: Number of random candidates if not provided
            
        Returns:
            Tuple of (predicted_best_pipeline, predicted_mean)
        """
        if self._gp.n_observations == 0:
            raise RuntimeError("No observations yet.")
        
        if candidates is None:
            candidates = self._generate_random_pipelines(n_candidates)
        
        embeddings = self._encoder.encode_batch(candidates)
        means, _ = self._gp.predict(embeddings)
        
        best_idx = means.argmax().item()
        return candidates[best_idx], means[best_idx].item()
    
    def _generate_random_pipelines(self, n: int) -> List[Dict[str, Any]]:
        """Generate random pipeline configurations.
        
        Simple random generation - can be overridden for domain-specific.
        """
        pipelines = []
        
        for _ in range(n):
            if self.pipeline_type == "nas":
                # Random NAS cell
                n_ops = np.random.randint(2, 6)
                cells = []
                for i in range(n_ops):
                    op = np.random.choice([
                        "conv_3x3", "conv_1x1", "skip_connect",
                        "max_pool", "avg_pool", "dil_conv_3x3"
                    ])
                    # Inputs from previous cells or input nodes
                    available = max(1, i)  # At least 1 option
                    n_inputs = np.random.randint(1, min(available, 3) + 1)
                    inputs = list(np.random.choice(range(available), min(n_inputs, available), replace=False))
                    cells.append({"op": op, "input": inputs})
                pipelines.append({"cells": cells})
            
            elif self.pipeline_type == "automl":
                # Random AutoML pipeline
                stages = []
                # Preprocessor
                if np.random.random() > 0.3:
                    stages.append({
                        "type": np.random.choice(["scaler", "normalizer"]),
                        "method": np.random.choice(["StandardScaler", "MinMaxScaler", "RobustScaler"])
                    })
                # Feature selection
                if np.random.random() > 0.5:
                    stages.append({
                        "type": "pca",
                        "n_components": int(np.random.choice([10, 20, 50, 100]))
                    })
                # Model
                model = np.random.choice(["xgb", "rf", "lr", "svm"])
                params = {}
                if model in ["xgb", "rf"]:
                    params["n_estimators"] = int(np.random.choice([50, 100, 200]))
                    params["max_depth"] = int(np.random.choice([3, 5, 7, 10]))
                stages.append({"type": model, **params})
                pipelines.append({"stages": stages})
            
            elif self.pipeline_type == "fhir":
                # Random FHIR pipeline
                resources = ["Patient"]
                n_steps = np.random.randint(1, 4)
                for _ in range(n_steps):
                    next_resource = np.random.choice([
                        "Observation.vitals", "Condition.cardiovascular",
                        "Medication.active", "Procedure.surgical"
                    ])
                    resources.append(next_resource)
                pipelines.append({"resource_chain": resources})
        
        return pipelines
    
    def run(
        self,
        objective: Callable[[Dict[str, Any]], float],
        n_iterations: int = 20,
        candidates_per_iter: Optional[List[Dict[str, Any]]] = None,
        verbose: bool = True,
    ) -> OptimizationResult:
        """Run complete optimization loop.
        
        Args:
            objective: Function that evaluates a pipeline and returns score
            n_iterations: Number of optimization iterations
            candidates_per_iter: Optional fixed candidate pool
            verbose: Print progress
            
        Returns:
            OptimizationResult with best pipeline and history
        """
        start_time = time.time()
        
        for i in range(n_iterations):
            # Suggest
            suggestions = self.suggest(
                n_suggestions=self.n_parallel,
                candidates=candidates_per_iter,
            )
            
            # Evaluate
            scores = [objective(p) for p in suggestions]
            
            # Observe
            self.observe(suggestions, scores)
            
            if verbose:
                best_p, best_s = self.best()
                print(f"Iter {i+1}/{n_iterations}: batch_best={max(scores):.4f}, overall_best={best_s:.4f}")
        
        total_time = time.time() - start_time
        best_pipeline, best_score = self.best()
        best_embedding = self._encoder.encode(best_pipeline)
        
        return OptimizationResult(
            best_pipeline=best_pipeline,
            best_score=best_score,
            best_embedding=best_embedding,
            n_iterations=n_iterations,
            history=self._history,
            total_time=total_time,
        )
    
    @property
    def n_observations(self) -> int:
        """Number of observations so far."""
        return self._gp.n_observations
    
    @property
    def history(self) -> List[Dict[str, Any]]:
        """Full optimization history."""
        return self._history
