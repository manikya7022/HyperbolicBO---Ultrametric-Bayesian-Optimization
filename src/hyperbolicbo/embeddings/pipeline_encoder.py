"""Pipeline encoder: maps pipeline JSON to Poincaré ball embeddings.

Supports three pipeline types:
1. NAS (cell-based): DAGs with operation nodes
2. AutoML (sequential): Linear stage pipelines  
3. FHIR (ontology-aware): Healthcare resource chains
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Literal
import hashlib
import json

import torch
from torch import Tensor
import networkx as nx
import numpy as np

from ..geometry.poincare import project_to_ball, adaptive_dimension


# Pipeline type literals
PipelineType = Literal["nas", "automl", "fhir"]


@dataclass
class PipelineNode:
    """A node in a pipeline graph."""
    id: str
    type: str  # Operation or stage type
    params: Dict[str, Any]  # Hyperparameters
    inputs: List[str]  # Input node IDs


def pipeline_to_graph(pipeline: Dict[str, Any], pipeline_type: PipelineType) -> nx.DiGraph:
    """Convert pipeline JSON to NetworkX graph.
    
    Args:
        pipeline: Pipeline configuration dict
        pipeline_type: One of "nas", "automl", "fhir"
        
    Returns:
        NetworkX directed graph
    """
    G = nx.DiGraph()
    
    if pipeline_type == "nas":
        # NAS: cell-based with operations
        # Expected format: {"cells": [{"op": "conv_3x3", "input": [0, 1]}, ...]}
        cells = pipeline.get("cells", pipeline.get("normal_cell", []))
        
        for i, cell in enumerate(cells):
            node_id = f"cell_{i}"
            G.add_node(node_id, op=cell.get("op", "identity"), params=cell)
            
            # Add edges from inputs
            for inp in cell.get("input", cell.get("inputs", [])):
                if isinstance(inp, int):
                    inp_id = f"cell_{inp}" if inp < i else f"input_{inp}"
                else:
                    inp_id = str(inp)
                
                if not G.has_node(inp_id):
                    G.add_node(inp_id, op="input", params={})
                G.add_edge(inp_id, node_id)
    
    elif pipeline_type == "automl":
        # AutoML: sequential stages
        # Expected format: {"stages": [{"type": "scaler", "method": "StandardScaler"}, ...]}
        stages = pipeline.get("stages", [])
        
        prev_id = None
        for i, stage in enumerate(stages):
            node_id = f"stage_{i}"
            G.add_node(node_id, op=stage.get("type", "unknown"), params=stage)
            
            if prev_id is not None:
                G.add_edge(prev_id, node_id)
            prev_id = node_id
    
    elif pipeline_type == "fhir":
        # FHIR: resource chain
        # Expected format: {"resource_chain": ["Patient", "Observation.vitals", ...]}
        chain = pipeline.get("resource_chain", [])
        features = pipeline.get("features", {})
        
        prev_id = None
        for i, resource in enumerate(chain):
            node_id = f"resource_{i}"
            G.add_node(
                node_id, 
                op=resource.split(".")[0], 
                params={"resource": resource, "features": features.get(resource, {})}
            )
            
            if prev_id is not None:
                G.add_edge(prev_id, node_id)
            prev_id = node_id
    
    return G


class PipelineEncoder:
    """Encodes pipeline configurations to Poincaré ball embeddings.
    
    Uses structure-aware encoding where:
    - Depth in pipeline → distance from origin
    - Branching factor → angular spread
    - Operation similarity → nearby embeddings
    """
    
    def __init__(
        self,
        dim: int = 2,
        pipeline_type: PipelineType = "automl",
        pretrained_ops: Optional[Dict[str, Tensor]] = None,
    ):
        """Initialize encoder.
        
        Args:
            dim: Embedding dimension
            pipeline_type: Type of pipelines to encode
            pretrained_ops: Optional pretrained operation embeddings
        """
        self.dim = dim
        self.pipeline_type = pipeline_type
        self.pretrained_ops = pretrained_ops or {}
        
        # Default operation embeddings (will be learned or can be pretrained)
        self._init_default_ops()
    
    def _init_default_ops(self):
        """Initialize default operation embeddings."""
        # Common operations for each pipeline type
        if self.pipeline_type == "nas":
            ops = [
                "conv_1x1", "conv_3x3", "conv_5x5",
                "dil_conv_3x3", "dil_conv_5x5",
                "max_pool", "avg_pool",
                "skip_connect", "none", "identity"
            ]
        elif self.pipeline_type == "automl":
            ops = [
                "scaler", "normalizer", "imputer",
                "pca", "selector", "encoder",
                "xgb", "rf", "lr", "svm", "mlp",
                "calibrator", "ensemble"
            ]
        elif self.pipeline_type == "fhir":
            ops = [
                "Patient", "Practitioner", "Observation",
                "Condition", "Procedure", "Medication",
                "Encounter", "DiagnosticReport"
            ]
        else:
            ops = []
        
        # Create evenly spaced embeddings on a circle (for 2D)
        for i, op in enumerate(ops):
            if op not in self.pretrained_ops:
                angle = 2 * np.pi * i / max(len(ops), 1)
                radius = 0.3  # Stay away from boundary
                
                if self.dim == 2:
                    emb = torch.tensor([radius * np.cos(angle), radius * np.sin(angle)])
                else:
                    # Higher dim: use random direction
                    emb = torch.randn(self.dim)
                    emb = emb / emb.norm() * radius
                
                self.pretrained_ops[op] = emb
    
    def _get_op_embedding(self, op: str) -> Tensor:
        """Get embedding for an operation, creating if needed."""
        if op in self.pretrained_ops:
            return self.pretrained_ops[op].clone()
        
        # Hash-based embedding for unknown ops
        hash_val = int(hashlib.md5(op.encode()).hexdigest()[:8], 16)
        torch.manual_seed(hash_val)
        emb = torch.randn(self.dim) * 0.3
        self.pretrained_ops[op] = emb
        
        return emb
    
    def encode(self, pipeline: Dict[str, Any]) -> Tensor:
        """Encode a single pipeline to Poincaré ball.
        
        Args:
            pipeline: Pipeline configuration dict
            
        Returns:
            Embedding tensor of shape (dim,)
        """
        G = pipeline_to_graph(pipeline, self.pipeline_type)
        
        if len(G.nodes) == 0:
            return torch.zeros(self.dim)
        
        # Compute node embeddings based on structure
        node_embeddings = {}
        
        # Topological order for depth calculation
        try:
            topo_order = list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            # Cyclic graph: use arbitrary order
            topo_order = list(G.nodes)
        
        # Depth: distance from source nodes
        depths = {}
        for node in topo_order:
            preds = list(G.predecessors(node))
            if not preds:
                depths[node] = 0
            else:
                depths[node] = max(depths.get(p, 0) for p in preds) + 1
        
        max_depth = max(depths.values()) if depths else 1
        
        # Compute embeddings
        for node in G.nodes:
            op = G.nodes[node].get("op", "unknown")
            base_emb = self._get_op_embedding(op)
            
            # Scale by depth: deeper nodes are closer to boundary
            depth_scale = 0.3 + 0.6 * (depths.get(node, 0) / max(max_depth, 1))
            
            # Add positional offset based on node index
            idx = topo_order.index(node) if node in topo_order else 0
            angle_offset = 0.1 * idx  # Small angular offset
            
            if self.dim >= 2:
                rotation = torch.tensor([
                    [np.cos(angle_offset), -np.sin(angle_offset)],
                    [np.sin(angle_offset), np.cos(angle_offset)]
                ])
                base_emb[:2] = rotation @ base_emb[:2]
            
            # Scale to appropriate depth
            norm = base_emb.norm().clamp(min=1e-6)
            node_embeddings[node] = base_emb / norm * depth_scale
        
        # Aggregate: weighted mean by node importance (out-degree)
        weights = torch.tensor([
            1.0 + G.out_degree(node) for node in G.nodes
        ])
        weights = weights / weights.sum()
        
        stacked = torch.stack([node_embeddings[n] for n in G.nodes])
        pipeline_emb = (weights.unsqueeze(-1) * stacked).sum(dim=0)
        
        return project_to_ball(pipeline_emb)
    
    def encode_batch(self, pipelines: List[Dict[str, Any]]) -> Tensor:
        """Encode multiple pipelines.
        
        Args:
            pipelines: List of pipeline configurations
            
        Returns:
            Embeddings tensor of shape (n, dim)
        """
        embeddings = [self.encode(p) for p in pipelines]
        return torch.stack(embeddings)
    
    def fit(self, pipelines: List[Dict[str, Any]], **kwargs):
        """Optionally train encoder on pipeline corpus.
        
        For now, uses fixed embeddings. Can be extended with Node2Vec training.
        """
        pass  # Placeholder for future Node2Vec integration


def fhir_hyperbolic_distance(
    pipeline1: Dict[str, Any],
    pipeline2: Dict[str, Any],
    encoder: PipelineEncoder,
    ontology_depth: float = 5.0,
) -> Tensor:
    """Custom hyperbolic distance for FHIR pipelines.
    
    Modifies base distance by ontology similarity.
    
    base_dist × exp(-ontology_depth × common_prefix_len)
    
    Args:
        pipeline1: First FHIR pipeline
        pipeline2: Second FHIR pipeline
        encoder: PipelineEncoder instance
        ontology_depth: Weight for ontology modifier
        
    Returns:
        Modified hyperbolic distance
    """
    from ..geometry.poincare import poincare_distance
    
    emb1 = encoder.encode(pipeline1)
    emb2 = encoder.encode(pipeline2)
    
    base_dist = poincare_distance(emb1.unsqueeze(0), emb2.unsqueeze(0)).squeeze()
    
    # Compute longest common prefix in resource chains
    chain1 = pipeline1.get("resource_chain", [])
    chain2 = pipeline2.get("resource_chain", [])
    
    common_len = 0
    for r1, r2 in zip(chain1, chain2):
        if r1.split(".")[0] == r2.split(".")[0]:
            common_len += 1
        else:
            break
    
    # Modifier: penalize early divergence
    modifier = np.exp(-ontology_depth * common_len)
    
    return base_dist * modifier
