"""NAS-Bench-201 integration for HyperbolicBO.

Provides an adapter for the NAS-Bench-201 benchmark, enabling
true architecture search comparison.

NAS-Bench-201 search space:
- 6 nodes per cell
- 5 operation options: none, skip_connect, conv_1x1, conv_3x3, avg_pool_3x3
- Total: 15,625 unique architectures with precomputed accuracies

Reference: Dong & Yang, "NAS-Bench-201: Extending the Scope of 
Reproducible Neural Architecture Search", ICLR 2020
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import itertools
import random
import hashlib

import numpy as np
import torch
from torch import Tensor


# NAS-Bench-201 operation space
OPERATIONS = ["none", "skip_connect", "conv_1x1", "conv_3x3", "avg_pool_3x3"]
NUM_OPS = len(OPERATIONS)
NUM_EDGES = 6  # (0,1), (0,2), (1,2), (0,3), (1,3), (2,3)

# Edge indices for 4-node cell
EDGE_INDICES = [
    (0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)
]


def arch_str_to_ops(arch_str: str) -> List[str]:
    """Parse NAS-Bench-201 architecture string to operation list.
    
    Format: |op~0|+|op~0|op~1|+|op~0|op~1|op~2|
    """
    ops = []
    for part in arch_str.split('+'):
        for op_part in part.split('|'):
            if '~' in op_part:
                op = op_part.split('~')[0]
                ops.append(op)
    return ops


def ops_to_arch_str(ops: List[str]) -> str:
    """Convert operation list to NAS-Bench-201 architecture string."""
    return f"|{ops[0]}~0|+|{ops[1]}~0|{ops[2]}~1|+|{ops[3]}~0|{ops[4]}~1|{ops[5]}~2|"


def arch_to_index(ops: List[str]) -> int:
    """Convert operations to unique index."""
    idx = 0
    for i, op in enumerate(ops):
        idx += OPERATIONS.index(op) * (NUM_OPS ** i)
    return idx


def index_to_arch(idx: int) -> List[str]:
    """Convert index to operations."""
    ops = []
    for _ in range(NUM_EDGES):
        ops.append(OPERATIONS[idx % NUM_OPS])
        idx //= NUM_OPS
    return ops


@dataclass
class NASBench201Result:
    """Result from NAS-Bench-201 query."""
    arch_str: str
    train_acc: float  # Training accuracy
    valid_acc: float  # Validation accuracy
    test_acc: float   # Test accuracy
    train_time: float  # Training time in seconds
    params: float     # Number of parameters
    flops: float      # FLOPs


class NASBench201:
    """NAS-Bench-201 benchmark adapter.
    
    Can work in two modes:
    1. With real benchmark data (requires downloaded files)
    2. With synthetic data (for testing without downloads)
    """
    
    TOTAL_ARCHS = NUM_OPS ** NUM_EDGES  # 15,625 architectures
    
    def __init__(
        self,
        dataset: str = "cifar10",
        data_path: Optional[str] = None,
        use_synthetic: bool = True,
    ):
        """Initialize NAS-Bench-201 adapter.
        
        Args:
            dataset: One of "cifar10", "cifar100", "ImageNet16-120"
            data_path: Path to NAS-Bench-201 data file
            use_synthetic: If True, use synthetic accuracies for testing
        """
        self.dataset = dataset
        self.data_path = data_path
        self.use_synthetic = use_synthetic
        
        self._api = None
        self._synthetic_cache = {}
        
        if not use_synthetic and data_path:
            self._load_api()
        elif not use_synthetic:
            print("Warning: No data_path provided, falling back to synthetic mode")
            self.use_synthetic = True
        
        if self.use_synthetic:
            self._init_synthetic()
    
    def _load_api(self):
        """Load real NAS-Bench-201 API."""
        try:
            from nas_201_api import NASBench201API
            self._api = NASBench201API(self.data_path, verbose=False)
            print(f"Loaded NAS-Bench-201 with {len(self._api)} architectures")
        except ImportError:
            print("NAS-Bench-201 API not installed. Install with: pip install nas-bench-201")
            print("Falling back to synthetic mode")
            self.use_synthetic = True
            self._init_synthetic()
        except Exception as e:
            print(f"Failed to load NAS-Bench-201: {e}")
            print("Falling back to synthetic mode")
            self.use_synthetic = True
            self._init_synthetic()
    
    def _init_synthetic(self):
        """Initialize synthetic benchmark data.
        
        Creates realistic accuracy distributions based on operation quality.
        Key insight: architectures with more conv_3x3 tend to be better,
        while those with many 'none' operations are worse.
        """
        print("Using synthetic NAS-Bench-201 data for testing")
        np.random.seed(42)
        
        # Operation quality scores (higher = better)
        self.op_scores = {
            "none": 0.0,
            "skip_connect": 0.6,
            "avg_pool_3x3": 0.5,
            "conv_1x1": 0.7,
            "conv_3x3": 1.0,
        }
        
        # Pre-generate some high-quality architectures
        self._best_archs = []
        for _ in range(50):
            # Bias toward better operations
            ops = []
            for _ in range(NUM_EDGES):
                weights = [self.op_scores[op] + 0.1 for op in OPERATIONS]
                weights = np.array(weights) / sum(weights)
                ops.append(np.random.choice(OPERATIONS, p=weights))
            self._best_archs.append(tuple(ops))
    
    def _compute_synthetic_accuracy(self, ops: List[str]) -> Tuple[float, float, float]:
        """Compute synthetic accuracy based on operation scores."""
        # Base score from operation quality
        base_score = sum(self.op_scores[op] for op in ops) / NUM_EDGES
        
        # Bonus for architectural patterns
        n_conv3 = sum(1 for op in ops if op == "conv_3x3")
        n_skip = sum(1 for op in ops if op == "skip_connect")
        n_none = sum(1 for op in ops if op == "none")
        
        # Good patterns: conv3x3 + skip connections (like ResNet)
        pattern_bonus = 0.05 * min(n_conv3, 3) + 0.03 * min(n_skip, 2)
        
        # Penalty for too many 'none' (disconnected)
        none_penalty = 0.1 * max(0, n_none - 1)
        
        # Final scores with noise
        valid_acc = min(0.95, max(0.1, base_score + pattern_bonus - none_penalty))
        valid_acc += np.random.normal(0, 0.01)
        valid_acc = np.clip(valid_acc, 0.1, 0.95)
        
        # Test acc slightly lower, train acc higher
        test_acc = valid_acc - np.random.uniform(0.01, 0.03)
        train_acc = valid_acc + np.random.uniform(0.02, 0.05)
        
        return train_acc, valid_acc, test_acc
    
    def query(self, arch: Dict[str, Any]) -> NASBench201Result:
        """Query benchmark for architecture performance.
        
        Args:
            arch: Architecture in HyperbolicBO format:
                  {"cells": [{"op": "conv_3x3", "input": [...]}]}
                  
        Returns:
            NASBench201Result with accuracies
        """
        # Convert to NAS-Bench-201 format
        ops = self._arch_to_ops(arch)
        arch_str = ops_to_arch_str(ops)
        
        if self.use_synthetic:
            return self._query_synthetic(ops, arch_str)
        else:
            return self._query_real(arch_str)
    
    def _arch_to_ops(self, arch: Dict[str, Any]) -> List[str]:
        """Convert HyperbolicBO architecture to operation list."""
        cells = arch.get("cells", [])
        
        if not cells:
            # Random architecture
            return [random.choice(OPERATIONS) for _ in range(NUM_EDGES)]
        
        ops = []
        for cell in cells[:NUM_EDGES]:
            op = cell.get("op", "conv_3x3")
            # Map to valid operations
            if op not in OPERATIONS:
                op = "conv_3x3" if "conv" in op else "skip_connect"
            ops.append(op)
        
        # Pad if needed
        while len(ops) < NUM_EDGES:
            ops.append("conv_3x3")
        
        return ops[:NUM_EDGES]
    
    def _query_synthetic(self, ops: List[str], arch_str: str) -> NASBench201Result:
        """Query synthetic benchmark."""
        # Cache for determinism
        cache_key = arch_str
        if cache_key in self._synthetic_cache:
            return self._synthetic_cache[cache_key]
        
        train_acc, valid_acc, test_acc = self._compute_synthetic_accuracy(ops)
        
        # Synthetic training time (correlated with conv ops)
        n_convs = sum(1 for op in ops if "conv" in op)
        train_time = 10.0 + 5.0 * n_convs + np.random.uniform(0, 2)
        
        # Parameters and FLOPs
        params = 0.1 + 0.2 * n_convs  # Millions
        flops = 10 + 50 * n_convs     # Millions
        
        result = NASBench201Result(
            arch_str=arch_str,
            train_acc=train_acc,
            valid_acc=valid_acc,
            test_acc=test_acc,
            train_time=train_time,
            params=params,
            flops=flops,
        )
        
        self._synthetic_cache[cache_key] = result
        return result
    
    def _query_real(self, arch_str: str) -> NASBench201Result:
        """Query real NAS-Bench-201 API."""
        idx = self._api.query_index_by_arch(arch_str)
        
        # Get metrics for specified dataset
        info = self._api.query_by_index(idx, self.dataset)
        
        return NASBench201Result(
            arch_str=arch_str,
            train_acc=info.get_train()['accuracy'] / 100,
            valid_acc=info.get_valid()['accuracy'] / 100,
            test_acc=info.get_test()['accuracy'] / 100,
            train_time=info.get_train()['time'],
            params=info.get_detail()['params'],
            flops=info.get_detail()['flops'],
        )
    
    def random_architecture(self) -> Dict[str, Any]:
        """Generate random architecture in HyperbolicBO format."""
        ops = [random.choice(OPERATIONS) for _ in range(NUM_EDGES)]
        return self._ops_to_arch(ops)
    
    def _ops_to_arch(self, ops: List[str]) -> Dict[str, Any]:
        """Convert operation list to HyperbolicBO architecture format."""
        cells = []
        for i, op in enumerate(ops):
            edge = EDGE_INDICES[i]
            cells.append({
                "op": op,
                "input": [edge[0]],  # Input from first node of edge
            })
        return {"cells": cells}
    
    def get_optimal_architecture(self) -> Dict[str, Any]:
        """Get the best architecture (for evaluation)."""
        if self.use_synthetic:
            # Best synthetic is usually all conv_3x3 with some skip
            best_ops = ["conv_3x3"] * 4 + ["skip_connect"] * 2
            return self._ops_to_arch(best_ops)
        else:
            # Query API for best
            best_idx = 0
            best_acc = 0
            for i in range(len(self._api)):
                arch_str = self._api.arch(i)
                info = self._api.query_by_index(i, self.dataset)
                acc = info.get_valid()['accuracy']
                if acc > best_acc:
                    best_acc = acc
                    best_idx = i
            
            arch_str = self._api.arch(best_idx)
            ops = arch_str_to_ops(arch_str)
            return self._ops_to_arch(ops)
    
    def __len__(self) -> int:
        """Number of architectures in search space."""
        return self.TOTAL_ARCHS


class NASBench201Objective:
    """Objective function wrapper for HyperbolicBO."""
    
    def __init__(
        self, 
        benchmark: NASBench201,
        metric: str = "valid_acc",
    ):
        """Initialize objective.
        
        Args:
            benchmark: NASBench201 instance
            metric: Which metric to optimize ("valid_acc", "test_acc", etc.)
        """
        self.benchmark = benchmark
        self.metric = metric
        self.query_count = 0
    
    def __call__(self, arch: Dict[str, Any]) -> float:
        """Evaluate architecture."""
        self.query_count += 1
        result = self.benchmark.query(arch)
        return getattr(result, self.metric)
    
    def reset(self):
        """Reset query counter."""
        self.query_count = 0
