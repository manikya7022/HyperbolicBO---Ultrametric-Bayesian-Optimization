"""HyperbolicBO: Ultrametric Bayesian Optimization using Poincaré ball geometry."""

from hyperbolicbo.optimizer.hbo import HyperbolicBO
from hyperbolicbo.geometry.poincare import poincare_distance, mobius_add, exp_map, log_map
from hyperbolicbo.gp.hyperbolic_gp import HyperbolicGP
from hyperbolicbo.acquisition.ei import hyperbolic_ei
from hyperbolicbo.acquisition.thompson import parallel_hyperbolic_ts

__version__ = "0.1.0"
__all__ = [
    "HyperbolicBO",
    "HyperbolicGP",
    "poincare_distance",
    "mobius_add",
    "exp_map",
    "log_map",
    "hyperbolic_ei",
    "parallel_hyperbolic_ts",
]
