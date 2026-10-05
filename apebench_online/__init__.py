"""APEBench-Online: Online Active Learning for PDE Surrogate Models.

This package provides utilities for training neural network surrogates
for Partial Differential Equations (PDEs) using online active learning
with the Melissa in situ processing framework.

Subpackages:
    core: Server implementations, solver interface, and training utilities
    models: PyTorch model wrappers (FNO, UNet, ScOT)
    metrics: Monitoring and plotting utilities
    sampler: Initial condition samplers with various distributions
    scenarios: PDE scenario configurations and registry
    validation: External validation framework
    vendor: Third-party model implementations (AL4PDE, PDEArena, ScOT)
    sbal: Pool based active sampling strategies
"""

__version__ = "0.1.0"
__author__ = "DATAMOVE Team, Inria"
__license__ = "MIT"

__all__ = [
    "core",
    "metrics",
    "models",
    "sampler",
    "scenarios",
    "sbal",
    "validation",
    "vendor",
]
