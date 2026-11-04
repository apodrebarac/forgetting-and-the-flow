"""The Forgetting and the Flow — degradation as method, in NumPy.

Two autonomous real-time systems that degrade, dissolve and forget a finite
image archive, continuously, without repeating themselves:

* :mod:`forgetting_and_flow.flow`        — forgetting as lossy recursion
* :mod:`forgetting_and_flow.forgetting`  — forgetting as irreversible state
* :mod:`forgetting_and_flow.terminal`    — the archive consuming itself
"""
from .flow import Flow, FlowParams, contribution, frames_to_replace
from .forgetting import Forgetting, ForgettingParams, ForgivenessWithheld, MemoryMask
from .runner import Run, RunConfig
from .terminal import Consumer, ConsumerParams

__all__ = [
    "Flow", "FlowParams", "contribution", "frames_to_replace",
    "Forgetting", "ForgettingParams", "ForgivenessWithheld", "MemoryMask",
    "Run", "RunConfig", "Consumer", "ConsumerParams",
]
__version__ = "0.1.0"
