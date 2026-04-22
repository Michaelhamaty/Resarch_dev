"""Public exports for the inference package (adapter interface + types)."""

from .adapter import InferenceAdapter
from .factory import build_adapter
from .stub import StubInferenceAdapter
from .types import InferenceResult

__all__ = [
    "InferenceAdapter",
    "InferenceResult",
    "StubInferenceAdapter",
    "build_adapter",
]
