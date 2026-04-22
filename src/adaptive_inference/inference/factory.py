"""Factory that maps a ``ModelConfig`` to an ``InferenceAdapter`` instance.

Keeping the dispatch table here (instead of in the runner) means the
runner never imports any concrete adapter implementation. A new adapter
kind (e.g. a real InternVL2 loader) is added by registering it here; no
caller changes.
"""

from __future__ import annotations

from ..config.models import ModelConfig
from .adapter import InferenceAdapter
from .stub import StubInferenceAdapter


def build_adapter(model_cfg: ModelConfig) -> InferenceAdapter:
    """Instantiate the adapter named by ``model_cfg.adapter_kind``."""

    kind = model_cfg.adapter_kind
    if kind == "stub":
        return StubInferenceAdapter(model_name=model_cfg.name)
    raise ValueError(
        f"Unknown adapter_kind={kind!r} for model {model_cfg.name!r}. "
        f"Phase 2 supports: 'stub'."
    )
