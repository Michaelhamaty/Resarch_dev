"""Abstract adapter boundary between the runner and a concrete model.

Any real model integration — InternVL2 via HF transformers, a remote
server, etc. — subclasses ``InferenceAdapter`` and implements ``run``.
The runner only talks to this interface, which is how Phase 2's stub
adapter and a later phase's real adapter can coexist without parallel
plumbing.
"""

from __future__ import annotations

import abc

from PIL import Image

from ..config.budgets import Budget
from ..config.prompts import PromptTemplate
from .types import InferenceResult


class InferenceAdapter(abc.ABC):
    """One-call-per-page model adapter."""

    model_name: str

    @abc.abstractmethod
    def run(
        self,
        page_id: str,
        image: Image.Image,
        budget: Budget,
        prompt: PromptTemplate,
    ) -> InferenceResult:
        """Run inference on one page and return a structured result."""
