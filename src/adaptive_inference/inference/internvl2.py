"""Real InternVL2 adapter — loads the HF model and runs per-page inference.

This adapter exists so the Phase 2/4 runner can produce *actual* model
output instead of stub HTML. It is intentionally written so the test
suite does not require torch / transformers to be installed: heavy
imports happen inside ``__init__`` (when a real adapter is constructed),
not at module import time. CI and the existing local dev loop stay
fast and dependency-light; only the GPU host needs the deep-learning
stack.

Wiring contract:
- ``Budget.max_tiles`` is mapped 1:1 to InternVL2's dynamic-tiling
  ``max_num`` parameter. That is the single compute knob the project
  cares about — the whole matched-budget thesis hinges on this number
  being honored exactly as configured.
- One call per page. No batching across pages (the runner is
  single-page; batching is a later optimization).
- ``InferenceResult.output_token_count`` reports the *generated* tokens
  only (input/visual tokens are deliberately excluded — the project's
  cost accounting separately tracks tile counts as the budget signal).

This adapter is deterministic given a fixed seed and ``do_sample=False``;
we use greedy decoding to match Phase 2's reproducibility contract.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from PIL import Image

from ..config.budgets import Budget
from ..config.prompts import PromptTemplate
from .adapter import InferenceAdapter
from .types import InferenceResult


# InternVL2's reference preprocessing constants. Pulled from the model
# card's `load_image` example. Kept here (not in a config) because they
# are model-architecture properties, not knobs we ever want to tune.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_INPUT_SIZE = 448
_MIN_TILES = 1


class InternVL2Adapter(InferenceAdapter):
    """Loads an InternVL2 model from HuggingFace and runs per-page inference.

    Parameters
    ----------
    model_name : str
        The logical model name from the registry (e.g. ``"internvl2-2b"``).
        Used only for ``InferenceResult`` provenance, not for loading.
    model_id : str
        The HuggingFace repo id (e.g. ``"OpenGVLab/InternVL2-2B"``).
    device : str | None
        ``"cuda"``, ``"mps"``, or ``"cpu"``. ``None`` auto-detects with
        CUDA preferred, then MPS, then CPU.
    dtype : str
        Torch dtype name (``"bfloat16"`` recommended for CUDA;
        ``"float16"`` is the safe MPS fallback).
    max_new_tokens : int
        Generation cap. The default is generous; tune in the model
        registry yaml if cost tracking requires it.
    """

    def __init__(
        self,
        *,
        model_name: str,
        model_id: str,
        device: str | None = None,
        dtype: str | None = None,
        max_new_tokens: int = 2048,
    ) -> None:
        # Heavy imports kept lazy so importing this module on a machine
        # without torch installed (e.g. during local unit tests) works.
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

        self._torch = torch
        self.device = device or _auto_device(torch)
        # Auto-pick dtype: bf16 only on capable hardware (Ampere+, sm_80+
        # for CUDA; Apple MPS supports fp16 most reliably). Without this
        # check, T4/V100 GPUs silently corrupt or OOM on bf16 weights.
        self.dtype = _resolve_dtype(torch, dtype or _auto_dtype(torch, self.device))

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False,
        )
        self.model = (
            AutoModel.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                trust_remote_code=True,
                attn_implementation="sdpa",
            )
            .to(self.device)
            .eval()
        )

    def run(
        self,
        page_id: str,
        image: Image.Image,
        budget: Budget,
        prompt: PromptTemplate,
    ) -> InferenceResult:
        started_at = _iso_now()
        t0 = time.perf_counter()

        pixel_values = _preprocess_image(
            image=image,
            max_num=budget.max_tiles,
            torch=self._torch,
            device=self.device,
            dtype=self.dtype,
        )

        # InternVL2's chat() helper is the documented interface. We pass
        # do_sample=False for greedy/deterministic decoding.
        # The question MUST contain an "<image>" token — InternVL2 uses it
        # to splice visual embeddings into the text stream. Project prompts
        # are intentionally model-agnostic, so we prepend it here.
        question = _ensure_image_token(prompt.template)
        generation_config: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
        }
        with self._torch.inference_mode():
            response = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=generation_config,
                history=None,
                return_history=False,
            )

        # Token count reported is the generated text re-tokenized; it
        # avoids depending on internals of model.chat that may not return
        # token ids.
        output_token_count = len(self.tokenizer.encode(response, add_special_tokens=False))

        runtime_ms = (time.perf_counter() - t0) * 1000.0
        finished_at = _iso_now()

        return InferenceResult(
            page_id=page_id,
            model_name=self.model_name,
            budget_name=budget.name,
            prompt_id=prompt.id,
            raw_text=response,
            output_token_count=output_token_count,
            runtime_ms=runtime_ms,
            started_at=started_at,
            finished_at=finished_at,
        )


# --------------------------------------------------------------------------- #
# Image preprocessing — adapted from the InternVL2 model card.                #
# --------------------------------------------------------------------------- #


def _preprocess_image(
    *,
    image: Image.Image,
    max_num: int,
    torch: Any,
    device: str,
    dtype: Any,
) -> Any:
    """Tile + normalize an image into the model's expected pixel_values tensor.

    Mirrors the reference dynamic-tiling preprocessor: choose the best
    aspect-ratio split given ``max_num`` tiles, resize, crop into 448-px
    square tiles, then ImageNet-normalize and stack.
    """

    from torchvision.transforms import functional as TF  # lazy import

    image = image.convert("RGB")
    tiles = _dynamic_tile_split(image, max_num=max_num)

    tensors = []
    for tile in tiles:
        t = TF.to_tensor(tile)  # [3, H, W] in [0,1]
        t = TF.normalize(t, mean=list(_IMAGENET_MEAN), std=list(_IMAGENET_STD))
        tensors.append(t)
    pixel_values = torch.stack(tensors, dim=0).to(dtype=dtype, device=device)
    return pixel_values


def _dynamic_tile_split(image: Image.Image, *, max_num: int) -> list[Image.Image]:
    """Split image into up to ``max_num`` 448x448 tiles by best aspect ratio.

    Reference behavior from OpenGVLab/InternVL2 model card. Returns at
    least one tile, never more than ``max_num``.
    """

    width, height = image.size
    aspect = width / height

    # Enumerate all (cols, rows) splits up to max_num tiles.
    candidates: list[tuple[int, int]] = []
    for n in range(_MIN_TILES, max_num + 1):
        for cols in range(1, n + 1):
            if n % cols != 0:
                continue
            rows = n // cols
            candidates.append((cols, rows))
    # Pick the (cols, rows) whose ratio is closest to the image aspect,
    # breaking ties toward smaller tile counts.
    candidates.sort(key=lambda cr: (abs((cr[0] / cr[1]) - aspect), cr[0] * cr[1]))
    cols, rows = candidates[0]

    target_w = cols * _INPUT_SIZE
    target_h = rows * _INPUT_SIZE
    resized = image.resize((target_w, target_h))

    tiles: list[Image.Image] = []
    for r in range(rows):
        for c in range(cols):
            box = (
                c * _INPUT_SIZE,
                r * _INPUT_SIZE,
                (c + 1) * _INPUT_SIZE,
                (r + 1) * _INPUT_SIZE,
            )
            tiles.append(resized.crop(box))
    return tiles


# --------------------------------------------------------------------------- #
# Small helpers                                                               #
# --------------------------------------------------------------------------- #


def _auto_device(torch: Any) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _auto_dtype(torch: Any, device: str) -> str:
    """Pick the safest dtype for the chosen device.

    bf16 is only safe on Ampere+ CUDA (sm_80+) and modern CPUs. On older
    GPUs (T4 sm_75, V100 sm_70) bf16 silently underperforms or OOMs. MPS
    supports fp16 reliably; bf16 is patchy. CPU stays fp32 to avoid
    accidentally running a 2B model in slow-emulated bf16.
    """

    if device == "cuda":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return "bfloat16"
        return "float16"
    if device == "mps":
        return "float16"
    return "float32"


def _ensure_image_token(text: str) -> str:
    """Guarantee the question passed to ``model.chat`` contains ``<image>``."""

    return text if "<image>" in text else f"<image>\n{text}"


def _resolve_dtype(torch: Any, name: str) -> Any:
    name = name.lower()
    table = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in table:
        raise ValueError(f"Unknown dtype {name!r}; expected one of {sorted(table)}.")
    return table[name]


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()
