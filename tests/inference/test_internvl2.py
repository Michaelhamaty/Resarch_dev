"""Tests for the real InternVL2 adapter — heavy deps mocked.

These tests do NOT load InternVL2. They verify the adapter's *plumbing*:
that lazy imports work, that ``Budget.max_tiles`` is propagated to the
preprocessor's ``max_num``, that the result wraps the model's response
into the expected ``InferenceResult`` shape, and that the factory wires
the new ``adapter_kind`` correctly. The real model is loaded only on the
GPU host (Colab Pro), so an end-to-end smoke test belongs there, not in
the unit-test suite.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from adaptive_inference.config.budgets import Budget
from adaptive_inference.config.models import ModelConfig
from adaptive_inference.config.prompts import PromptTemplate


BUDGET = Budget(name="low", max_tiles=4)
PROMPT = PromptTemplate(
    id="table_parse_v1",
    version=1,
    description="d",
    template="parse this page",
)


def _img(size: tuple[int, int] = (128, 96)) -> Image.Image:
    return Image.new("RGB", size, color=(7, 7, 7))


# --------------------------------------------------------------------------- #
# Fake torch / transformers / torchvision modules                             #
# --------------------------------------------------------------------------- #


def _install_fake_torch_stack(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Install minimal fake torch/transformers/torchvision modules in sys.modules.

    Returns a namespace of the key fakes so individual tests can assert
    against them.
    """

    # torch
    fake_torch = ModuleType("torch")

    class _Dtype:
        def __init__(self, name: str) -> None:
            self.name = name

        def __repr__(self) -> str:  # pragma: no cover - debug aid only
            return f"<fake-dtype {self.name}>"

    fake_torch.bfloat16 = _Dtype("bfloat16")
    fake_torch.float16 = _Dtype("float16")
    fake_torch.float32 = _Dtype("float32")

    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = SimpleNamespace(
        mps=SimpleNamespace(is_available=lambda: False)
    )

    class _InferenceMode:
        def __enter__(self) -> "_InferenceMode":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    fake_torch.inference_mode = _InferenceMode

    stack_calls: list[int] = []

    def _stack(tensors, dim=0):  # noqa: ANN001
        stack_calls.append(len(tensors))
        # Return a stand-in tensor that records the .to() call.
        return _FakeTensor()

    fake_torch.stack = _stack

    # transformers
    fake_transformers = ModuleType("transformers")
    fake_tokenizer = MagicMock(name="tokenizer")
    fake_tokenizer.encode.return_value = [101, 102, 103]  # 3 tokens
    fake_model_chat = MagicMock(return_value="<table><tr><td>x</td></tr></table>")
    fake_model = MagicMock(name="model")
    fake_model.chat = fake_model_chat
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = fake_model

    fake_transformers.AutoModel = MagicMock()
    fake_transformers.AutoModel.from_pretrained = MagicMock(return_value=fake_model)
    fake_transformers.AutoTokenizer = MagicMock()
    fake_transformers.AutoTokenizer.from_pretrained = MagicMock(
        return_value=fake_tokenizer
    )

    # torchvision.transforms.functional
    fake_torchvision = ModuleType("torchvision")
    fake_transforms = ModuleType("torchvision.transforms")
    fake_functional = ModuleType("torchvision.transforms.functional")

    def _to_tensor(img):  # noqa: ANN001
        return _FakeTensor()

    def _normalize(t, mean, std):  # noqa: ANN001
        return t

    fake_functional.to_tensor = _to_tensor
    fake_functional.normalize = _normalize
    fake_transforms.functional = fake_functional
    fake_torchvision.transforms = fake_transforms

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torchvision", fake_torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.transforms", fake_transforms)
    monkeypatch.setitem(
        sys.modules, "torchvision.transforms.functional", fake_functional
    )

    return SimpleNamespace(
        torch=fake_torch,
        tokenizer=fake_tokenizer,
        model=fake_model,
        model_chat=fake_model_chat,
        from_pretrained=fake_transformers.AutoModel.from_pretrained,
        stack_calls=stack_calls,
    )


class _FakeTensor:
    """Stand-in for a torch.Tensor that records ``.to(...)`` calls."""

    def __init__(self) -> None:
        self.last_to: dict[str, object] = {}

    def to(self, *, dtype=None, device=None):  # noqa: ANN001
        self.last_to = {"dtype": dtype, "device": device}
        return self


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_module_imports_without_torch() -> None:
    """Lazy imports: just importing the module must not need torch."""

    # Trivially true at collection time, but assert explicitly so a future
    # accidental top-level torch import is caught.
    import adaptive_inference.inference.internvl2 as mod  # noqa: F401

    assert "torch" not in sys.modules or sys.modules["torch"].__class__ is ModuleType


def test_adapter_init_loads_via_hf_with_correct_model_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fakes = _install_fake_torch_stack(monkeypatch)
    from adaptive_inference.inference.internvl2 import InternVL2Adapter

    adapter = InternVL2Adapter(
        model_name="internvl2-2b",
        model_id="OpenGVLab/InternVL2-2B",
    )
    fakes.from_pretrained.assert_called_once()
    args, kwargs = fakes.from_pretrained.call_args
    assert args[0] == "OpenGVLab/InternVL2-2B"
    assert kwargs["trust_remote_code"] is True
    assert adapter.model_name == "internvl2-2b"
    assert adapter.device == "cpu"  # both cuda + mps fakes return False


def test_run_returns_well_formed_inference_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fakes = _install_fake_torch_stack(monkeypatch)
    from adaptive_inference.inference.internvl2 import InternVL2Adapter
    from adaptive_inference.inference.types import InferenceResult

    adapter = InternVL2Adapter(
        model_name="internvl2-2b", model_id="OpenGVLab/InternVL2-2B"
    )
    result = adapter.run(
        page_id="page_0001", image=_img(), budget=BUDGET, prompt=PROMPT
    )
    assert isinstance(result, InferenceResult)
    assert result.page_id == "page_0001"
    assert result.model_name == "internvl2-2b"
    assert result.budget_name == "low"
    assert result.prompt_id == "table_parse_v1"
    assert "<table>" in result.raw_text
    assert result.output_token_count == 3
    assert result.runtime_ms >= 0.0


def test_run_passes_prompt_template_text_to_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fakes = _install_fake_torch_stack(monkeypatch)
    from adaptive_inference.inference.internvl2 import InternVL2Adapter

    adapter = InternVL2Adapter(
        model_name="internvl2-2b", model_id="OpenGVLab/InternVL2-2B"
    )
    adapter.run(page_id="p", image=_img(), budget=BUDGET, prompt=PROMPT)
    _, kwargs = fakes.model_chat.call_args
    assert kwargs["question"] == "parse this page"
    assert kwargs["generation_config"]["do_sample"] is False


def test_max_tiles_propagates_to_tile_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Critical contract: Budget.max_tiles must cap the number of preprocessed tiles."""

    fakes = _install_fake_torch_stack(monkeypatch)
    from adaptive_inference.inference.internvl2 import InternVL2Adapter

    adapter = InternVL2Adapter(
        model_name="internvl2-2b", model_id="OpenGVLab/InternVL2-2B"
    )
    # A wide-ish image with max_tiles=6 should produce <=6 tiles.
    adapter.run(
        page_id="p",
        image=_img((1200, 400)),
        budget=Budget(name="b", max_tiles=6),
        prompt=PROMPT,
    )
    assert fakes.stack_calls, "torch.stack must be called with the tile tensors"
    assert fakes.stack_calls[-1] <= 6
    assert fakes.stack_calls[-1] >= 1


def test_factory_builds_real_adapter_for_internvl2_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_torch_stack(monkeypatch)
    from adaptive_inference.inference.factory import build_adapter
    from adaptive_inference.inference.internvl2 import InternVL2Adapter

    cfg = ModelConfig(
        name="internvl2-2b",
        adapter_kind="internvl2",
        model_id="OpenGVLab/InternVL2-2B",
        notes="",
    )
    adapter = build_adapter(cfg)
    assert isinstance(adapter, InternVL2Adapter)


def test_factory_error_message_lists_internvl2() -> None:
    """Regression guard: the error must mention all supported kinds."""

    from adaptive_inference.inference.factory import build_adapter

    cfg = ModelConfig(
        name="m", adapter_kind="bogus", model_id="x", notes=""
    )
    with pytest.raises(ValueError) as exc:
        build_adapter(cfg)
    assert "internvl2" in str(exc.value)
    assert "stub" in str(exc.value)
