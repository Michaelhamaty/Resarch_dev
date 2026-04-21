import importlib


def test_subpackages_importable():
    for name in (
        "adaptive_inference",
        "adaptive_inference.runner",
        "adaptive_inference.inference",
        "adaptive_inference.parsing",
        "adaptive_inference.verifier",
    ):
        importlib.import_module(name)
