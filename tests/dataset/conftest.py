from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def sample_pages_path() -> Path:
    return REPO_ROOT / "data" / "fixtures" / "sample_pages.json"


@pytest.fixture(scope="session")
def phase1_config_path() -> Path:
    return REPO_ROOT / "configs" / "dataset" / "phase1.yaml"
