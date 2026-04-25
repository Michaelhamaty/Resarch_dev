"""Bridge from Phase 5 frozen artifact to Phase 6 runner inputs.

Pure translation layer. Does three jobs:

1. Convert ``FrozenBudget`` records into the existing ``Budget`` dataclass
   that every runner already speaks (name + max_tiles).
2. Reconstruct the ``ModelConfig`` for a given frozen-budget entry by
   merging the budget's ``model_name``/``adapter_kind`` with the extra
   metadata (``model_id``, notes) from the user-supplied
   ``configs/models/internvl2.yaml`` — so every Phase 6 run records the
   same model identity the rest of the repo uses.
3. Derive the calibration-measured reparse rate from the frozen
   artifact. The arithmetic is explicit about the cost model and raises
   if the inputs are inconsistent, rather than silently clipping.

No I/O beyond reading the model config YAML the caller points at.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..calibration.artifact import FrozenBudget, FrozenBudgets
from ..config.budgets import Budget
from ..config.models import ModelConfig, load_model_config


def frozen_to_budget(fb: FrozenBudget) -> Budget:
    """Reinterpret a frozen-artifact budget as the runner-facing ``Budget``."""

    return Budget(name=fb.name, max_tiles=fb.max_tiles)


def model_cfg_for(fb: FrozenBudget, model_config_path: str) -> ModelConfig:
    """Resolve the ``ModelConfig`` for a frozen budget against the repo registry.

    The frozen artifact carries ``model_name`` and ``adapter_kind`` but
    not ``model_id`` or notes. We pull those from the repo's model
    config by name, and assert that the ``adapter_kind`` agrees. A
    mismatch would mean the frozen artifact and the repo disagree on
    what "internvl2-2b" means, which is exactly the drift Phase 6 must
    catch early.
    """

    cfg = load_model_config(model_config_path, fb.model_name)
    if cfg.adapter_kind != fb.adapter_kind:
        raise ValueError(
            f"Frozen budget {fb.name!r} declares adapter_kind={fb.adapter_kind!r} "
            f"but model config at {model_config_path} has {cfg.adapter_kind!r} "
            f"for model {fb.model_name!r}. Refusing to proceed; re-freeze "
            f"Phase 5 or update the model registry before running Phase 6."
        )
    return cfg


@dataclass(frozen=True)
class CalibrationReparseRate:
    """The random-escalation policy's per-page probability, with provenance.

    ``degenerate`` signals that the derivation yielded ``0.0`` exactly;
    this happens when the adaptive system never escalated on the
    calibration split (e.g. when the verifier always says PASS under
    stubbed outputs). We still run, but the manifest prints a loud
    warning so Phase 7 knows the random baseline is a no-op.
    """

    probability: float
    source: str
    measured_cost_tiles: float
    low_max_tiles: int
    high_max_tiles: int
    degenerate: bool


def derive_calibration_reparse_rate(fb: FrozenBudgets) -> CalibrationReparseRate:
    """Compute the adaptive reparse rate that Phase 5 measured on calibration.

    Cost model (additive): ``cost_per_page = low + I[reparse] * high``.
    With ``measured_cost_tiles`` as the mean over the calibration split,

        p = (measured_cost_tiles - low_max_tiles) / high_max_tiles.

    The result is checked against ``[0.0, 1.0]``. Out-of-range values
    mean Phase 5 is using a different cost model than Phase 6 assumes,
    so we raise rather than clip — this forces a human to reconcile the
    accounting instead of silently producing a misleading random baseline.
    """

    sel = fb.adaptive_selection
    low = sel.low_max_tiles
    high = sel.high_max_tiles
    measured = sel.measured_cost_tiles

    if high <= 0:
        raise ValueError(
            f"Frozen adaptive_selection has non-positive high_max_tiles={high}; "
            f"cannot derive reparse rate."
        )

    p = (measured - low) / high
    if p < 0.0 or p > 1.0:
        raise ValueError(
            "Derived calibration reparse rate "
            f"p=({measured} - {low}) / {high} = {p:.6f} is outside [0,1]. "
            "This indicates the Phase 5 cost-accounting model differs from "
            "the additive assumption cost=low+I[reparse]*high that Phase 6 uses. "
            "Resolve the discrepancy before running the random baseline."
        )

    return CalibrationReparseRate(
        probability=p,
        source="phase5_frozen_artifact.adaptive_selection",
        measured_cost_tiles=measured,
        low_max_tiles=low,
        high_max_tiles=high,
        degenerate=(p == 0.0),
    )
