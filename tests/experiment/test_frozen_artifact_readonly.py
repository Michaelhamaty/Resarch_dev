"""Guard test: Phase 6 must never modify the frozen calibration artifact.

Contract 6 is the cornerstone of the matched-budget claim. This test
freezes the artifact's bytes (SHA-256), runs Phase 6 end-to-end on a
fixture, and re-checks the SHA. Any drift is a contract violation.
"""

from __future__ import annotations

from adaptive_inference.experiment.manifest import sha256_of_file
from adaptive_inference.experiment.runner import Phase6Config, run_phase6


def test_phase6_does_not_mutate_frozen_artifact(phase6_fixture):
    sha_before = sha256_of_file(phase6_fixture.frozen_budgets_path)

    cfg = Phase6Config(
        run_set_id="phase6_readonly_test",
        frozen_budgets_path=phase6_fixture.frozen_budgets_path,
        split_name="held_out_eval_split",
        held_out_manifest_path=phase6_fixture.manifest_path,
        records_path=phase6_fixture.records_path,
        image_root=phase6_fixture.image_root,
        model_config_path=phase6_fixture.model_config_path,
        prompt_config_path=phase6_fixture.prompt_config_path,
        output_root=phase6_fixture.output_root,
        random_seeds=(0,),
    )
    run_phase6(cfg, allow_stubbed_8b=True)

    sha_after = sha256_of_file(phase6_fixture.frozen_budgets_path)
    assert sha_before == sha_after, (
        "Phase 6 modified the frozen calibration artifact. Contract 6 says "
        "calibration is frozen before held-out evaluation; this test caught "
        "a violation."
    )
