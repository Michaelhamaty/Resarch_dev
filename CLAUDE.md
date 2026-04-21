# Project Instructions

Read this first before making changes:
- docs/specs/adaptive_inference_build_brief.md

Goal:
Build the MVP research system for matched-budget adaptive inference on complex English table pages.

Rules:
- Keep the project narrow and simple.
- Do not add learned routing.
- Do not add OCR in the core trigger path.
- Do not add crop repair.
- Do not add multi-step repair loops.
- Keep the architecture modular.
- Keep files focused and not overly large.
- Add tests for important logic.
- Do one milestone at a time.

Important:
- This is a research MVP, not a product.
- Protect the fairness/accounting ideas from the spec.
- Do not broaden scope unless asked.

Implementation order:
1. repo scaffold
2. config and interfaces
3. single-page inference runner scaffold
4. deterministic structural verifier
5. calibration utilities
6. evaluation/logging pipeline