# Adaptive Inference Build Brief

## Purpose of this document

This document turns the ideas from **Intense Research Plan.md** into a practical build brief that can be handed to an implementation agent such as Claude Code. It focuses on the most important project ideas, the MVP, the backend/core system design, and a clean path from research concept to working software.

The goal is not to build a general document parser. The goal is to build a **conditional-compute research system** for complex table parsing.

---

## What this project is really building

At its core, this project is a **decision system over inference**, not just a table parser.

A normal parser spends the same compute budget on every page.
This project instead asks:

> Can a small model spend compute more intelligently by using a cheap first pass, checking whether the result looks structurally broken, and only then paying for a more expensive second pass?

So the real product is:

- a **small-model parsing pipeline**
- with a **deterministic structural verifier**
- and a **one-shot escalation policy**
- under **matched average end-to-end compute**

The main research thesis is:

> Better inference policy may recover part of the quality gap that would otherwise require a larger model.

That framing should guide every implementation decision.

---

## Core research idea distilled

The narrow question is whether **InternVL2-2B** can outperform fixed-cost 2B baselines, and recover some of the gap to **InternVL2-8B**, by doing the following on English table pages:

1. Run a **low-budget full-page parse**.
2. Check whether predicted tables are structurally suspicious.
3. If suspicious, rerun the **same page** with a **higher image-tile budget**.
4. Keep the final result while staying within a matched average compute budget.

This means the project depends on four core ideas:

### 1. Heterogeneous difficulty
Not all table pages are equally hard. Fixed-cost inference wastes compute on easy pages and under-spends on hard pages.

### 2. Precision-first escalation
The verifier does not need to know whether the parse is fully correct. It only needs to identify cases where spending extra compute is likely worth it.

### 3. Budget redistribution
The system is allowed to spend less on some pages and more on others, as long as the **average cost over the evaluation distribution** stays matched.

### 4. Controlled comparison
This is not a product benchmark free-for-all. It is a tightly controlled experiment with pinned data, frozen budgets, fixed prompts, and identical timing conditions.

---

## The MVP in one sentence

Build a batch inference engine that:

- parses each English table page once with **InternVL2-2B at low budget**,
- runs a **deterministic HTML table structure verifier**,
- optionally reparses the whole page once at **higher budget**,
- stores final output and full cost metadata,
- and compares this system against fixed-cost baselines.

---

## MVP scope to preserve

The best way to succeed is to keep the MVP narrow and disciplined.

### Include

- English table pages only
- full-page parsing only
- HTML table output only
- deterministic verifier only
- one low-budget pass + at most one high-budget reparse
- strict runtime / cost logging
- evaluator-compatible outputs
- matched-budget comparisons to fixed baselines

### Exclude for the MVP

- learned routers
- OCR-assisted routing
- crop-level repair
- multi-step repair loops
- semantic self-critique agents
- multilingual expansion
- general page parsing claims
- rotated-table handling unless already simple and stable

This discipline matters. The narrower the MVP, the easier it is to defend the experiment and the easier it is to build correctly.

---

## Why the backend matters

The backend is not just infrastructure. It is the thing that makes the scientific claim believable.

If the backend is messy, the experiment is messy.
If the timing is inconsistent, the fairness claim weakens.
If the routing logic is hard to audit, the policy claim weakens.

So the backend must do four jobs well:

1. **Orchestration** — run each page through the correct state transitions
2. **Accounting** — measure online costs consistently
3. **Logging** — preserve all decisions and artifacts for later analysis
4. **Reproducibility** — ensure every comparison uses the same conditions

---

## Recommended system model

Treat each page as a deterministic transaction through a state machine.

### Page state machine

```text
RECEIVED
  -> PREPROCESSED
  -> FIRST_PASS_DONE
  -> VERIFIED
  -> REPARSE_DONE (optional)
  -> FINALIZED
  -> SCORED
```

Each transition should create structured logs and stable artifacts.

### Why this is the right mental model

This makes the project easy to reason about:

- every page has one lifecycle
- every decision is inspectable
- every runtime cost can be assigned to a stage
- every failure can be localized to a stage

That is much better than thinking of the project as “a script that runs the model.”

---

## Recommended backend architecture

For the MVP, the safest architecture is a **single-process or lightly modular batch runner**, not a distributed microservice system.

### Recommended MVP architecture

```text
Dataset Loader
   -> Page Runner
      -> Preprocessor
      -> Model Inference Adapter
      -> Output Parser
      -> Structural Verifier
      -> Escalation Policy
      -> Artifact Writer
      -> Metrics Logger
```

### Why this is the best MVP architecture

- simplest to debug
- easiest to keep deterministic
- lowest overhead for timing fairness
- easiest for Claude Code to build incrementally
- avoids distributed latency noise
- keeps research logic and execution logic tightly aligned

### Avoid for now

- queue systems
- multi-service deployment
- async worker farms
- web dashboards as first-class priority
- heavy orchestration frameworks

Those can come later if the experiment grows.

---

## Core modules to build

### 1. Dataset / subset module
Responsible for:

- loading the pinned OmniDocBench snapshot
- extracting English table-page IDs
- extracting the pre-registered hard subset
- loading calibration split and evaluation split

Important principle:
This module should produce **frozen page lists**. Once those are locked, the rest of the pipeline should consume them as facts.

---

### 2. Inference adapter
Responsible for:

- loading InternVL2 models
- accepting page images and a tile budget
- applying a fixed prompt template
- returning raw generated page output
- reporting output token count and timing

Important principle:
The inference adapter should hide model-specific details behind a stable interface.

Suggested conceptual interface:

```text
run_page_inference(page_image, model_name, tile_budget, prompt_config) -> InferenceResult
```

Where `InferenceResult` includes:

- raw output text
- decode metadata
- token count
- runtime breakdown
- model identifier
- budget identifier

---

### 3. Output normalization layer
Responsible for:

- extracting HTML tables from page markdown
- standardizing representation for later evaluation
- preserving raw outputs alongside normalized ones

Important principle:
Never throw away the original model output. Always keep:

- raw generated text
- normalized extracted table blocks
- final chosen output

This is critical for debugging failures and verifying the verifier.

---

### 4. Deterministic structural verifier
Responsible for the main adaptive decision.

It should check:

- table presence
- HTML parsability
- span expansion validity
- rectangular grid consistency
- degenerate table structure

Important principle:
This verifier is a **high-precision trigger**, not a correctness oracle.

That means it should be conservative.
A false positive costs extra compute.
A false negative misses a chance to improve.
For the MVP, it is okay to miss some semantically wrong but structurally valid outputs.

Suggested conceptual interface:

```text
verify_tables(page_output) -> VerifierResult
```

Where `VerifierResult` includes:

- decision: PASS or REPARSE
- failure codes
- number of predicted tables
- parseability flags
- span-normalization status
- normalized table summaries

---

### 5. Escalation policy
Responsible for one simple rule:

```text
if verifier says REPARSE:
    rerun same page with same model at B_high
else:
    keep first-pass result
```

Important principle:
The policy should stay intentionally dumb in the MVP.
Do not sneak in extra heuristics unless explicitly part of a planned ablation.

---

### 6. Artifact and logging system
Responsible for writing everything needed for reproducibility.

For each page, log:

- page ID
- split name
- subset membership
- first-pass budget
- whether reparse happened
- reparse budget if triggered
- verifier decision
- verifier failure codes
- first-pass runtime
- verifier runtime
- reparse runtime
- total runtime
- output token counts
- raw output paths
- final output path

Important principle:
Think of this as a **forensic trail**. Every result should be reconstructable.

---

### 7. Evaluation runner
Responsible for:

- emitting evaluator-compatible files
- calling OmniDocBench evaluation consistently
- collecting TEDS and edit-distance outputs
- storing result JSONs per system

Important principle:
The evaluation layer should be **separate from online inference**. The paper claim depends on online cost accounting, but offline scoring should still be automated and reproducible.

---

## Recommended repository structure

```text
repo/
  configs/
    models/
    prompts/
    budgets/
    runs/
  data/
    splits/
    subsets/
  src/
    dataset/
    inference/
    parsing/
    verifier/
    policy/
    runner/
    logging/
    evaluation/
    analysis/
  tests/
    verifier/
    parsing/
    smoke/
  outputs/
    runs/
    metrics/
    analysis/
  scripts/
    subset_extraction/
    calibration/
    main_runs/
    analysis/
  docs/
    methods/
    runbooks/
```

Why this helps:

- separates research constants from code
- keeps outputs auditable
- makes Claude Code less likely to mix analysis and runtime logic
- encourages clean experiment-driven organization

---

## Build contracts Claude Code should follow

To reduce drift, the implementation should honor explicit contracts.

### Contract 1: One page, one policy path
Each page gets:

- exactly one first pass
- at most one full-page reparse
- never a third pass

### Contract 2: No hidden compute
All online steps in the tested pipeline count toward compute accounting.

### Contract 3: Reproducible prompts
Prompt template must be versioned and fixed across compared systems.

### Contract 4: Stable output format
All predicted tables must be emitted as HTML blocks inside page markdown.

### Contract 5: Raw artifacts are preserved
Never overwrite raw outputs without keeping a copy.

### Contract 6: Calibration is frozen before held-out evaluation
Budgets are chosen on the calibration split and then locked.

### Contract 7: Timing is comparable
Same hardware, batch size, precision policy, page order, and decode settings across systems.

---

## The most important theory behind the verifier

The verifier is not trying to solve table understanding.
It is trying to solve **compute allocation**.

That is the central theoretical simplification that makes the project feasible.

A full semantic verifier would be much harder. But a structural verifier can still be useful because many catastrophic low-budget failures show up as malformed or degenerate table structure.

So the verifier acts like a sensor in a control loop:

- first pass = probe
- verifier = sensor
- reparse = control action
- compute budget = constrained resource
- final metric = objective

This is one of the strongest conceptual frames for the whole project.

---

## Recommended MVP implementation order

This order is designed for smooth execution and low chaos.

### Phase 1 — Freeze the experiment universe
Build first:

- dataset pinning
- subset extraction
- split files
- page ID manifests

Success condition:
You can point to an immutable list of page IDs for calibration and held-out evaluation.

### Phase 2 — Build single-pass inference scaffold
Build next:

- page loader
- prompt template
- model adapter for 2B and 8B
- output writer
- runtime logging

Success condition:
You can run a smoke set and generate evaluator-compatible outputs.

### Phase 3 — Build output extraction and verifier
Build next:

- HTML extraction
- parser
- span expansion
- grid normalization
- verifier codes
- verifier unit tests

Success condition:
The verifier returns deterministic PASS / REPARSE results and has synthetic tests for malformed tables.

### Phase 4 — Add adaptive routing
Build next:

- one-shot reparse logic
- final output replacement logic
- page-level end-to-end logging

Success condition:
One page can fully traverse the adaptive pipeline with stable artifacts.

### Phase 5 — Calibration tooling
Build next:

- budget sweep runner
- calibration summaries
- matched-cost search
- locked budget export

Success condition:
You can freeze `B_low`, `B_high`, `B_fix_2B`, and `B_fix_8B` before the main evaluation.

### Phase 6 — Main benchmark runs
Build next:

- fixed 2B low-budget baseline
- fixed 2B matched-cost baseline
- random-escalation baseline
- fixed 8B matched-cost baseline
- adaptive 2B run

Success condition:
All systems complete under the same run harness.

### Phase 7 — Analysis package
Build last:

- result tables
- cost-vs-accuracy plots
- reparse-helped / hurt slices
- gap-closure calculations
- manual review sampling helpers

Success condition:
You can produce paper-ready summaries from stored logs.

---

## MVP technologies worth considering

Keep technology choices in service of the theory.
Do not let the stack become the project.

### Likely useful technologies

- **Python** for orchestration, parsing, evaluation wrappers, and analysis
- **PyTorch / Hugging Face ecosystem** for model loading and inference plumbing if compatible with your InternVL2 setup
- **HTML parsing libraries** for deterministic structural checks
- **JSON / JSONL / Parquet** for structured logging and metrics
- **pytest** for verifier and parsing tests

### Guiding rule
Choose boring tools for the backend.
The novelty should live in the inference policy and experiment design, not in infrastructure cleverness.

---

## Suggested data model

For clean implementation, define a small number of stable records.

### PageRecord
Represents a page in the experiment universe.

Fields might include:

- page_id
- image_path
- split
- contains_table
- is_english_table_page
- is_hard_table_page
- metadata attributes

### InferenceResult
Represents one model call.

Fields might include:

- page_id
- model_name
- tile_budget
- raw_text
- output_token_count
- runtime_ms
- stage_name

### VerifierResult
Represents the structural check.

Fields might include:

- page_id
- decision
- failure_codes
- predicted_table_count
- html_parse_ok
- span_normalization_ok

### PageRunRecord
Represents the final page transaction.

Fields might include:

- page_id
- first_pass_result
- verifier_result
- reparse_result_or_null
- final_output_source
- total_runtime_ms
- total_output_tokens
- final_budget_used

This kind of explicit typing will make Claude Code much more reliable.

---

## Key baselines that must not be skipped

If you want the research claim to survive scrutiny, these baselines matter.

### Fixed 2B low-budget baseline
Shows the value of any extra compute at all.

### Fixed 2B matched-cost baseline
Shows whether adaptive routing is better than simply spending the same average budget upfront.

### Random escalation baseline
Shows whether the policy is actually useful, versus escalation being helpful regardless of decision quality.

### Fixed 8B matched-cost baseline
Shows how much of the larger-model gap the adaptive small model can recover.

These are not optional decorations. They are what make the central claim legible.

---

## What to optimize for first

Claude Code should optimize for the following order of priorities:

1. **Correctness of pipeline behavior**
2. **Deterministic logging and reproducibility**
3. **Clean verifier implementation**
4. **Accurate cost accounting**
5. **Ease of running baselines**
6. **Only then: speed, refactoring, convenience tooling**

This order matters because a fast system with muddy accounting is scientifically weak.

---

## Future implementation ideas after the MVP

Once the MVP is stable, the following expansions make sense.

### 1. Better routing
Move from a rule-based structural verifier to:

- learned routing
- hybrid routing
- confidence-based routing
- semantic verification signals

### 2. Local repair
Instead of reparsing the whole page, escalate only on detected table regions or suspicious sections.

### 3. Multi-step policy
Allow more than one escalation option, such as:

- low -> medium
- medium -> high

### 4. OCR-assisted evidence
Use OCR-derived structural evidence to support routing decisions.

### 5. Broader scope
Expand beyond English tables to:

- multilingual tables
- rotated tables
- general document parsing

### 6. More formal policy analysis
Study whether routing precision, recall, and expected gain can be modeled more explicitly.

These are future directions, not MVP requirements.

---

## Risks Claude Code should guard against

### 1. Scope creep
The fastest way to derail this project is to turn it into a general document AI platform.

### 2. Hidden variability
If prompts, decode settings, hardware, or page order drift across systems, comparisons weaken.

### 3. Bad artifact hygiene
Losing raw outputs or mixing run folders will make failure analysis painful.

### 4. Over-smart routing
Adding many heuristics early will make the adaptive system hard to interpret.

### 5. Weak testing of the verifier
The verifier is central. It needs unit tests on malformed and valid synthetic tables before large runs.

### 6. Calibration leakage
Do not tune budgets on the held-out evaluation split.

### 7. Premature optimization
Do not start with distributed systems or complex scheduling unless the single-runner approach clearly breaks.

---

## Minimal test strategy

Claude Code should build tests early.

### Unit tests
For:

- HTML extraction
- HTML parse failures
- rowspan / colspan expansion
- rectangularity checks
- degenerate table detection

### Smoke tests
For:

- one page through fixed-cost 2B
- one page through adaptive 2B
- artifact generation
- evaluator-compatible output generation

### Consistency tests
For:

- stable prompt rendering
- stable logging schemas
- deterministic page ordering

This test layer is what makes iteration safe.

---

## A simple success definition for the build

A successful first build is not “state of the art.”
A successful first build is:

- the pipeline runs end-to-end on a smoke set
- the verifier makes deterministic decisions
- the adaptive system triggers reparses correctly
- per-page artifacts are complete
- baselines run under the same harness
- evaluation outputs are usable
- calibration can freeze budgets cleanly

That is the foundation. Results come after that.

---

## Final guidance for implementation agents

If an implementation agent is building this, it should think like a research engineer, not like a startup app engineer.

The right mindset is:

- keep the system narrow
- make states explicit
- log everything important
- preserve every artifact
- separate online inference from offline analysis
- freeze experimental constants early
- prefer boring infrastructure and clear interfaces

The most important outcome is a **trustworthy adaptive inference experiment**.
Not a flashy stack.
Not a giant codebase.
Not a universal parser.

---

## Recommended immediate next tasks

1. Create repo structure and config skeleton.
2. Implement subset extraction and frozen split manifests.
3. Implement a single-page inference adapter for InternVL2-2B.
4. Define the page output contract and artifact schema.
5. Implement HTML table extraction and verifier tests.
6. Add one-shot adaptive routing.
7. Add calibration runner.
8. Add all required baselines.
9. Add evaluation and analysis scripts.

---

## One-paragraph handoff summary

This project should be implemented as a deterministic batch inference system for complex English table pages. Each page gets a low-budget full-page parse with InternVL2-2B, a deterministic structural verification pass over predicted HTML tables, and at most one higher-budget full-page reparse if structural problems are detected. The backend should prioritize reproducibility, cost accounting, explicit artifacts, and baseline comparability over infrastructure complexity. The MVP should stay narrow: no learned router, no OCR routing, no crop repair, no multi-step repair. Once the core pipeline is stable and auditable, future work can expand into learned routing, local repair, broader datasets, and richer policy analysis.
