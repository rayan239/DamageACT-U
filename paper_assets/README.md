# DamageACT-U paper assets — repository edition

This directory contains the version-controlled, paper-facing evidence package built from the frozen DamageACT-U experiment. It is intentionally **source-first**: canonical/derived CSV and JSON evidence, LaTeX tables, editable SVG figures, captions, methodology specifications, claim guardrails, and QA notes are tracked here.

The frozen scientific release remains `paper-v1.0.0` at commit `618ce84790c7e96efd0a7df3dea98f8b37b36cd9`; this paper-assets commit does **not** move or rewrite that tag.

## Rendering

PNG and PDF copies are deterministic rendered derivatives and are intentionally not duplicated in Git. Rebuild them with:

```bat
python scripts\12_build_paper_assets.py --repo-root . --out paper_assets_rendered --overwrite
```

The generator hard-fails if key frozen results or decision guardrails drift.

## Main scientific guardrails

- final event-track experts: Seed 42 only;
- training-from-scratch reproduction is not claimed;
- post-TEST tuning is forbidden;
- clean-effectiveness criterion: **FAIL**;
- class safety: **PASS**;
- wrong-PRE safety: **PASS**;
- gate suppression: **PASS**;
- temporal-utility generalization: **PASS**;
- `ALL_PRECOMMITTED_CRITERIA_PASS=False`.

## Structure

- `figures/main/` — main manuscript vector figures;
- supplementary figures are generated deterministically by `scripts/12_build_paper_assets.py` and are not duplicated in Git;
- `tables/main/` — main tables as CSV + LaTeX;
- `tables/supplementary/` — supplementary tables as CSV + LaTeX;
- `data/canonical/` — frozen paper-result summaries copied from the canonical evidence tree;
- `data/derived/` — small deterministic paper-facing summaries;
- `MODEL_ARCHITECTURE_SPEC.md` — exact executed architecture;
- `ROUTER_TRAINING_OBJECTIVE_SPEC.md` — exact router objective;
- `CLAIM_TO_ASSET_MAP.md` — claim/evidence/guardrail map;
- `PAPER_ASSET_QA_REPORT.md` — devil's-advocate QA record.

Large per-building predictions and model checkpoints remain outside ordinary Git under the existing external-asset policy.
