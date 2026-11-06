# Legacy archive

This directory preserves superseded Agent-generated material for auditability. It is not the recommended implementation or reproduction path.

- `experiments/`: duplicate prototypes and the historical Kaggle driver. These files mix detectors, model patching, evaluation, and plotting in single scripts.
- `examples/`: the obsolete end-to-end demo based on the historical injection stack.
- `AI_HANDOVER.md`, `CHANGE_HANDOVER_MULTI_TURN.md`, `PROJECT.md`, `TEST_INFRA.md`, `TEST_READY.md`: historical Agent handover notes and specifications.
- `WHITEPAPER_*.md`: narrative explanations of the historical SG-Möbius formulation.
- `paper_audit_dossier.json`: historical paper-audit metadata.

The files under `paper/`, `formal_verification/`, and the original modules under `src/` remain in place because the existing regression and paper-integrity pipelines reference them. Their algebraic checks are not evidence of learned reversal detection or improved downstream accuracy.

Use `src/phase_attention.py`, `experiments/run_controlled_math.py`, `experiments/kaggle_reversal_eval.py`, and `RESEARCH_PLAN.md` for the current minimal research path.
