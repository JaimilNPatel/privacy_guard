# Validation record

The final package was executed locally, not supplied as untested source.

- **Unit and integration checks:** 11 pytest tests pass. These include exhaustive feasible-transition budget/permission checks, state-index bijection, consent exposure and rejection semantics, fatigue updates, one-step Bellman consistency, a 50,000-session independent small-MDP Monte Carlo check, deterministic trajectories/training, observation aliasing, and actual sampled learning improvement.
- **Saved-result audit:** `python -m privacyguard.verify` reproduces 96 policy/seed evaluations and 600 stored scalar/interval checks, verifies all 40 checkpoint hashes, checks the exported seed-0 model, and confirms every exported browser action matches Python.
- **Training evidence:** 40 runs, 4,800,000 episodes, 52,325,865 sampled transitions. Raw counts are obtained from saved per-action visit tables; the exploring-start episodes can have shorter horizons.
- **Browser checks:** Headless Chrome executes the real offline page, changes presets, verifies empty-budget denial and permission blocking, runs/repeats a complete session, checks 7,128 nonterminal states, compares immediate-reward calculations, and checks desktop/mobile layout. No JavaScript errors or mobile horizontal overflow remain. Browser reward parity differs by at most floating-point roundoff (below 1e-10). Evidence is in `results/demo_verification.json` and `docs/demo_preview.png`.
- **Command-line demonstration:** `python -m privacyguard demo` executes a 12-step saved-policy trace; `python -m privacyguard explain --state 6 6 0 2 1 1` prints actual learned Q values and training counts.
- **Reproduction smoke run:** `python -m privacyguard.experiments --quick --workers 2` completes the full train/evaluate/plot pipeline in a separate output folder. Its results are not used in the report.
- **Report:** Tectonic 0.17.0 successfully compiles the LaTeX; the final PDF has exactly two A4 pages, including references, and was rendered and visually inspected. Both pages, all figure labels, table values, names, roll numbers, and GitHub-account status were checked. The Overleaf source bundle is self-contained.

The validation distinguishes simulated safety from a real privacy guarantee: the action guard is correct for the stated model, while the exposure and utility assumptions remain synthetic. The repository publication/portal submission checks remain external; see `SUBMISSION_CHECKLIST.md`.
