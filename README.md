# PrivacyGuard

**Reinforcement learning for adaptive data-sharing decisions**  
Patel Jaimil Nareshkumar (B23CH1029) · Nitin Koronde R. (B23CH1028)

PrivacyGuard learns whether to deny, share a coarse response, ask for consent, or share full data. A 12-request session has a limited exposure budget. Repeated consent prompts cause fatigue. A tabular Q-learning agent learns which requests deserve scarce resources, with a separate hard permission and budget guard.

**Start here:** open `demo/index.html` (or double-click `START_DEMO.bat`). The saved agent, measured results, and live session audit work offline without Python. Read `report/main.pdf` for the two-page submission report. See `docs/RESULTS.md` for the actual experiment summary.

The code, simulator, training loop, experiment runner, plots, and demo were implemented for this project; no public RL project was forked or copied. The mathematical algorithms are standard and cited. `docs/PROVENANCE.md` records assistance and the actual development history. Read and understand the implementation before presenting it as your team's submission.

## Install and run

Use Python 3.10 or later. Tested with Python 3.11.9, NumPy 2.4.6, Matplotlib 3.11.1, and pytest 9.1.1 (the verification record contains the actual installed versions).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python -m privacyguard demo
python -m privacyguard evaluate --episodes 5000 --seed 90000
python -m privacyguard explain --state 6 6 0 2 1 1
```

On macOS/Linux activate with `source .venv/bin/activate`. All later commands are identical. If `python` is unavailable use `python3` or, on Windows, `py -3`.

`requirements-lock.txt` records the exact tested dependency versions for Python 3.11. For a matching environment use `python -m pip install -r requirements-lock.txt` instead of the broad requirements file.

The package runs directly from this folder; installation with `pip install -e .` is optional. There is no PyTorch, GPU, API key, internet request, personal dataset, or external RL framework in the project.

## Reproduce the submitted experiments

```powershell
python -m privacyguard.experiments --episodes 120000 --seeds 8 --eval-episodes 5000 --workers 4
python -m privacyguard.verify
python -m privacyguard.report_assets
```

The runner trains five variants from scratch for each seed, evaluates every policy on 5,000 held-out sessions per seed, saves raw seed means and checkpoints, exports the offline demo, and regenerates figures. `--workers 1` uses less CPU. `--resume` reuses only compatible saved runs. Training takes a few minutes on the tested machine; actual timings appear in `results/manifest.json` and depend on hardware. To check the pipeline quickly:

```powershell
python -m privacyguard.experiments --quick --workers 2
```

The quick run writes `results_quick/` and cannot overwrite submitted results. It is a smoke test, not the report experiment. The final submitted agent is always seed 0, chosen in advance; no selection by test score occurs. Evaluation uses greedy learned actions with exploration disabled. Results are deterministic under the same code, configuration, and NumPy RNG implementation; figures may vary slightly across rendering versions.

## Why RL is necessary here

Full disclosure of a low-value request can exhaust the budget before a high-value request arrives. Asking for consent consumes no information budget when rejected, but costs attention and reduces the acceptance probability of later prompts. Actions therefore change future rewards. Remaining requests, budget, and fatigue are observed, making the simulator a finite episodic Markov decision process.

The main solution is model-free Q-learning. SARSA is a second RL comparator. A gamma=0 learner tests the value of future planning. Budget-blind and fatigue-blind Q learners test the value of those observations. Greedy and fixed strategies are evaluation references only. Exact backward induction is an analytical ceiling and simulator check; it is never used for policy training, imitation, initialisation, or checkpoint selection.

## State, actions, and reward

Six observations: `(requests_remaining, exposure_budget, consent_fatigue, utility_bin, sensitivity_bin, full_permission)`. Cardinalities: `13 × 11 × 3 × 3 × 3 × 2 = 7,722`, including terminal states; 7,128 nonterminal combinations. The full Q table has 30,888 scalar entries. Terminal combinations and some structurally unreachable combinations are retained for simple indexing.

Four actions: `0=Deny`, `1=Coarse`, `2=Ask consent`, `3=Full`. The guard blocks full sharing without existing permission, and blocks any action that could spend more exposure than remains. Coarse sharing assumes a pre-existing limited permission. A consent acceptance permits this single disclosure only. A rejection discloses nothing. Every strategy and training variant uses the same guard.

For utility `U ∈ {1,3,7}`, full exposure `E ∈ {1,2,4}`, coarse utility is `0.60 U` and coarse exposure is `ceil(0.35 E)`. For realised utility `v`, exposure `e`, fatigue `f`, and prompt indicator `I_ask`:

`reward = v - 0.45 e - (0.25 + 0.30 f) I_ask - 0.15 U I[v=0]`.

Consent acceptance probability is `clip(0.92 - 0.12 sensitivity_bin - 0.25 fatigue, 0.10, 0.95)`. Asking increases fatigue by one up to two; every other action reduces it by one down to zero. Exposure is never replenished inside a session. New request contexts are independent draws with the distributions in `results/config.json`.

These are explicit synthetic assumptions, not empirically calibrated user preferences. Consent never discounts the physical exposure cost. The budget is not a differential-privacy guarantee, and the coarse response is an abstract action, not a proven anonymisation transform. See `docs/MODEL_CARD.md` for assumptions and limits.

## Repository guide

| Path | Purpose |
|---|---|
| `privacyguard/env.py` | Simulator, observations, costs, stochastic consent, hard guard |
| `privacyguard/train.py` | Q-learning, SARSA, and observation/planning ablations |
| `privacyguard/analysis.py` | Paired evaluation, reference policies, exact diagnostic |
| `privacyguard/experiments.py` | Multiprocess seeded experiments and raw artifacts |
| `privacyguard/plots.py` | Figures regenerated from experiment CSV files |
| `privacyguard/verify.py` | Cross-file consistency, checkpoint, and saved-result audit |
| `agents/q_learning_seed0.npz` | Preselected trained agent, configuration, and visit counts |
| `results/` | All 40 checkpoints, raw CSVs, hashes, curves, shifts, and metadata |
| `demo/` | Offline interactive demonstration using the exported trained agent |
| `report/` | LaTeX source, generated numeric macros, figures, and compiled PDF |
| `docs/` | Model details, results, viva guide, compliance, proposal, provenance |
| `tests/` | Invariant, stochastic, environment, and learning checks |

## Build the LaTeX report

`report/main.pdf` is already compiled. To rebuild, install TeX Live/MiKTeX or Tectonic, then run:

```powershell
python -m privacyguard.report_assets
cd report
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Or run `tectonic main.tex` in `report/`. For Overleaf upload the contents of `report/` and select `main.tex`. The report is self-contained and does not require paths outside that directory.

## GitHub submission

The supplied URL, `https://github.com/JaimilNPatel`, is an account profile, not a verified project repository. The report clearly labels it as such. Do not submit the profile as the required repository link.

Create an empty repository in that account, then from this folder:

```powershell
git remote add origin https://github.com/JaimilNPatel/PrivacyGuard.git
git push -u origin main
```

That remote is a suggested repository name, not an assertion that it exists. Use the actual repository URL if a different name is created. Update `report/team.tex` with that verified repository URL and rebuild the PDF. If private, grant access to `shubhambagwari` and `divyaanshmertia` as specified in the Phase 3 PDF. No remote publication or course-portal submission has been performed by this package.

The included Git history contains genuine development commits made while building this package, labelled `Codex`. No earlier work, team authorship, commit dates, or external submission is fabricated. Team members should continue their real work and commit their own changes; do not manufacture history.
