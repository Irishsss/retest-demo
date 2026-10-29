# ReTest mini-demo

**Know which solar measurement is worth repeating.**
Proof of concept for the ReTest proposal, TeraOpenScience Idea Innovation Business Science Challenge 2026 (#IdeaInnovationBusiness).

## What it shows
- A pvlib digital twin of a two-week PV module comparison test.
- A module with a true **+2.0%** gain is reported at **+5.2%** because the week 2 irradiance sensor reads 3% low. That crosses the +3% go/no-go threshold and leads to a wrong "advance".
- The planner compares a reference-cell check, a synchronized temperature check and a full rerun by value of information plus cost. It picks two cheap checks (2 units vs 8 for a rerun), and the probability of clearing +3% drops from 85% to 2%. Decision: do not advance (correct).
- Mini-benchmark on 500 borderline cases (reported gain between +2% and +6%):

| Policy | Wrong decisions | Avg follow-up test cost (units) |
|---|---|---|
| Trust the measurement | 22.4% | 0.00 |
| Fixed checklist (both sensor checks) | 8.0% | 2.00 |
| Always full rerun | 7.8% | 8.00 |
| **ReTest planner** | **9.6%** | **1.04** |

ReTest roughly halves follow-up test spend against the fixed checklist, with slightly more wrong decisions in this run. That trade-off is set by the loss value a lab chooses, and making it explicit is the point.

## Honest limits
- Weather is **simulated** (pvlib clear-sky with a random cloud factor, Lubbock TX). Phase 1 replays public DOE PVDAQ data.
- Costs and losses are **relative units**, not prices.
- The planner is one-step value of information. Phase 1 benchmarks it against goal-driven and risk-aware experimental design, with deliberately wrong priors and correlated faults.

## Run it
```bash
pip install -r requirements.txt
jupyter notebook ReTest_demo.ipynb
```
Files: `retest_core.py` (twin, explanations, planner), `benchmark.py` (policy comparison), `ReTest_demo.ipynb` (walkthrough, about 1 minute to run).

Team: Rishika Vats (technical lead), Ayush Raj, Uswa Tariq.
