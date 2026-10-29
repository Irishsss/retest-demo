"""Mini-benchmark: compare ReTest's planner with simple baselines on borderline cases."""
import numpy as np
import pandas as pd
import retest_core as rc


def run_benchmark(h, n_cases=500, seed=42, band=(0.02, 0.06)):
    rng = np.random.default_rng(seed)
    prior = h["prior"].values
    rows, n = [], 0
    while n < n_cases:
        i = rng.choice(len(h), p=prior)
        g, d, b = h.loc[i, ["gain", "drift", "bias"]]
        observed = h.loc[i, "apparent"] + rng.normal(0, rc.SIGMA_GAIN)
        if not (band[0] <= observed <= band[1]):
            continue                       # ReTest is for borderline results only
        n += 1
        truth_pass = g >= rc.THRESHOLD - 1e-9
        w0 = rc.update(prior, h, "original", observed)

        # 1. Accept the measured result as-is
        rows.append(dict(policy="Trust the measurement", cost=0.0,
                         wrong=(observed >= rc.THRESHOLD) != truth_pass))

        # 2. Fixed checklist: always run both sensor checks
        w = w0
        for t in ["Reference-cell check", "Synchronized temperature check"]:
            w = rc.update(w, h, t, rc.simulate_outcome(t, (g, d, b), rng))
        rows.append(dict(policy="Fixed checklist", cost=2.0,
                         wrong=(rc.p_pass(w, h) >= 0.5) != truth_pass))

        # 3. Always repeat the full test
        w = rc.update(w0, h, "Full rerun", rc.simulate_outcome("Full rerun", (g, d, b), rng))
        rows.append(dict(policy="Always full rerun", cost=8.0,
                         wrong=(rc.p_pass(w, h) >= 0.5) != truth_pass))

        # 4. ReTest planner
        adv, spent, done, _, _ = rc.retest_policy(h, observed, (g, d, b), rng)
        rows.append(dict(policy="ReTest planner", cost=spent, wrong=adv != truth_pass))
    df = pd.DataFrame(rows)
    out = df.groupby("policy", sort=False).agg(
        cases=("wrong", "size"), wrong_decision_rate=("wrong", "mean"), avg_test_cost=("cost", "mean"))
    out["avg_total_cost"] = out["avg_test_cost"] + rc.LOSS_WRONG * out["wrong_decision_rate"]
    return out
