"""
ReTest mini-demo: core model.

A small, honest prototype of the ReTest idea:
  1. A pvlib digital twin of a PV module test.
  2. Competing explanations (true gain, irradiance-sensor drift, temperature-sensor bias)
     that can all produce the same measured result.
  3. A planner that picks the follow-up check with the best value of information
     for one decision: does the new module clear a +3% gain threshold?

Everything here runs on SIMULATED weather built with pvlib's clear-sky model.
Costs and losses are relative units chosen for illustration, not prices.
"""

from itertools import product

import numpy as np
import pandas as pd
import pvlib

# ---------------------------------------------------------------------------
# Experiment setup
# ---------------------------------------------------------------------------
LAT, LON, TZ, ALT = 33.58, -101.85, "America/Chicago", 990   # Lubbock, TX
TILT, AZIMUTH = 30, 180
PDC0 = 400.0           # W, baseline module nameplate
GAMMA = -0.0035        # 1/C, power temperature coefficient
THRESHOLD = 0.03       # decision: advance the new module only if true gain >= +3%

# Hypothesis grid: every combination is one "explanation" of the data
GAINS = np.round(np.arange(-0.01, 0.0801, 0.005), 4)   # true gain of new module
DRIFTS = np.array([0.0, -0.015, -0.03])                 # irradiance sensor reads low by 0, 1.5, 3%
BIASES = np.array([-3.0, 0.0, 3.0])                     # temperature sensor bias, C

# Prior beliefs (assumptions, stated openly)
P_DRIFT = {0.0: 0.70, -0.015: 0.15, -0.03: 0.15}
P_BIAS = {-3.0: 0.10, 0.0: 0.80, 3.0: 0.10}

# Measurement noise of each observation (1 sigma)
SIGMA_GAIN = 0.004     # apparent gain from the original test
SIGMA_REF = 0.003      # reference-cell check on irradiance sensor ratio
SIGMA_TEMP = 0.5       # synchronized temperature check, C
SIGMA_RERUN = 0.006    # clean full rerun, includes weather-to-weather variability

# Relative costs (illustrative units, not dollars)
COSTS = {"Reference-cell check": 1.0, "Synchronized temperature check": 1.0, "Full rerun": 8.0}
LOSS_WRONG = 20.0      # cost of a wrong advance or a wrong rejection, same units


# ---------------------------------------------------------------------------
# Digital twin
# ---------------------------------------------------------------------------
def simulate_weather(start="2026-04-06", days=7, seed=0):
    """Clear-sky irradiance with a smooth random cloud factor, plus a diurnal air temperature."""
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, periods=days * 96, freq="15min", tz=TZ)
    loc = pvlib.location.Location(LAT, LON, TZ, ALT)
    cs = loc.get_clearsky(times, model="ineichen")
    sp = loc.get_solarposition(times)

    # cloud factor: AR(1) process clipped to [0.35, 1]
    k = np.empty(len(times)); k[0] = 0.9
    for i in range(1, len(times)):
        k[i] = 0.92 * k[i - 1] + 0.08 * 0.85 + rng.normal(0, 0.05)
    k = np.clip(k, 0.35, 1.0)

    ghi, dni, dhi = cs["ghi"] * k, cs["dni"] * k**1.5, cs["dhi"] * (1 + 0.3 * (1 - k))
    poa = pvlib.irradiance.get_total_irradiance(
        TILT, AZIMUTH, sp["apparent_zenith"], sp["azimuth"], dni, ghi, dhi
    )["poa_global"].clip(lower=0)

    hour = times.hour + times.minute / 60
    t_air = 20 + 9 * np.sin((hour - 9) / 24 * 2 * np.pi)
    wind = pd.Series(2.0 + rng.normal(0, 0.3, len(times)).clip(-1.5, 1.5), index=times)
    return pd.DataFrame({"poa": poa, "t_air": t_air, "wind": wind}, index=times)


_TEMP = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]


def module_power(weather, pdc0):
    """True DC power of a module under the given weather (pvlib SAPM temperature + PVWatts DC)."""
    t_cell = pvlib.temperature.sapm_cell(weather["poa"], weather["t_air"], weather["wind"], **_TEMP)
    return pvlib.pvsystem.pvwatts_dc(weather["poa"], t_cell, pdc0, GAMMA), t_cell


def apparent_gain(week1, week2, gain, drift, bias):
    """
    What the lab would report. Baseline module measured in week 1 with healthy sensors.
    New module measured in week 2 with a possibly drifted irradiance sensor and a
    possibly biased temperature sensor. Each week is normalized by the twin's
    expected power computed from the *recorded* sensor values.
    """
    p1, t1 = module_power(week1, PDC0)
    expected1 = pvlib.pvsystem.pvwatts_dc(week1["poa"], t1, PDC0, GAMMA)
    pr1 = p1.sum() / expected1.sum()

    p2, t2 = module_power(week2, PDC0 * (1 + gain))
    poa_rec = week2["poa"] * (1 + drift)       # sensor reads low when drift < 0
    t_rec = t2 + bias
    expected2 = pvlib.pvsystem.pvwatts_dc(poa_rec, t_rec, PDC0, GAMMA)
    pr2 = p2.sum() / expected2.sum()
    return pr2 / pr1 - 1


def build_hypotheses(week1, week2):
    """Run the twin for every explanation on the grid; attach prior probabilities."""
    rows = []
    for g, d, b in product(GAINS, DRIFTS, BIASES):
        rows.append({"gain": g, "drift": d, "bias": b,
                     "apparent": apparent_gain(week1, week2, g, d, b),
                     "prior": P_DRIFT[d] * P_BIAS[b] / len(GAINS)})
    h = pd.DataFrame(rows)
    h["prior"] /= h["prior"].sum()
    return h


# ---------------------------------------------------------------------------
# Bayesian update and decision
# ---------------------------------------------------------------------------
def _norm_lik(x, mu, s):
    return np.exp(-0.5 * ((x - mu) / s) ** 2)


def update(weights, h, test, value):
    if test == "original":
        lik = _norm_lik(value, h["apparent"].values, SIGMA_GAIN)
    elif test == "Reference-cell check":
        lik = _norm_lik(value, h["drift"].values, SIGMA_REF)
    elif test == "Synchronized temperature check":
        lik = _norm_lik(value, h["bias"].values, SIGMA_TEMP)
    elif test == "Full rerun":
        lik = _norm_lik(value, h["gain"].values, SIGMA_RERUN)
    w = weights * lik
    return w / w.sum()


def p_pass(weights, h):
    return float(weights[h["gain"].values >= THRESHOLD - 1e-9].sum())


def decision_loss(weights, h):
    """Expected loss of deciding now with the better of advance / reject."""
    pp = p_pass(weights, h)
    return LOSS_WRONG * min(pp, 1 - pp)


def simulate_outcome(test, truth, rng):
    g, d, b = truth
    if test == "Reference-cell check":
        return d + rng.normal(0, SIGMA_REF)
    if test == "Synchronized temperature check":
        return b + rng.normal(0, SIGMA_TEMP)
    if test == "Full rerun":
        return g + rng.normal(0, SIGMA_RERUN)


def expected_loss_after(weights, h, test, rng, n=200):
    """Preposterior analysis: average decision loss after seeing this test's (unknown) result."""
    idx = rng.choice(len(h), size=n, p=weights)
    losses = []
    for i in idx:
        truth = tuple(h.loc[i, ["gain", "drift", "bias"]])
        y = simulate_outcome(test, truth, rng)
        losses.append(decision_loss(update(weights, h, test, y), h))
    return float(np.mean(losses))


def rank_checks(weights, h, done, rng):
    """Score every available check: expected decision loss after it, plus its cost."""
    rows = [{"action": "Decide now", "cost": 0.0,
             "expected_loss": decision_loss(weights, h)}]
    for t, c in COSTS.items():
        if t in done:
            continue
        rows.append({"action": t, "cost": c, "expected_loss": expected_loss_after(weights, h, t, rng)})
    r = pd.DataFrame(rows)
    r["total"] = r["cost"] + r["expected_loss"]
    return r.sort_values("total").reset_index(drop=True)


def retest_policy(h, observed, truth, rng, max_steps=3, log=False):
    w = update(h["prior"].values, h, "original", observed)
    done, spent, steps = [], 0.0, []
    for _ in range(max_steps):
        ranking = rank_checks(w, h, done, rng)
        best = ranking.iloc[0]
        steps.append((ranking, p_pass(w, h)))
        if best["action"] == "Decide now":
            break
        y = simulate_outcome(best["action"], truth, rng)
        w = update(w, h, best["action"], y)
        done.append(best["action"]); spent += best["cost"]
    advance = p_pass(w, h) >= 0.5
    return advance, spent, done, w, steps
