"""Checks a reviewer can run to confirm the key building blocks.

Run:  python -m pytest tests/        (or)   python tests/test_core.py
Tests that need the patient data or the analysis outputs are skipped if those files are absent.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thkfre.config import (RAW_DATA, DERIVED_DIR, KFRE_S0_2Y_NORTH_AMERICA,  # noqa: E402
                           KFRE_S0_2Y_NON_NORTH_AMERICA, THKFRE_SLOPE, THKFRE_S0_2Y)
from thkfre.kfre import kfre_risk, thkfre_risk, linear_predictor  # noqa: E402
from thkfre.finegray import fit_finegray, fit_cox  # noqa: E402
from thkfre import metrics as M  # noqa: E402


def _skip_if_missing(path):
    if not Path(path).exists():
        print(f"  (skipped: {Path(path).name} not found)")
        return True
    return False


# ---------------------------------------------------------------- equations
def test_non_north_american_baseline_from_paper():
    # Tangri et al. JAMA 2016: baseline 2-year risk lowered by 32.9%
    assert round(1 - (1 - KFRE_S0_2Y_NORTH_AMERICA) * (1 - 0.329), 4) == KFRE_S0_2Y_NON_NORTH_AMERICA


def test_kfre_matches_risks_in_hospital_file():
    """Our KFRE code reproduces the 2-year risks pre-computed in the hospital extract (NA baseline)."""
    if _skip_if_missing(RAW_DATA):
        return
    raw = pd.read_excel(RAW_DATA)
    ok = raw.UACR_true >= 1                      # rows whose UACR was not corrected in step 1
    df = pd.DataFrame({"age10": raw.age / 10, "male": raw.gender, "egfr5": raw.egfr_new / 5,
                       "ln_uacr": np.log(raw.UACR_true)})[ok]
    diff = np.abs(kfre_risk(df, KFRE_S0_2Y_NORTH_AMERICA) - raw.prob_2y_4v[ok])
    assert diff.max() < 1e-9


def test_thkfre_is_monotone_recalibration_of_kfre():
    df = pd.DataFrame({"age10": [5, 7, 8.5], "male": [1, 0, 1], "egfr5": [3, 6, 10], "ln_uacr": [7, 4, 1]})
    lp = linear_predictor(df)
    expected = 1 - THKFRE_S0_2Y ** np.exp(THKFRE_SLOPE * lp)
    assert np.allclose(thkfre_risk(df), expected)
    # same ranking as the KFRE
    assert (np.argsort(thkfre_risk(df)) == np.argsort(kfre_risk(df, KFRE_S0_2Y_NON_NORTH_AMERICA))).all()


def test_worked_example():
    """Hand-checkable example for the calculator: 60-year-old man, eGFR 20, UACR 1000 mg/g."""
    df = pd.DataFrame({"age10": [6.0], "male": [1], "egfr5": [4.0], "ln_uacr": [np.log(1000)]})
    lp = (-0.2201 * (6.0 - 7.036) + 0.2467 * (1 - 0.5642) - 0.5567 * (4.0 - 7.222)
          + 0.4510 * (np.log(1000) - 5.137))
    assert abs(linear_predictor(df)[0] - lp) < 1e-12
    print(f"  worked example: LP = {lp:.4f}; KFRE-nonNA = {kfre_risk(df, 0.9832)[0]:.3f}; "
          f"TH-KFRE = {thkfre_risk(df)[0]:.3f}")


# ---------------------------------------------------------------- statistics
def _toy(seed=1, n=400, competing=True):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t1 = rng.exponential(1 / np.exp(0.8 * x))
    t2 = rng.exponential(2.0, n) if competing else np.full(n, np.inf)
    c = rng.uniform(0.5, 3, n)
    time = np.minimum.reduce([t1, t2, c])
    status = np.select([time == t1, time == t2], [1, 2], 0)
    return x, time, status


def test_aalen_johansen_equals_km_without_competing_events():
    _, time, status = _toy(competing=False)
    km = 1.0
    for u in np.unique(time[status == 1]):
        if u > 1.5:
            break
        km *= 1 - ((time == u) & (status == 1)).sum() / (time >= u).sum()
    assert abs(M.aalen_johansen(time, status, 1.5) - (1 - km)) < 1e-12


def test_finegray_without_competing_events_equals_cox():
    x, time, status = _toy(competing=False)
    fg = fit_finegray(x, time, status)
    cox = fit_cox(x, time, (status == 1).astype(int))
    assert abs(fg.beta[0] - cox.beta[0]) < 1e-10


def test_finegray_recovers_direction_and_cif():
    x, time, status = _toy(seed=3, n=800)
    fg = fit_finegray(x, time, status)
    assert fg.beta[0] > 0.3                         # true cause-specific effect 0.8
    # mean predicted CIF close to Aalen-Johansen estimate
    pred = fg.predict_cif(x - 0, 1.0)
    assert abs(pred.mean() - M.aalen_johansen(time, status, 1.0)) < 0.02


def test_harrell_c_extremes():
    time = np.array([1., 2., 3., 4.])
    status = np.array([1, 1, 0, 0])
    assert M.harrell_c(time, status, np.array([4., 3., 2., 1.]), 10) == 1.0
    assert M.harrell_c(time, status, np.array([1., 2., 3., 4.]), 10) == 0.0


def test_perfect_calibration_gives_unit_slope():
    x, time, status = _toy(seed=5, n=500)
    fg = fit_finegray(x, time, status)
    p = fg.predict_cif(x, 1.0)
    slope, _ = M.calibration_slope(time, status, p)
    assert abs(slope - 1) < 1e-6


# ---------------------------------------------------------------- frozen cohort
def test_frozen_cohort_counts():
    f = DERIVED_DIR / "analysis_cohort.csv"
    if _skip_if_missing(f):
        return
    d = pd.read_csv(f)
    assert len(d) == 1057
    assert (d.status == 1).sum() == 90 and (d.status == 2).sum() == 68
    assert d[["age", "male", "egfr", "uacr_mg_g", "time_days", "status"]].notna().all().all()
    assert d.egfr.max() < 60 and d.uacr_mg_g.min() >= 1


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"{len(tests)} tests passed")
