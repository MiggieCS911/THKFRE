"""Check the Python Fine-Gray implementation against R output.

Reference values: R 4.3, survival::coxph() and survival::finegray() + coxph(weights = fgwt),
run on the raw (uncleaned) analysis file with the same data preparation as this test
(age/10, sex, eGFR/5, ln UACR, centred; death as the competing event).

Run:  python -m pytest tests/      (or)   python tests/test_finegray.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thkfre.config import RAW_DATA          # noqa: E402
from thkfre.finegray import fit_finegray, fit_cox    # noqa: E402

R_FINEGRAY_BETA = np.array([-0.100458205567105,   # age / 10
                            0.342005782912661,    # male
                            -0.740823304465269,   # eGFR / 5
                            0.785703433384327])   # ln(UACR)


R_COX_BETA = np.array([-0.0429895526140158, 0.471810847710451,
                       -0.74194417037758, 0.813736545512384])   # coxph, death censored


def _raw_design():
    df = pd.read_excel(RAW_DATA)
    X = np.column_stack([df.age / 10, df.gender, df.egfr_new / 5, np.log(df.UACR_true)])
    return df, X - X.mean(axis=0)


def test_cox_matches_r():
    df, X = _raw_design()
    fit = fit_cox(X, df.time_to_kidneyfailure.to_numpy(float), df.outcome.to_numpy(int))
    diff = np.abs(fit.beta - R_COX_BETA)
    print("Cox  max |difference| vs R =", diff.max())
    assert diff.max() < 1e-6


def test_matches_r():
    df = pd.read_excel(RAW_DATA)
    X = np.column_stack([df.age / 10, df.gender, df.egfr_new / 5, np.log(df.UACR_true)])
    X = X - X.mean(axis=0)                      # centring does not change beta
    status = np.where(df.outcome == 1, 1, np.where(df.Dead == 1, 2, 0))
    fit = fit_finegray(X, df.time_to_kidneyfailure.to_numpy(float), status)
    diff = np.abs(fit.beta - R_FINEGRAY_BETA)
    print("Python:", np.round(fit.beta, 6))
    print("R     :", np.round(R_FINEGRAY_BETA, 6))
    print("max |difference| =", diff.max())
    assert diff.max() < 1e-6


if __name__ == "__main__":
    test_cox_matches_r()
    test_matches_r()
    print("PASS")
