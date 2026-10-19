"""Model fitting and prediction for the updating ladder (shared by steps 4 and 5).

M0a/M0b published KFRE; M1 recalibrated baseline; M2 recalibrated baseline + slope
(Fine-Gray on KFRE LP); M3a refit Cox (death censored); M3b refit Fine-Gray.
"""
import numpy as np
from scipy.optimize import brentq

from thkfre.config import (HORIZON_DAYS as H, PREDICTORS,
                           KFRE_S0_2Y_NORTH_AMERICA, KFRE_S0_2Y_NON_NORTH_AMERICA)
from thkfre.kfre import linear_predictor, risk_from_lp
from thkfre.finegray import fit_finegray, fit_cox
from thkfre import metrics as M

MODEL_NAMES = ["M0a KFRE-NA", "M0b KFRE-nonNA", "M1 Recal. baseline",
               "M2 Recal. baseline+slope", "M3a Refit Cox", "M3b Refit Fine-Gray"]
UPDATED_MODELS = MODEL_NAMES[2:]


def fit_models(df):
    """Fit M1, M2, M3a, M3b on `df`. Returns dict of model specifications (pure numbers),
    so the same function can be re-run inside bootstrap resamples (step 5)."""
    t = df.time_days.to_numpy(float)
    s = df.status.to_numpy(int)
    lp = linear_predictor(df)
    observed = M.aalen_johansen(t, s, H)

    # M1: Thai baseline so that mean predicted = observed
    s0_m1 = brentq(lambda s0: np.mean(risk_from_lp(lp, s0)) - observed, 0.5, 0.999999)

    # M2: Fine-Gray on LP
    fg2 = fit_finegray(lp, t, s, names=["KFRE_LP"])
    b = float(fg2.beta[0])
    s0_m2 = float(np.exp(-fg2.cumhaz_at(H)))

    # M3: refit, centred on cohort means
    means = {v: float(df[v].mean()) for v in PREDICTORS}
    X = np.column_stack([df[v].to_numpy(float) - means[v] for v in PREDICTORS])
    cox = fit_cox(X, t, (s == 1).astype(int), names=PREDICTORS)
    fg = fit_finegray(X, t, s, names=PREDICTORS)
    return {
        "M1": {"s0": s0_m1},
        "M2": {"slope": b, "slope_se": float(fg2.se[0]), "s0": s0_m2},
        "M3a": {"beta": dict(zip(PREDICTORS, cox.beta)), "se": dict(zip(PREDICTORS, cox.se)),
                "means": means, "s0": float(np.exp(-cox.cumhaz_at(H)))},
        "M3b": {"beta": dict(zip(PREDICTORS, fg.beta)), "se": dict(zip(PREDICTORS, fg.se)),
                "means": means, "s0": float(np.exp(-fg.cumhaz_at(H)))},
        "_fits": {"fg2": fg2, "cox": cox, "fg": fg, "X": X},
    }


def predict_all(spec, df):
    """2-year risk from every model for the patients in df."""
    lp = linear_predictor(df)
    out = {
        "M0a KFRE-NA": risk_from_lp(lp, KFRE_S0_2Y_NORTH_AMERICA),
        "M0b KFRE-nonNA": risk_from_lp(lp, KFRE_S0_2Y_NON_NORTH_AMERICA),
        "M1 Recal. baseline": risk_from_lp(lp, spec["M1"]["s0"]),
        "M2 Recal. baseline+slope": risk_from_lp(spec["M2"]["slope"] * lp, spec["M2"]["s0"]),
    }
    for key, name in [("M3a", "M3a Refit Cox"), ("M3b", "M3b Refit Fine-Gray")]:
        m = spec[key]
        lp_new = sum(m["beta"][v] * (df[v].to_numpy(float) - m["means"][v]) for v in PREDICTORS)
        out[name] = risk_from_lp(lp_new, m["s0"])
    return out
