"""Published 4-variable KFRE and simple updating helpers."""
import numpy as np

from thkfre.config import KFRE_BETA, KFRE_MEANS, PREDICTORS


def linear_predictor(df, beta=KFRE_BETA, means=KFRE_MEANS):
    """Centred linear predictor. `df` needs columns age10, male, egfr5, ln_uacr."""
    return sum(beta[v] * (df[v].to_numpy(float) - means[v]) for v in PREDICTORS)


def risk_from_lp(lp, s0):
    """2-year risk = 1 - S0 ** exp(LP)."""
    return 1.0 - np.power(s0, np.exp(lp))


def kfre_risk(df, s0):
    return risk_from_lp(linear_predictor(df), s0)


def cloglog(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(-np.log(1.0 - p))


def thkfre_risk(df):
    """TH-KFRE v1.0 2-year risk (locked equation; see config.py)."""
    from thkfre.config import THKFRE_SLOPE, THKFRE_S0_2Y
    return risk_from_lp(THKFRE_SLOPE * linear_predictor(df), THKFRE_S0_2Y)
