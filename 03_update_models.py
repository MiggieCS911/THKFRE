"""Step 4 - Updating the KFRE: closed-testing ladder (full cohort, apparent performance).

Models (all predict 2-year risk of kidney failure):
  M0a  KFRE-NA       published, S0 = 0.9750
  M0b  KFRE-nonNA    published, S0 = 0.9832
  M1   Recalibration-in-the-large: KFRE linear predictor (LP) unchanged, Thai baseline
       S0 chosen so that mean predicted risk = observed Aalen-Johansen 2-y incidence
  M2   Recalibration of intercept + slope: Fine-Gray model with the KFRE LP as the only
       covariate -> risk = 1 - S0 ^ exp(b * LP)
  M3a  Full refit, cause-specific Cox (death censored; same approach as the original KFRE)
  M3b  Full refit, Fine-Gray (death = competing event)
Refit predictors: age/10, male, eGFR/5, ln(UACR mg/g), centred on the Thai cohort means.

Likelihood-ratio tests (Fine-Gray partial likelihood):
  slope test   : M2 (b estimated)  vs  b = 1            (1 df)
  refit test   : M3b (4 coefficients) vs M2 (b * LP)    (3 df)
Heuristic shrinkage (van Houwelingen): (LR chi2 - df) / LR chi2 for the refitted models.

Apparent performance only; optimism correction is done in step 5 (04_bootstrap_validation.py).
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thkfre.config import DERIVED_DIR, TABLE_DIR, FIG_DIR, LOG_DIR, HORIZON_DAYS, PREDICTORS
from thkfre.kfre import linear_predictor
from thkfre.finegray import loglik_at
from thkfre.models import fit_models, predict_all
from thkfre import metrics as M

H = HORIZON_DAYS
LABELS = {"age10": "Age (per 10 years)", "male": "Male sex",
          "egfr5": "eGFR (per 5 mL/min/1.73 m2)", "ln_uacr": "ln UACR (mg/g)"}
log_lines = []


def log(msg=""):
    print(msg)
    log_lines.append(str(msg))


def main():
    df = pd.read_csv(DERIVED_DIR / "analysis_cohort.csv")
    t = df.time_days.to_numpy(float)
    s = df.status.to_numpy(int)
    spec = fit_models(df)
    fits = spec.pop("_fits")
    lp = linear_predictor(df)

    # ---------------- likelihood-ratio tests
    ll_b1 = loglik_at([1.0], lp, t, s)
    ll_m2 = fits["fg2"].loglik
    ll_m3 = fits["fg"].loglik
    ll_null = loglik_at(np.zeros(4), fits["X"], t, s)
    lr_slope = 2 * (ll_m2 - ll_b1)
    lr_refit = 2 * (ll_m3 - ll_m2)
    lr_fg_null = 2 * (ll_m3 - ll_null)
    ll_cox_null = loglik_at(np.zeros(4), fits["X"], t, (s == 1).astype(int))
    lr_cox_null = 2 * (fits["cox"].loglik - ll_cox_null)
    tests = pd.DataFrame([
        {"test": "Slope: M2 vs b=1", "LR_chi2": lr_slope, "df": 1, "p": chi2.sf(lr_slope, 1)},
        {"test": "Refit: M3b vs M2", "LR_chi2": lr_refit, "df": 3, "p": chi2.sf(lr_refit, 3)},
    ])
    tests.to_csv(TABLE_DIR / "03_lr_tests.csv", index=False)
    shrink = {"M3a Cox": (lr_cox_null - 4) / lr_cox_null, "M3b Fine-Gray": (lr_fg_null - 4) / lr_fg_null}

    log("UPDATED MODEL SPECIFICATIONS")
    log(f"  M1  Thai baseline S0(2y) = {spec['M1']['s0']:.5f}")
    log(f"  M2  slope b = {spec['M2']['slope']:.4f} (SE {spec['M2']['slope_se']:.4f}); S0(2y) = {spec['M2']['s0']:.5f}")
    for key in ("M3a", "M3b"):
        m = spec[key]
        log(f"  {key} S0(2y) = {m['s0']:.5f}; means " + ", ".join(f"{v}={m['means'][v]:.4f}" for v in PREDICTORS))
    log("")
    log("LIKELIHOOD-RATIO TESTS (Fine-Gray partial likelihood)")
    log(tests.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    log(f"Heuristic shrinkage: " + ", ".join(f"{k} = {v:.3f}" for k, v in shrink.items()))

    # ---------------- sample size check (Riley et al. BMJ 2020;368:m441), criteria 1 and 3
    n = len(df)
    r2_cs = 1 - np.exp(-lr_cox_null / n)                 # Cox-Snell R2 of the refitted Cox model
    phi = (s == 1).mean()
    log("")
    log(f"SAMPLE SIZE (Riley 2020): n = {n}, events = {(s == 1).sum()}, outcome proportion = {phi:.3f}, "
        f"Cox-Snell R2 = {r2_cs:.3f}")
    for p_, lab in [(4, "full refit (4 parameters)"), (2, "recalibration (2 parameters)")]:
        need = p_ / ((0.9 - 1) * np.log(1 - r2_cs / 0.9))
        log(f"  criterion 1 (expected shrinkage >= 0.90), {lab}: n required = {np.ceil(need):.0f}")
    need3 = (1.96 / 0.05) ** 2 * phi * (1 - phi)
    log(f"  criterion 3 (2-y risk estimated within +/-0.05): n required = {np.ceil(need3):.0f}")

    # ---------------- coefficient table (refit models vs published KFRE)
    from thkfre.config import KFRE_BETA
    coef_rows = []
    for v in PREDICTORS:
        row = {"predictor": LABELS[v], "KFRE_beta": KFRE_BETA[v], "KFRE_HR": np.exp(KFRE_BETA[v])}
        for key, lab in [("M3a", "Cox"), ("M3b", "FG")]:
            b_, se_ = spec[key]["beta"][v], spec[key]["se"][v]
            row.update({f"{lab}_beta": b_, f"{lab}_SE": se_, f"{lab}_HR": np.exp(b_),
                        f"{lab}_HR_lo": np.exp(b_ - 1.96 * se_), f"{lab}_HR_hi": np.exp(b_ + 1.96 * se_),
                        f"{lab}_p": 2 * norm.sf(abs(b_ / se_))})
        coef_rows.append(row)
    coefs = pd.DataFrame(coef_rows)
    coefs.to_csv(TABLE_DIR / "03_coefficients.csv", index=False)
    log("")
    log("COEFFICIENTS (HR, 95% CI, p)")
    for _, r in coefs.iterrows():
        log(f"  {r.predictor:30s} KFRE {r.KFRE_HR:.3f} | Cox {r.Cox_HR:.3f} ({r.Cox_HR_lo:.3f}-{r.Cox_HR_hi:.3f}) p={r.Cox_p:.3g}"
            f" | FG {r.FG_HR:.3f} ({r.FG_HR_lo:.3f}-{r.FG_HR_hi:.3f}) p={r.FG_p:.3g}")

    # ---------------- apparent performance
    preds = predict_all(spec, df)
    perf = pd.DataFrame([{"model": k, **M.performance_summary(t, s, p, H)} for k, p in preds.items()])
    perf.to_csv(TABLE_DIR / "03_apparent_performance.csv", index=False)
    log("")
    log(f"APPARENT PERFORMANCE (observed AJ 2-y incidence = {M.aalen_johansen(t, s, H):.4f})")
    log(perf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---------------- grouped calibration for all models
    cuts = [0, 0.02, 0.06, 0.10, 0.20, 0.40, np.inf]
    grp = pd.DataFrame([{"model": k, **row} for k, p in preds.items()
                        for row in M.grouped_calibration(t, s, p, H, cuts)])
    grp.to_csv(TABLE_DIR / "03_grouped_calibration.csv", index=False)

    # ---------------- save specification (equations) and predictions
    (TABLE_DIR / "03_model_specifications.json").write_text(json.dumps(spec, indent=2))
    out = df[["study_id", "time_days", "status"]].copy()
    for k, p in preds.items():
        out[k] = p
    out.to_csv(DERIVED_DIR / "03_predictions.csv", index=False)

    # ---------------- draft figure
    fig, ax = plt.subplots(2, 3, figsize=(13, 8.5), sharex=True, sharey=True)
    for a, (name, p) in zip(ax.ravel(), preds.items()):
        g, o, cal = M.flexible_calibration(t, s, p, H)
        gg = grp[grp.model == name]
        a.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
        a.plot(g, o, lw=2, c="#2471a3")
        a.scatter(gg.predicted, gg.observed_AJ, s=22, c="#2471a3", zorder=3)
        a.set_title(f"{name}\nO/E {perf.set_index('model').loc[name, 'O_E']:.2f}, ICI {cal['ICI']:.3f}", fontsize=9)
        a.set_xlim(0, 1); a.set_ylim(0, 1)
    for a in ax[1]:
        a.set_xlabel("Predicted 2-year risk")
    for a in ax[:, 0]:
        a.set_ylabel("Observed 2-year cumulative incidence")
    fig.suptitle("DRAFT - apparent calibration of original and updated models", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "draft_03_calibration_all_models.png", dpi=150)
    (LOG_DIR / "03_update_log.txt").write_text("\n".join(log_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
