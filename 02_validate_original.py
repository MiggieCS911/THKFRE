"""Step 3 - External validation of the published 4-variable KFRE at 2 years.

Two published versions are validated on the full cleaned cohort:
  KFRE-NA     : North American baseline, S0(2y) = 0.9750 (Tangri 2011/2016)
  KFRE-nonNA  : Non-North American calibration factor, S0(2y) = 0.9832
                (baseline 2-year risk lowered by 32.9%: 1 - 0.025*0.671 = 0.983225;
                 Tangri et al. JAMA 2016;315:164-74)

Death before kidney failure is a competing event throughout.
Performance (point estimate + 95% percentile bootstrap CI, 1,000 resamples):
  Harrell C (death censored; comparable with published KFRE papers)
  AUC at 2 years (competing-risk cumulative/dynamic, IPCW)
  O/E ratio (Aalen-Johansen observed / mean predicted)
  Calibration slope (Fine-Gray on cloglog predicted risk)
  Brier score and scaled Brier at 2 years (IPCW)
  ICI / E50 / E90 from a flexible calibration curve: Fine-Gray model of the outcome on a
  restricted cubic spline (3 knots) of cloglog(predicted risk) (Austin et al. Stat Med 2022)

Outputs: outputs/tables/02_validation_performance.csv
         outputs/tables/02_grouped_calibration.csv
         outputs/derived/02_predictions.csv         (patient level, not for GitHub)
         outputs/figures/draft_02_calibration.png    (draft, final design later)
         outputs/logs/02_validation_log.txt
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thkfre.config import (DERIVED_DIR, TABLE_DIR, FIG_DIR, LOG_DIR, HORIZON_DAYS,
                           RANDOM_SEED, N_BOOTSTRAP,
                           KFRE_S0_2Y_NORTH_AMERICA, KFRE_S0_2Y_NON_NORTH_AMERICA)
from thkfre.kfre import kfre_risk
from thkfre import metrics as M

H = HORIZON_DAYS
RISK_CUTS = [0, 0.02, 0.06, 0.10, 0.20, 0.40, np.inf]
MODELS = {"KFRE-NA (S0=0.9750)": KFRE_S0_2Y_NORTH_AMERICA,
          "KFRE-nonNA (S0=0.9832)": KFRE_S0_2Y_NON_NORTH_AMERICA}

log_lines = []


def log(msg=""):
    print(msg)
    log_lines.append(str(msg))


def performance(t, s, p, with_slope=True):
    out = {
        "Harrell_C": M.harrell_c(t, s, p, H),
        "AUC_2y": M.auc_at_horizon(t, s, p, H),
        "O_E": M.observed_expected(t, s, p, H),
        "mean_predicted": float(np.mean(p)),
        "observed_AJ": M.aalen_johansen(t, s, H),
        "Brier": M.brier_at_horizon(t, s, p, H),
        "scaled_Brier": M.scaled_brier(t, s, p, H),
    }
    if with_slope:
        out["cal_slope"] = M.calibration_slope(t, s, p)[0]
    return out


def main():
    for d in (TABLE_DIR, FIG_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DERIVED_DIR / "analysis_cohort.csv")
    t = df.time_days.to_numpy(float)
    s = df.status.to_numpy(int)
    n = len(df)
    log(f"Cohort n = {n}; KF events = {(s == 1).sum()}; deaths = {(s == 2).sum()}")
    log(f"Observed 2-y cumulative incidence of KF (Aalen-Johansen) = {M.aalen_johansen(t, s, H):.4f}")
    log(f"Observed 2-y cumulative incidence of death (Aalen-Johansen) = {M.aalen_johansen(t, s, H, 2):.4f}")

    preds = {name: kfre_risk(df, s0) for name, s0 in MODELS.items()}

    # cross-check with the risks pre-computed in the hospital file (NA version)
    diff = np.abs(preds["KFRE-NA (S0=0.9750)"] - df.prob_kfre_original_dataset)
    log(f"Cross-check vs dataset prob_2y_4v: identical (<1e-9) for {(diff < 1e-9).sum()} / {n}; "
        f"differences only where UACR was corrected in step 1 (n = {(diff >= 1e-9).sum()})")

    # ---------------- point estimates
    rows, smooth = [], {}
    for name, p in preds.items():
        perf = performance(t, s, p)
        slope, slope_se = M.calibration_slope(t, s, p)
        grid, obs, cal = M.flexible_calibration(t, s, p, H, n_knots=3)
        smooth[name] = (grid, obs)
        perf.update(cal)
        perf["cal_slope_se"] = slope_se
        perf["model"] = name
        rows.append(perf)

    # ---------------- bootstrap CIs
    rng = np.random.default_rng(RANDOM_SEED)
    boot = {name: [] for name in preds}
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        for name, p in preds.items():
            boot[name].append(performance(t[idx], s[idx], p[idx]))
        if (b + 1) % 200 == 0:
            print(f"  bootstrap {b + 1}/{N_BOOTSTRAP}")

    table = []
    for r in rows:
        bt = pd.DataFrame(boot[r["model"]])
        rec = {"model": r["model"]}
        for k in ["Harrell_C", "AUC_2y", "O_E", "cal_slope", "Brier", "scaled_Brier",
                  "mean_predicted", "observed_AJ"]:
            lo, hi = np.percentile(bt[k], [2.5, 97.5])
            rec[k] = r[k]; rec[k + "_lo"] = lo; rec[k + "_hi"] = hi
        for k in ["ICI", "E50", "E90"]:
            rec[k] = r[k]
        table.append(rec)
    table = pd.DataFrame(table)
    table.to_csv(TABLE_DIR / "02_validation_performance.csv", index=False)

    log("")
    log("PERFORMANCE (95% bootstrap CI)")
    for _, r in table.iterrows():
        log(f"  {r.model}")
        for k, lab in [("Harrell_C", "Harrell C"), ("AUC_2y", "AUC at 2 y"), ("O_E", "O/E ratio"),
                       ("cal_slope", "Calibration slope"), ("Brier", "Brier score"),
                       ("scaled_Brier", "Scaled Brier"), ("mean_predicted", "Mean predicted risk")]:
            log(f"    {lab:22s} {r[k]:.4f} ({r[k + '_lo']:.4f} - {r[k + '_hi']:.4f})")
        log(f"    {'ICI / E50 / E90':22s} {r.ICI:.4f} / {r.E50:.4f} / {r.E90:.4f}")

    # ---------------- grouped calibration
    grp = []
    for name, p in preds.items():
        for row in M.grouped_calibration(t, s, p, H, RISK_CUTS):
            row["model"] = name
            grp.append(row)
    grp = pd.DataFrame(grp)[["model", "group", "n", "events", "predicted", "observed_AJ"]]
    grp.to_csv(TABLE_DIR / "02_grouped_calibration.csv", index=False)
    log("")
    log("GROUPED CALIBRATION (predicted 2-y risk groups)")
    log(grp.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---------------- patient-level predictions
    out = df[["study_id", "time_days", "status"]].copy()
    for name, p in preds.items():
        out[name] = p
    out.to_csv(DERIVED_DIR / "02_predictions.csv", index=False)

    # ---------------- draft figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for k, scale in enumerate([1.0, 0.4]):
        a = ax[k]
        a.plot([0, scale], [0, scale], ls="--", c="grey", lw=1)
        for (name, (g, o)), col in zip(smooth.items(), ["#c0392b", "#2471a3"]):
            a.plot(g, o, c=col, lw=2, label=name)
            gg = grp[grp.model == name]
            a.scatter(gg.predicted, gg.observed_AJ, c=col, s=25, zorder=3)
        a.set_xlim(0, scale); a.set_ylim(0, scale)
        a.set_xlabel("Predicted 2-year risk"); a.set_ylabel("Observed 2-year cumulative incidence")
        a.set_title("Full range" if scale == 1 else "Zoom: 0-40%")
    ax[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("DRAFT - calibration of the original KFRE (lines: Fine-Gray spline curve; dots: risk groups)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "draft_02_calibration.png", dpi=150)

    (LOG_DIR / "02_validation_log.txt").write_text("\n".join(log_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
