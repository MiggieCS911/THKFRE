"""Step 6 - Clinical usefulness of TH-KFRE v1.0 vs the published KFRE.

Models compared (all 2-year risk, death as competing event):
  KFRE-NA      published, S0 = 0.9750
  KFRE-nonNA   published, S0 = 0.9832
  TH-KFRE v1.0 locked equation in thkfre/config.py (1 - 0.99464 ^ exp(1.343 * KFRE LP))

A. Threshold performance at 2-year risk >= 10% (multidisciplinary CKD care),
   >= 20% and >= 40% (KRT preparation / vascular access):
   n flagged, sensitivity, specificity, PPV, NPV, number needed to refer (1/PPV),
   95% percentile bootstrap CIs (1,000 resamples).
B. Risk categories (<10%, 10-<40%, >=40%): predicted vs observed, and reclassification
   of patients between KFRE-nonNA and TH-KFRE.
C. Decision curve analysis (net benefit, thresholds 1-50%) with bootstrap CIs for the
   difference in net benefit (TH-KFRE minus KFRE-nonNA) at key thresholds.

Note: TH-KFRE was derived in this cohort; its performance here is apparent performance
(optimism was shown to be negligible in step 5).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thkfre.config import (DERIVED_DIR, TABLE_DIR, FIG_DIR, LOG_DIR, HORIZON_DAYS, RANDOM_SEED,
                           N_BOOTSTRAP, KFRE_S0_2Y_NORTH_AMERICA, KFRE_S0_2Y_NON_NORTH_AMERICA)
from thkfre.kfre import kfre_risk, thkfre_risk
from thkfre import metrics as M

H = HORIZON_DAYS
CUTOFFS = [0.05, 0.10, 0.20, 0.40]   # 5% is exploratory (sensitivity-matching)
NB_KEY = [0.05, 0.10, 0.20, 0.40]
CATS = [0, 0.10, 0.40, np.inf]
CAT_LABELS = ["<10%", "10-<40%", ">=40%"]
log_lines = []


def log(msg=""):
    print(msg)
    log_lines.append(str(msg))


def main():
    df = pd.read_csv(DERIVED_DIR / "analysis_cohort.csv")
    t, s = df.time_days.to_numpy(float), df.status.to_numpy(int)
    n = len(df)
    preds = {"KFRE-NA": kfre_risk(df, KFRE_S0_2Y_NORTH_AMERICA),
             "KFRE-nonNA": kfre_risk(df, KFRE_S0_2Y_NON_NORTH_AMERICA),
             "TH-KFRE v1.0": thkfre_risk(df)}
    rng = np.random.default_rng(RANDOM_SEED)
    boots = [rng.integers(0, n, n) for _ in range(N_BOOTSTRAP)]

    # ---------------- A. threshold performance
    keys = ["sensitivity", "specificity", "PPV", "NPV", "NNR", "pct_flagged"]
    rows = []
    for name, p in preds.items():
        for c in CUTOFFS:
            point = M.threshold_metrics(t, s, p, c, H)
            bt = pd.DataFrame([M.threshold_metrics(t[i], s[i], p[i], c, H) for i in boots])
            rec = {"model": name, "threshold": c, **point}
            for k in keys:
                rec[k + "_lo"], rec[k + "_hi"] = np.percentile(bt[k], [2.5, 97.5])
            rows.append(rec)
        print(f"  thresholds done: {name}")
    thr = pd.DataFrame(rows)
    thr.to_csv(TABLE_DIR / "05_threshold_performance.csv", index=False)
    log("A. THRESHOLD PERFORMANCE (95% bootstrap CI)")
    for _, r in thr.iterrows():
        log(f"  {r.model:13s} >= {r.threshold:.0%}: flagged {r.n_flagged} ({r.pct_flagged:.1%}); "
            f"events captured {r.events_captured}/{r.events_total}")
        log(f"      Sens {r.sensitivity:.3f} ({r.sensitivity_lo:.3f}-{r.sensitivity_hi:.3f})  "
            f"Spec {r.specificity:.3f} ({r.specificity_lo:.3f}-{r.specificity_hi:.3f})  "
            f"PPV {r.PPV:.3f} ({r.PPV_lo:.3f}-{r.PPV_hi:.3f})  "
            f"NPV {r.NPV:.3f} ({r.NPV_lo:.3f}-{r.NPV_hi:.3f})  NNR {r.NNR:.2f} ({r.NNR_lo:.2f}-{r.NNR_hi:.2f})")

    # ---------------- B. risk categories and reclassification
    cat = {k: pd.cut(p, CATS, right=False, labels=CAT_LABELS) for k, p in preds.items()}
    cat_rows = []
    for name, p in preds.items():
        for lab in CAT_LABELS:
            m = (cat[name] == lab)
            cat_rows.append({"model": name, "category": lab, "n": int(m.sum()),
                             "events": int(((s == 1) & m).sum()),
                             "mean_predicted": float(p[m].mean()) if m.any() else np.nan,
                             "observed_AJ": M.aalen_johansen(t[m], s[m], H) if m.any() else np.nan})
    cats = pd.DataFrame(cat_rows)
    cats.to_csv(TABLE_DIR / "05_risk_categories.csv", index=False)
    log("")
    log("B. RISK CATEGORIES (predicted vs observed 2-y incidence)")
    log(cats.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    recl_rows = []
    for ref in ["KFRE-nonNA", "KFRE-NA"]:
        for a in CAT_LABELS:
            for b in CAT_LABELS:
                m = (cat[ref] == a) & (cat["TH-KFRE v1.0"] == b)
                recl_rows.append({"reference": ref, "reference_category": a, "THKFRE_category": b,
                                  "n": int(m.sum()), "events": int(((s == 1) & m).sum()),
                                  "observed_AJ": M.aalen_johansen(t[m], s[m], H) if m.any() else np.nan})
    recl = pd.DataFrame(recl_rows)
    recl.to_csv(TABLE_DIR / "05_reclassification.csv", index=False)
    log("")
    log("RECLASSIFICATION (rows: published KFRE category; columns: TH-KFRE category; n [events])")
    for ref in ["KFRE-nonNA", "KFRE-NA"]:
        r = recl[recl.reference == ref]
        tab = r.assign(cell=r.n.astype(str) + " [" + r.events.astype(str) + "]").pivot(
            index="reference_category", columns="THKFRE_category", values="cell").loc[CAT_LABELS, CAT_LABELS]
        log(f"  vs {ref}:")
        log("    " + tab.to_string().replace("\n", "\n    "))
        moved = r[r.reference_category != r.THKFRE_category]
        log(f"    patients changing category: {moved.n.sum()} ({moved.n.sum() / n:.1%}); "
            f"events among them: {moved.events.sum()}")
    down = (preds["KFRE-nonNA"] >= 0.10) & (preds["TH-KFRE v1.0"] < 0.10)
    log(f"  Moved from >=10% (KFRE-nonNA) to <10% (TH-KFRE): n = {down.sum()}, "
        f"TH-KFRE mean predicted {preds['TH-KFRE v1.0'][down].mean():.3f}, "
        f"KFRE-nonNA mean predicted {preds['KFRE-nonNA'][down].mean():.3f}, "
        f"observed {M.aalen_johansen(t[down], s[down], H):.3f} ({((s == 1) & down).sum()} events)")

    # ---------------- C. decision curves
    grid = np.round(np.arange(0.01, 0.501, 0.01), 2)
    dca = pd.DataFrame({"threshold": grid,
                        "treat_all": [M.net_benefit_treat_all(t, s, pt, H) for pt in grid],
                        "treat_none": 0.0})
    for name, p in preds.items():
        dca[name] = [M.net_benefit(t, s, p, pt, H) for pt in grid]
    dca.to_csv(TABLE_DIR / "05_decision_curve.csv", index=False)

    diff_rows = []
    for pt in NB_KEY:
        d_point = M.net_benefit(t, s, preds["TH-KFRE v1.0"], pt, H) - M.net_benefit(t, s, preds["KFRE-nonNA"], pt, H)
        d_boot = [M.net_benefit(t[i], s[i], preds["TH-KFRE v1.0"][i], pt, H)
                  - M.net_benefit(t[i], s[i], preds["KFRE-nonNA"][i], pt, H) for i in boots]
        lo, hi = np.percentile(d_boot, [2.5, 97.5])
        diff_rows.append({"threshold": pt, "NB_THKFRE": M.net_benefit(t, s, preds["TH-KFRE v1.0"], pt, H),
                          "NB_KFRE_nonNA": M.net_benefit(t, s, preds["KFRE-nonNA"], pt, H),
                          "NB_KFRE_NA": M.net_benefit(t, s, preds["KFRE-NA"], pt, H),
                          "NB_treat_all": M.net_benefit_treat_all(t, s, pt, H),
                          "diff_THKFRE_minus_nonNA": d_point, "diff_lo": lo, "diff_hi": hi,
                          "net_extra_true_positives_per_1000": 1000 * d_point})
    diffs = pd.DataFrame(diff_rows)
    diffs.to_csv(TABLE_DIR / "05_net_benefit_key_thresholds.csv", index=False)
    log("")
    log("C. NET BENEFIT at key thresholds (difference TH-KFRE minus KFRE-nonNA, 95% bootstrap CI)")
    log(diffs.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # useful range: thresholds where each model beats both default strategies
    for name in preds:
        ok = dca[(dca[name] > dca.treat_all) & (dca[name] > 0)].threshold
        log(f"  {name}: net benefit > treat-all and > treat-none for thresholds "
            f"{ok.min():.0%}-{ok.max():.0%}" if len(ok) else f"  {name}: none")

    # ---------------- draft figure
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(dca.threshold, dca.treat_all, c="grey", lw=1, label="Treat all")
    ax.axhline(0, c="black", lw=1, label="Treat none")
    for (name, col, ls) in [("KFRE-NA", "#c0392b", "--"), ("KFRE-nonNA", "#7d3c98", ":"),
                            ("TH-KFRE v1.0", "#2471a3", "-")]:
        ax.plot(dca.threshold, dca[name], c=col, ls=ls, lw=2, label=name)
    ax.set_ylim(-0.01, 0.09); ax.set_xlim(0, 0.5)
    ax.set_xlabel("Threshold 2-year risk"); ax.set_ylabel("Net benefit")
    ax.legend(); ax.set_title("DRAFT - decision curves (death as competing event)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "draft_05_decision_curve.png", dpi=150)
    (LOG_DIR / "05_clinical_utility_log.txt").write_text("\n".join(log_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
