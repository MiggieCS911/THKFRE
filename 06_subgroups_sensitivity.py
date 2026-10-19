"""Step 7 - Subgroup and sensitivity analyses.

Models: KFRE-NA (S0 0.9750), KFRE-nonNA (S0 0.9832), TH-KFRE v1.0 (locked).

A. Subgroups (pre-specified): age <65 / >=65 (post hoc: 65-74 / >=75); female / male; eGFR >=30 / <30;
   UACR <300 / >=300 mg/g.
   Reported: n, KF events, deaths, Harrell C, AUC(2y), O/E with 95% bootstrap CI
   (1,000 resamples), calibration slope with model-based 95% CI.
   Slopes are not reported when a subgroup has < 20 KF events (unstable).
   TH-KFRE was derived in the full cohort, so its subgroup results are apparent.

B. Sensitivity analyses
   B1. Death treated as censoring instead of a competing event: observed risk by
       Kaplan-Meier (1 - S) instead of Aalen-Johansen; O/E recomputed (overall and by age).
   B2. Excluding the 10 patients whose UACR was corrected in step 1.
   B3. Excluding the patient with the extreme UACR (87,788.8 mg/g).
"""
import numpy as np
import pandas as pd

from thkfre.config import (DERIVED_DIR, TABLE_DIR, LOG_DIR, HORIZON_DAYS, RANDOM_SEED, N_BOOTSTRAP,
                           KFRE_S0_2Y_NORTH_AMERICA, KFRE_S0_2Y_NON_NORTH_AMERICA)
from thkfre.kfre import kfre_risk, thkfre_risk
from thkfre import metrics as M

H = HORIZON_DAYS
MIN_EVENTS_SLOPE = 20
log_lines = []


def log(msg=""):
    print(msg)
    log_lines.append(str(msg))


def predictions(df):
    return {"KFRE-NA": kfre_risk(df, KFRE_S0_2Y_NORTH_AMERICA),
            "KFRE-nonNA": kfre_risk(df, KFRE_S0_2Y_NON_NORTH_AMERICA),
            "TH-KFRE v1.0": thkfre_risk(df)}


def km_risk(time, status, horizon, event_code=1):
    """1 - Kaplan-Meier at horizon with all other outcomes censored."""
    e = (np.asarray(status) == event_code).astype(int)
    return M.aalen_johansen(time, e, horizon, 1)   # AJ with a single event type = 1 - KM


def summarise(df, label, rng, with_ci=True):
    t, s = df.time_days.to_numpy(float), df.status.to_numpy(int)
    n_ev = int((s == 1).sum())
    out = []
    for name, p in predictions(df).items():
        rec = {"subgroup": label, "model": name, "n": len(df), "KF_events": n_ev,
               "deaths": int((s == 2).sum()), "observed_AJ": M.aalen_johansen(t, s, H),
               "mean_predicted": float(p.mean()),
               "Harrell_C": M.harrell_c(t, s, p, H), "AUC_2y": M.auc_at_horizon(t, s, p, H),
               "O_E": M.observed_expected(t, s, p, H)}
        if with_ci:
            oe = []
            for _ in range(N_BOOTSTRAP):
                i = rng.integers(0, len(df), len(df))
                oe.append(M.observed_expected(t[i], s[i], p[i], H))
            rec["O_E_lo"], rec["O_E_hi"] = np.percentile(oe, [2.5, 97.5])
        if n_ev >= MIN_EVENTS_SLOPE:
            b, se = M.calibration_slope(t, s, p)
            rec.update({"cal_slope": b, "cal_slope_lo": b - 1.96 * se, "cal_slope_hi": b + 1.96 * se})
        else:
            rec.update({"cal_slope": np.nan, "cal_slope_lo": np.nan, "cal_slope_hi": np.nan})
        out.append(rec)
    return out


def fmt(r):
    slope = (f"slope {r.cal_slope:.2f} ({r.cal_slope_lo:.2f}-{r.cal_slope_hi:.2f})"
             if pd.notna(r.cal_slope) else f"slope n/a (<{MIN_EVENTS_SLOPE} events)")
    return (f"    {r.model:13s} C {r.Harrell_C:.3f}  AUC {r.AUC_2y:.3f}  "
            f"O/E {r.O_E:.2f} ({r.O_E_lo:.2f}-{r.O_E_hi:.2f})  {slope}")


def main():
    df = pd.read_csv(DERIVED_DIR / "analysis_cohort.csv")
    rng = np.random.default_rng(RANDOM_SEED)

    # ---------------- A. subgroups
    groups = [
        ("Age <65", df.age < 65), ("Age >=65", df.age >= 65),
        ("Age 65-74", df.age.between(65, 74)), ("Age >=75", df.age >= 75),
        ("Female", df.male == 0), ("Male", df.male == 1),
        ("eGFR >=30", df.egfr >= 30), ("eGFR <30", df.egfr < 30),
        ("UACR <300", df.uacr_mg_g < 300), ("UACR >=300", df.uacr_mg_g >= 300),
    ]
    rows = []
    for label, mask in groups:
        rows += summarise(df[mask].reset_index(drop=True), label, rng)
    sub = pd.DataFrame(rows)
    sub.to_csv(TABLE_DIR / "06_subgroups.csv", index=False)
    log("A. SUBGROUPS")
    for label, _ in groups:
        g = sub[sub.subgroup == label]
        r0 = g.iloc[0]
        log(f"  {label}: n={r0.n}, KF events={r0.KF_events}, deaths={r0.deaths}, "
            f"observed 2-y KF={r0.observed_AJ:.3f}")
        for _, r in g.iterrows():
            log(fmt(r))

    # ---------------- B1. death censored vs competing
    log("")
    log("B1. OBSERVED RISK: death as competing event (Aalen-Johansen) vs censored (1 - Kaplan-Meier)")
    b1 = []
    for label, mask in [("All", df.age >= 0), ("Age <65", df.age < 65), ("Age 65-74", df.age.between(65, 74)),
                        ("Age >=75", df.age >= 75)]:
        d = df[mask]
        t, s = d.time_days.to_numpy(float), d.status.to_numpy(int)
        aj, km = M.aalen_johansen(t, s, H), km_risk(t, s, H)
        p = predictions(d)
        rec = {"group": label, "n": len(d), "KF_events": int((s == 1).sum()), "deaths": int((s == 2).sum()),
               "observed_AJ": aj, "observed_KM": km, "KM_minus_AJ": km - aj}
        for name, pr in p.items():
            rec[f"O_E_AJ {name}"] = aj / pr.mean()
            rec[f"O_E_KM {name}"] = km / pr.mean()
        b1.append(rec)
        log(f"  {label:10s} n={len(d):4d} deaths={rec['deaths']:3d}  AJ {aj:.4f}  KM {km:.4f}  "
            f"(KM overstates by {km - aj:.4f});  TH-KFRE O/E: AJ {rec['O_E_AJ TH-KFRE v1.0']:.2f} vs "
            f"KM {rec['O_E_KM TH-KFRE v1.0']:.2f}")
    pd.DataFrame(b1).to_csv(TABLE_DIR / "06_sensitivity_competing_risk.csv", index=False)

    # ---------------- B2 / B3. data-correction sensitivity
    log("")
    log("B2/B3. DATA SENSITIVITY")
    sens = []
    for label, d in [("Main analysis", df),
                     ("B2: exclude UACR-corrected (n=10)", df[df.uacr_corrected == 0]),
                     ("B3: exclude extreme UACR 87,788.8", df[df.uacr_mg_g < 50000])]:
        sens += summarise(d.reset_index(drop=True), label, rng)
    sens = pd.DataFrame(sens)
    sens.to_csv(TABLE_DIR / "06_sensitivity_data.csv", index=False)
    for label in sens.subgroup.unique():
        g = sens[sens.subgroup == label]
        log(f"  {label}: n={g.iloc[0].n}, KF events={g.iloc[0].KF_events}")
        for _, r in g.iterrows():
            log(fmt(r))

    (LOG_DIR / "06_subgroups_sensitivity_log.txt").write_text("\n".join(log_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
