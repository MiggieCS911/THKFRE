"""Step 1 - Clean raw data and freeze the analysis cohort.

Input : ../data/validation_dataset_ThaiKFRE.xlsx  (raw hospital extract)
Output: outputs/derived/analysis_cohort.csv       (de-identified, one row per patient)
        outputs/logs/01_cleaning_log.txt          (every change, with counts)
        outputs/tables/flow_counts.csv            (numbers for the flow diagram)

Rules (agreed 16 Oct 2026):
  1. UACR recorded by the lab as "< 1"           -> 1 mg/g (detection limit)
  2. UACR missing but raw note "<value> (Repeated)" -> use that value
  3. No UACR result at all (no specimen / "-")    -> exclude
  4. Baseline eGFR >= 60                          -> exclude
UACR source column is `UACR_true` (mg/g).

Outcome coding (2-year horizon):
  status = 1  kidney failure (maintenance dialysis > 3 months or transplant)
  status = 2  death before kidney failure (competing event)
  status = 0  censored (alive without KRT at last follow-up / 730 days)
"""
import re
import sys

import numpy as np
import pandas as pd

from thkfre.config import (RAW_DATA, DERIVED_DIR, TABLE_DIR, LOG_DIR, HORIZON_DAYS,
                           EGFR_MAX_EXCLUSIVE, UACR_DETECTION_LIMIT_MG_G)

PLACEHOLDER = 1e-6   # raw file stores unknown/undetectable UACR as 1e-7

log_lines = []


def log(msg=""):
    print(msg)
    log_lines.append(msg)


def main():
    for d in (DERIVED_DIR, TABLE_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    raw = pd.read_excel(RAW_DATA)
    df = raw.copy()
    df["study_id"] = np.arange(1, len(df) + 1)
    log(f"Raw file: {RAW_DATA.name}  rows = {len(df)}")
    assert df["hn"].is_unique, "duplicate patients in raw file"
    flow = [("Patients in analysis file", len(df))]

    # ---------------- UACR repair -------------------------------------------
    raw_uacr = df["uacr"].astype(str).str.strip()
    raw_alt = df["uacrxxxx"].astype(str).str.strip()
    is_ph = df["UACR_true"] < PLACEHOLDER

    below_dl = is_ph & raw_uacr.str.startswith("<")
    df.loc[below_dl, "UACR_true"] = UACR_DETECTION_LIMIT_MG_G
    log(f"Rule 1a: UACR '< 1' set to {UACR_DETECTION_LIMIT_MG_G} mg/g: n = {below_dl.sum()} "
        f"(study_id {df.loc[below_dl, 'study_id'].tolist()})")
    floor = (~is_ph) & (df["UACR_true"] < UACR_DETECTION_LIMIT_MG_G)
    log(f"Rule 1b: recorded UACR below detection limit {df.loc[floor, 'UACR_true'].round(2).tolist()} "
        f"set to {UACR_DETECTION_LIMIT_MG_G} mg/g: n = {floor.sum()} "
        f"(study_id {df.loc[floor, 'study_id'].tolist()})")
    df.loc[floor, "UACR_true"] = UACR_DETECTION_LIMIT_MG_G

    repeated = is_ph & ~below_dl & raw_alt.str.contains("Repeated", case=False)
    for i in df.index[repeated]:
        value = float(re.match(r"[\d.]+", raw_alt[i]).group())
        df.loc[i, "UACR_true"] = value
        log(f"Rule 2: study_id {df.loc[i, 'study_id']}: UACR from note '{raw_alt[i]}' -> {value} mg/g")

    df["uacr_corrected"] = (below_dl | floor | repeated).astype(int)
    no_uacr = df["UACR_true"] < PLACEHOLDER
    log(f"Rule 3: no UACR result -> exclude n = {no_uacr.sum()} "
        f"(raw notes: {raw_uacr[no_uacr].tolist()} / {raw_alt[no_uacr].tolist()})")
    df = df[~no_uacr]
    flow.append(("Excluded: no UACR result", int(no_uacr.sum())))

    # ---------------- eGFR --------------------------------------------------
    high = df["egfr_new"] >= EGFR_MAX_EXCLUSIVE
    log(f"Rule 4: baseline eGFR >= {EGFR_MAX_EXCLUSIVE:g} -> exclude n = {high.sum()} "
        f"(eGFR {df.loc[high, 'egfr_new'].tolist()})")
    df = df[~high]
    flow.append((f"Excluded: eGFR >= {EGFR_MAX_EXCLUSIVE:g}", int(high.sum())))
    flow.append(("Analysis cohort", len(df)))

    # ---------------- outcome -----------------------------------------------
    time = df["time_to_kidneyfailure"].astype(float)
    days_to_event_date = (df["Date of event"] - df["vstdate"]).dt.days
    death_in_window = (df["Dead"] == 1) & (days_to_event_date <= HORIZON_DAYS)
    late_death = (df["Dead"] == 1) & ~death_in_window
    log(f"Deaths after day {HORIZON_DAYS} (alive and event-free at 2 y -> censored at {HORIZON_DAYS}): "
        f"n = {late_death.sum()} (days {days_to_event_date[late_death].tolist()})")
    status = np.select([df["outcome"] == 1, death_in_window], [1, 2], 0)
    assert ((df["outcome"] == 1) & (df["Dead"] == 1)).sum() == 0
    assert (time[death_in_window] == days_to_event_date[death_in_window]).all()
    assert time.max() <= HORIZON_DAYS and time.min() > 0
    # KRT events after 2 years are not events at the 2-year horizon
    late_krt = ((df["Event : No0,HD1,PD2,KT3"] > 0) & (df["outcome"] == 0)).sum()
    log(f"KRT after day {HORIZON_DAYS} (counted as no event at 2 y): n = {late_krt}")

    # ---------------- analysis file -----------------------------------------
    out = pd.DataFrame({
        "study_id": df["study_id"],
        "index_date": df["vstdate"].dt.date,
        "index_year": df["vstdate"].dt.year,
        "age": df["age"],
        "male": df["gender"],
        "egfr": df["egfr_new"],
        "uacr_mg_g": df["UACR_true"],
        "uacr_corrected": df["uacr_corrected"],     # 1 = changed by rules 1a/1b/2
        "creatinine": pd.to_numeric(df["cr"], errors="coerce"),
        "sodium": pd.to_numeric(df["na"], errors="coerce"),
        "potassium": pd.to_numeric(df["k"], errors="coerce"),
        "chloride": df["cl"], "bicarbonate": df["co2"],
        "calcium": df["ca"], "phosphate": df["po4"],
        "krt_modality": df["Event : No0,HD1,PD2,KT3"].where(status == 1, 0)
                         .map({0: "none", 1: "HD", 2: "PD", 3: "KT"}),
        "time_days": time,
        "status": status,
    })
    out["age10"] = out["age"] / 10
    out["egfr5"] = out["egfr"] / 5
    out["ln_uacr"] = np.log(out["uacr_mg_g"])
    out["prob_kfre_original_dataset"] = df["prob_2y_4v"]   # kept for cross-check only

    n = len(out)
    log("")
    log(f"FINAL COHORT n = {n}")
    log(f"  kidney failure within 2 y : {(out.status == 1).sum()} ({(out.status == 1).mean():.1%})")
    log(f"  death before KF           : {(out.status == 2).sum()} ({(out.status == 2).mean():.1%})")
    log(f"  censored before day {HORIZON_DAYS} : {((out.status == 0) & (out.time_days < HORIZON_DAYS)).sum()}")
    log(f"  UACR range (mg/g)         : {out.uacr_mg_g.min():g} - {out.uacr_mg_g.max():g}")
    log(f"  eGFR range                : {out.egfr.min():g} - {out.egfr.max():g}")
    log(f"  age range                 : {out.age.min()} - {out.age.max()}")
    assert out[["age", "male", "egfr", "uacr_mg_g", "time_days", "status"]].notna().all().all()

    out.to_csv(DERIVED_DIR / "analysis_cohort.csv", index=False)
    pd.DataFrame(flow, columns=["step", "n"]).to_csv(TABLE_DIR / "flow_counts.csv", index=False)
    (LOG_DIR / "01_cleaning_log.txt").write_text("\n".join(log_lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
