"""Step 5 - Internal validation: bootstrap optimism correction + temporal validation.

A. Bootstrap optimism correction (Harrell 1996; Steyerberg 2019, ch. 17)
   For b = 1..B: draw a bootstrap sample, repeat the WHOLE updating procedure of step 4
   (M1, M2, M3a, M3b), and compute performance
     - in the bootstrap sample (apparent_b), and
     - in the original cohort (test_b).
   optimism = mean(apparent_b - test_b);  corrected = apparent(original) - optimism.
   The published models (M0a, M0b) are not fitted, so they have no optimism.
   Also recorded per resample: LR test of refit vs recalibration (model-selection stability).

B. Temporal validation (sensitivity analysis)
   Develop all updated models on 2021 index visits, test on 2022 index visits.

Outputs: outputs/tables/04_optimism_corrected.csv
         outputs/tables/04_temporal_validation.csv
         outputs/tables/04_bootstrap_raw.csv        (every resample, for checking)
         outputs/logs/04_bootstrap_log.txt
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import chi2

from thkfre.config import DERIVED_DIR, TABLE_DIR, LOG_DIR, HORIZON_DAYS, RANDOM_SEED, N_BOOTSTRAP
from thkfre.models import fit_models, predict_all, MODEL_NAMES, UPDATED_MODELS
from thkfre import metrics as M

H = HORIZON_DAYS
MEASURES = ["Harrell_C", "AUC_2y", "O_E", "cal_slope", "Brier", "scaled_Brier", "ICI", "E90"]
log_lines = []
_DF = None


def log(msg=""):
    print(msg)
    log_lines.append(str(msg))


def _init(df):
    global _DF
    _DF = df


def one_resample(args):
    b, seed = args
    df = _DF
    rng = np.random.default_rng(seed)
    boot = df.iloc[rng.integers(0, len(df), len(df))].reset_index(drop=True)
    spec = fit_models(boot)
    fits = spec.pop("_fits")
    lr_refit = 2 * (fits["fg"].loglik - fits["fg2"].loglik)
    p_boot = predict_all(spec, boot)
    p_orig = predict_all(spec, df)
    tb, sb = boot.time_days.to_numpy(float), boot.status.to_numpy(int)
    to, so = df.time_days.to_numpy(float), df.status.to_numpy(int)
    rows = []
    for name in UPDATED_MODELS:
        app = M.performance_summary(tb, sb, p_boot[name], H)
        test = M.performance_summary(to, so, p_orig[name], H)
        rows.append({"b": b, "model": name, **{f"app_{k}": app[k] for k in MEASURES},
                     **{f"test_{k}": test[k] for k in MEASURES},
                     "M2_slope": spec["M2"]["slope"], "LR_refit": lr_refit,
                     "refit_p": chi2.sf(lr_refit, 3)})
    return rows


def main():
    df = pd.read_csv(DERIVED_DIR / "analysis_cohort.csv")
    t, s = df.time_days.to_numpy(float), df.status.to_numpy(int)
    n_boot = int(os.environ.get("THKFRE_NBOOT", N_BOOTSTRAP))

    # ---------------- apparent performance on the full cohort
    spec = fit_models(df)
    spec.pop("_fits")
    preds = predict_all(spec, df)
    apparent = {k: M.performance_summary(t, s, p, H) for k, p in preds.items()}

    # ---------------- A. bootstrap
    seeds = np.random.SeedSequence(RANDOM_SEED).generate_state(n_boot)
    workers = max(1, (os.cpu_count() or 1))
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(df,)) as ex:
        for i, res in enumerate(ex.map(one_resample, enumerate(seeds), chunksize=5)):
            rows.extend(res)
            if (i + 1) % 100 == 0:
                print(f"  bootstrap {i + 1}/{n_boot}", flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(TABLE_DIR / "04_bootstrap_raw.csv", index=False)

    out = []
    for name in MODEL_NAMES:
        rec = {"model": name}
        for k in MEASURES:
            rec[f"apparent_{k}"] = apparent[name][k]
            if name in UPDATED_MODELS:
                r = raw[raw.model == name]
                opt = float((r[f"app_{k}"] - r[f"test_{k}"]).mean())
            else:
                opt = 0.0
            rec[f"optimism_{k}"] = opt
            rec[f"corrected_{k}"] = apparent[name][k] - opt
        out.append(rec)
    out = pd.DataFrame(out)
    out.to_csv(TABLE_DIR / "04_optimism_corrected.csv", index=False)

    log(f"BOOTSTRAP OPTIMISM CORRECTION (B = {n_boot}, seed {RANDOM_SEED})")
    show = out[["model"] + [f"corrected_{k}" for k in MEASURES]].copy()
    show.columns = ["model"] + MEASURES
    log("Optimism-corrected performance:")
    log(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    opt = out[["model"] + [f"optimism_{k}" for k in MEASURES]].copy()
    opt.columns = ["model"] + MEASURES
    log("")
    log("Optimism (apparent - corrected):")
    log(opt.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    sel = raw.drop_duplicates("b")
    log("")
    log("MODEL-SELECTION STABILITY")
    log(f"  M2 slope across resamples: median {sel.M2_slope.median():.3f} "
        f"(2.5-97.5%: {sel.M2_slope.quantile(.025):.3f}-{sel.M2_slope.quantile(.975):.3f})")
    log(f"  Refit (M3b) significantly better than M2 (p<0.05) in "
        f"{(sel.refit_p < 0.05).mean():.1%} of resamples")

    # ---------------- B. temporal validation
    dev, val = df[df.index_year == 2021], df[df.index_year == 2022]
    log("")
    log(f"TEMPORAL VALIDATION: develop 2021 (n={len(dev)}, events={(dev.status == 1).sum()}) "
        f"-> test 2022 (n={len(val)}, events={(val.status == 1).sum()})")
    spec_t = fit_models(dev)
    spec_t.pop("_fits")
    pv = predict_all(spec_t, val)
    tv, sv = val.time_days.to_numpy(float), val.status.to_numpy(int)
    temp = pd.DataFrame([{"model": k, **M.performance_summary(tv, sv, p, H)} for k, p in pv.items()])
    # 95% bootstrap CI of O/E in the 2022 test set (predictions held fixed)
    rng = np.random.default_rng(RANDOM_SEED + 1)
    oe = {k: [] for k in pv}
    for _ in range(n_boot):
        i = rng.integers(0, len(val), len(val))
        for k, p in pv.items():
            oe[k].append(M.observed_expected(tv[i], sv[i], p[i], H))
    temp["O_E_lo"] = [np.percentile(oe[k], 2.5) for k in temp.model]
    temp["O_E_hi"] = [np.percentile(oe[k], 97.5) for k in temp.model]
    temp.to_csv(TABLE_DIR / "04_temporal_validation.csv", index=False)
    log(temp[["model"] + MEASURES + ["O_E_lo", "O_E_hi"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    log(f"  2021-derived M2: slope {spec_t['M2']['slope']:.3f}, S0 {spec_t['M2']['s0']:.5f}")

    (LOG_DIR / "04_bootstrap_log.txt").write_text("\n".join(log_lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
