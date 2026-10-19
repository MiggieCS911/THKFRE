# TH-KFRE: 2-year Kidney Failure Risk Equation in a Thai CKD cohort

Analysis code for the validation and updating of the 4-variable Kidney Failure
Risk Equation (KFRE) at a 2-year horizon in patients with CKD stage 3-5 (non-dialysis),
Faculty of Medicine Vajira Hospital, Bangkok.

Study plan (written before the analysis): `docs/THKFRE_research_plan.md`. Reporting follows TRIPOD+AI.

## Data
Patient-level data are not included in this repository. Place the raw extract at
`../data/validation_dataset_ThaiKFRE.xlsx` (one folder above the repository), or point to it with
an environment variable:
```bash
export THKFRE_DATA=/path/to/validation_dataset_ThaiKFRE.xlsx
```
De-identified data are available from the corresponding author on reasonable request with
approval of the institutional review board.

## TH-KFRE v1.0
```
LP   = -0.2201 x (age/10 - 7.036) + 0.2467 x (male - 0.5642)
       - 0.5567 x (eGFR/5 - 7.222) + 0.4510 x (ln UACR[mg/g] - 5.137)     (published 4-variable KFRE)
2-year risk of kidney failure = 1 - 0.99464 ^ exp(1.343 x LP)
```
Use with eGFR < 60 mL/min/1.73 m2 and a measured UACR in mg/g (enter results below the detection
limit as 1 mg/g). Worked example: a 60-year-old man with eGFR 20 and UACR 1000 mg/g has
LP = 2.928 and a 2-year risk of 24.0% (27.1% with the published non-North American KFRE).

## How to run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tests/test_core.py          # 10 quick checks (~5 s)
python tests/test_finegray.py      # Python Cox / Fine-Gray vs R (needs the data file)
python run_all.py                  # whole analysis, ~20 min on 2 cores (bootstrap is the slow step)
python run_all.py --quick          # smoke test with 50 bootstrap resamples (not the published numbers)
```
Every script writes its numbers to `outputs/logs/`, tables to `outputs/tables/`,
figures to `outputs/figures/` and the manuscript to `outputs/manuscript/`.

## Scripts
| Script | Purpose | Status |
|---|---|---|
| `run_all.py` | Runs every step in order, logs run times | done |
| `01_clean_data.py` | Apply cleaning rules, freeze cohort, de-identify | done |
| `02_validate_original.py` | External validation of published KFRE (NA and non-NA) | done |
| `03_update_models.py` | Updating ladder: recalibration, Cox and Fine-Gray refit, LR tests | done |
| `04_bootstrap_validation.py` | Bootstrap optimism correction (1,000x) + temporal validation 2021->2022 | done |
| `05_clinical_utility.py` | Threshold performance, reclassification, decision curves (TH-KFRE v1.0 vs published KFRE) | done |
| `06_subgroups_sensitivity.py` | Subgroups (age, sex, eGFR, UACR) and sensitivity analyses | done |
| `07_figures.py` | Figures 1-5 for BMC Nephrology (85/170 mm, 300 dpi, embedded fonts) | done |
| `08_tables.py` | Tables 1-4 and Additional file 1 (Tables S1-S4) as Word tables | done |
| `manuscript/build_manuscript.py` | Manuscript draft, numbers pulled from outputs (added to the public repository after publication) | private |
| `manuscript/build_tripod_checklist.py` | TRIPOD+AI checklist (added after publication) | private |

## Verification
- `tests/test_finegray.py` reproduces R `coxph` (Cox) and `survival::finegray` + `coxph`
  (Fine-Gray) coefficients (max difference < 1e-6). The Fine-Gray model is implemented in
  `thkfre/finegray.py` because no standard Python package provides it.
- `tests/test_core.py` checks the KFRE equation against the risks in the hospital file, the
  non-North American baseline against Tangri 2016, the TH-KFRE equation, and the statistical
  functions on simulated data (Aalen-Johansen = Kaplan-Meier without competing events,
  Fine-Gray = Cox without competing events, Harrell C extremes, unit calibration slope).
- Reproducibility (16 Oct 2026): a clean run from the raw data reproduced every table, log,
  Word document and figure pixel-for-pixel (Python 3.13, 22 min on 2 cores). On a second
  machine (Python 3.10, different package versions) steps 01-03, 05 and 06 agreed with the
  reference outputs to within 5e-13, and steps 04 (short bootstrap), 07, 08 and the
  manuscript scripts ran without error.

All analysis choices (paths, horizon, seeds, cleaning rules, KFRE coefficients)
are in `thkfre/config.py`.

## Outputs included in this repository
Aggregate results only: `outputs/tables/*.csv` (performance, coefficients, calibration groups,
decision curves, subgroups; no patient-level rows), `outputs/figures/` and `outputs/logs/`.
Patient-level files (`outputs/derived/`) are excluded by `.gitignore`.

## Citation and licence
Please cite the accompanying paper (see `CITATION.cff`). Code released under the MIT licence.
