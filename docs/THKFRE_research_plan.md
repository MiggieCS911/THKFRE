# TH-KFRE Research Plan
*2-year, 4-variable Kidney Failure Risk Equation — validation, updating and implementation in a Thai CKD cohort*

Version 1.0 · 16 Oct 2026 · Status: agreed design, analysis not yet started

---

## 0. Decisions already made

| Decision | Choice | Reason |
|---|---|---|
| Validation design | **Full cohort (n = 1,062) + bootstrap optimism correction** (replaces the 80:20 split) | The 20% split holds only 18 events; ~100 events is the usual minimum for a validation set. Bootstrap uses all 90 events and is the TRIPOD-preferred internal validation. |
| Software | **Python only** | Fine-Gray model implemented in Python and verified against R (see §0.1). |
| Reporting standard | **TRIPOD+AI** checklist | Expected by nephrology journals for prediction model papers. |
| Competing risk | **Death before kidney failure = competing event** in all evaluation (observed risk via Aalen-Johansen). | 73 deaths (6.9%) vs 90 KRT events; ignoring death biases calibration. |
| UACR below detection | Lab reports "< 1" → set to **0.001 g/g = 1 mg/g** (detection limit) | Agreed 16 Oct 2026 (unit to be confirmed). |
| eGFR ≥ 60 at baseline | **Exclude** (3 patients) | Agreed 16 Oct 2026. |
| Final model | **TH-KFRE v1.0 = KFRE linear predictor recalibrated (baseline + slope)**: risk = 1 − 0.99464^exp(1.343 × LP). Refit Cox / Fine-Gray not carried forward. | Closed testing: slope p = 0.0001, refit p = 0.075; equal optimism-corrected performance. Agreed 16 Oct 2026. |
| Paper framing | Published KFRE (non-North American calibration) is validated in Thai CKD; TH-KFRE gives better-calibrated absolute risk but no meaningful gain in referral decisions. | Agreed 16 Oct 2026. |
| Calculator threshold | Keep guideline **10%** (MDT CKD care) and **40%** (KRT preparation); report the trade-off at 10%. | Agreed 16 Oct 2026. |

### 0.1 Fine-Gray in Python — verification
A Fine-Gray subdistribution-hazard model was coded in Python (censoring-weighted risk sets as in R `survival::finegray`, weighted Cox partial likelihood with Efron ties). On the full cohort it reproduced the earlier R output to 6 decimal places:

| Variable | R (`finegray` + `coxph`) | Python | 
|---|---|---|
| Age (per 10 y) | −0.100458 | −0.100458 |
| Male | 0.342006 | 0.342006 |
| eGFR (per 5) | −0.740823 | −0.740823 |
| ln(UACR) | 0.785703 | 0.785703 |

Note: in this cohort almost all censoring is administrative at 730 days (only 3 patients censored earlier), so the censoring weights are close to 1 and the Fine-Gray model is numerically stable.

---

## Final deliverables (agreed 16 Oct 2026)
1. **Complete, human-verifiable analysis code** ready to post on GitHub: numbered scripts, a README with run instructions, `requirements.txt`, fixed random seeds, no patient identifiers in the repository (data file excluded via `.gitignore`), and a log showing every number in the paper comes from the code.
2. **Publication-quality figures** designed together (journal-ready resolution, consistent style, colour-blind-safe), plus the tables.
3. Manuscript text (Methods/Results) rewritten from the new outputs.
4. TH-KFRE v1.0 equation + reference implementation for the clinical tool.

---

## Part A — Academic publication

### Step 1. Freeze cohort and data rules
- [ ] One consistent patient flow diagram: screened → excluded (with reasons) → 1,062 analysed; confirm the screening counts with the EHR query.
- [ ] Index date = first qualifying clinic visit (`vstdate`); follow-up capped at 730 days.
- [ ] Outcome = maintenance dialysis > 3 months or kidney transplant within 2 years (`outcome`). Competing event = death before KRT (`Dead`).
- [ ] Data fixes to decide and document:
  - 13 patients with UACR = 1e-7 (placeholder for undetectable) → replace with lab lower limit of detection (confirm value with lab).
  - 3 patients with eGFR ≥ 60 → confirm eligibility (exclude or keep with justification).
  - Confirm eGFR equation (CKD-EPI 2009 vs 2021) and UACR unit (mg/g).
- [ ] Missing data: the 4 predictors and outcome are complete; no imputation needed for the main models.

### Step 2. Statistical analysis plan (SAP)
- [ ] Write the SAP before running final analyses (this document → short SAP appendix).
- [ ] Pre-specify models, metrics, thresholds (10%, 40%) and subgroups below.

### Step 3. External validation of the original KFRE (primary aim)
Apply both published 4-variable versions to the full cohort:
- **KFRE-NA** (North American, 2-y baseline 0.9750 — this is what `prob_2y_4v` in the dataset uses)
- **KFRE-nonNA** (Non-North American 2-y baseline — *verify exact value against Tangri et al., JAMA 2016*)

Metrics (all at 2 years, 95% CI by 1,000 bootstrap resamples):
- Discrimination: Harrell's C and Uno's time-dependent C/AUC at 730 days
- Calibration-in-the-large: observed/expected (O/E) ratio, observed = Aalen-Johansen cumulative incidence (≈ 8.5%)
- Calibration slope
- Smoothed calibration curve vs observed cumulative incidence (pseudo-values or Aalen-Johansen by risk groups)
- Overall accuracy: Brier score at 730 days (IPCW / competing-risk version) and scaled Brier

### Step 4. Model updating — closed testing ladder
Only move up a level if it clearly improves calibration (likelihood-ratio / closed testing procedure, Vergouwe 2017):
1. **Recalibration-in-the-large** — update baseline risk only (keep KFRE β)
2. **Logistic/linear recalibration** — baseline + one slope on the KFRE linear predictor
3. **Full refit (de novo)**:
   - 3a. Cause-specific Cox (death censored)
   - 3b. Fine-Gray (death competing) — **preferred for absolute risk prediction**
   - Same predictors and transforms as KFRE: age/10, male, eGFR/5, ln(UACR), centred on cohort means.
   - Consider uniform shrinkage (bootstrap-estimated) of refitted β.

### Step 5. Internal validation
- [ ] Bootstrap optimism correction (500–1,000 resamples): repeat the entire model-building step in each resample, report optimism-corrected C, calibration slope, O/E, Brier.
- [ ] Sensitivity: temporal validation — develop on 2021 index visits, test on 2022 index visits.
- [ ] Sample size justification: Riley et al. criteria for 4 parameters with 90 events (~22 events per parameter); report expected shrinkage.

### Step 6. Clinical usefulness
- [ ] Performance at KDIGO-style thresholds — 2-y risk ≥ 10% (multidisciplinary CKD clinic) and ≥ 40% (KRT preparation): sensitivity, specificity, PPV, NPV, number needed to refer (1/PPV).
- [ ] Risk-group table (<10%, 10–<20%, 20–<40%, ≥40%): n, predicted vs observed cumulative incidence.
- [ ] Decision curve analysis accounting for competing risk, threshold 1–50%, all models vs treat-all/treat-none.

### Step 7. Sensitivity and subgroup analyses
- [ ] Age < 65 vs ≥ 65 (competing-risk effect of death)
- [ ] eGFR < 30 vs ≥ 30; CKD stage 3 vs 4–5
- [ ] UACR A1–A2 vs A3
- [ ] Excluding UACR placeholder values / eGFR ≥ 60

### Step 8. Outputs for the manuscript
- Table 1: baseline characteristics (overall; KRT vs no KRT; died)
- Table 2: performance of all models (apparent + optimism-corrected)
- Table 3: final equations (β, means, baseline risk at 2 y)
- Table 4: threshold performance / risk groups
- Figure 1: flow diagram
- Figure 2: cumulative incidence of KRT and death (Aalen-Johansen)
- Figure 3: calibration plots
- Figure 4: decision curves
- Supplement: TRIPOD+AI checklist, full code, bootstrap details

### Step 9. Write and submit
- [ ] Write Methods and Results directly from the analysis outputs.
- [ ] Correct AI-use declaration to name the tools actually used.
- [ ] Candidate journals: *Kidney Medicine*, *Kidney International Reports*, *Clinical Kidney Journal*, *BMC Nephrology*, *Nephrology* (APSN).

---

## Part B — TH-KFRE as a clinical tool

### Step 10. Lock the final equation
- Choose the model from Step 4 (expected: original KFRE β + Thai baseline, unless refit is clearly better).
- Publish as **TH-KFRE v1.0**: exact coefficients, centring values, baseline risk, Python reference implementation, test cases.

### Step 11. Input specification
- UACR unit handling (mg/g; convert mg/mmol × 8.84); rule for values below detection limit.
- Behaviour when UACR is missing (e.g. UPCR→UACR conversion, flagged as lower confidence) — decide.
- Valid range checks: age ≥ 18, eGFR < 60, sex coding.

### Step 12. Calculator and EHR integration
- Phase 1: web / mobile calculator (TH-KFRE v1.0), Thai + English.
- Phase 2: automatic calculation in the hospital information system when eGFR and UACR are resulted.
- Display risk band + recommended action (≥10% → MDT CKD clinic referral; ≥40% → KRT planning / vascular access).

### Step 13. External validation (next study)
- Multicentre Thai validation: regional, provincial and primary-care settings; ideally prospective.

### Step 14. Implementation and monitoring
- Silent deployment (predictions logged, not shown) → compare with outcomes → go live.
- Annual recalibration check (O/E, calibration slope); update baseline if drift.
- Institutional approval; check Thai FDA rules for medical-device software.

---

## Working folder structure (to be created)
```
Claude_power/
├── THKFRE_research_plan.md          ← this file
├── data/validation_dataset_ThaiKFRE.xlsx   (original, read-only)
├── analysis_v2/
│   ├── 01_clean_data.py
│   ├── 02_validate_original.py
│   ├── 03_update_models.py          (recalibration, Cox, Fine-Gray)
│   ├── 04_bootstrap_validation.py
│   ├── 05_clinical_utility.py       (thresholds, DCA)
│   ├── thkfre/                      (reusable functions incl. Fine-Gray)
│   ├── outputs/tables/  outputs/figures/
│   └── README.md
```
