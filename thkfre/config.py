"""Shared settings for the TH-KFRE analysis.

Every script imports paths and constants from here so that a reader can
check all analysis choices in one place.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
REPO_DIR = Path(__file__).resolve().parents[1]          # analysis_v2/
# Raw extract (not in the repository). Default: ../data/ next to the repository folder;
# override with the environment variable THKFRE_DATA=/path/to/validation_dataset_ThaiKFRE.xlsx
import os
RAW_DATA = Path(os.environ.get("THKFRE_DATA",
                               REPO_DIR.parent / "data" / "validation_dataset_ThaiKFRE.xlsx"))
OUT_DIR = REPO_DIR / "outputs"
DERIVED_DIR = OUT_DIR / "derived"      # patient-level files: never commit
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
LOG_DIR = OUT_DIR / "logs"

# ---------------------------------------------------------------- study design
HORIZON_DAYS = 730                     # 2-year prediction horizon
RANDOM_SEED = 20261016
N_BOOTSTRAP = 1000

# ---------------------------------------------------------------- cleaning rules
# (agreed with the PI on 16 Oct 2026; see THKFRE_research_plan.md)
EGFR_MAX_EXCLUSIVE = 60.0              # exclude baseline eGFR >= 60
UACR_DETECTION_LIMIT_MG_G = 1.0        # lab reports "< 1" mg/g (= 0.001 g/g)

# ---------------------------------------------------------------- published KFRE
# 4-variable KFRE (Tangri et al. JAMA 2011; 2016). Linear predictor:
#   -0.2201*(age/10 - 7.036) + 0.2467*(male - 0.5642)
#   -0.5567*(eGFR/5 - 7.222) + 0.4510*(ln UACR[mg/g] - 5.137)
KFRE_BETA = {"age10": -0.2201, "male": 0.2467, "egfr5": -0.5567, "ln_uacr": 0.4510}
KFRE_MEANS = {"age10": 7.036, "male": 0.5642, "egfr5": 7.222, "ln_uacr": 5.137}
# 2-year baseline survival. The dataset's precomputed risks use 0.9750.
KFRE_S0_2Y_NORTH_AMERICA = 0.9750
# Non-North America: Tangri et al. JAMA 2016 lowered baseline 2-y risk by 32.9%:
#   1 - 0.0250 * (1 - 0.329) = 0.983225, published rounded as 0.9832.
KFRE_S0_2Y_NON_NORTH_AMERICA = 0.9832

PREDICTORS = ["age10", "male", "egfr5", "ln_uacr"]

# ---------------------------------------------------------------- TH-KFRE v1.0 (locked 16 Oct 2026)
# Model M2 of 03_update_models.py: recalibrated baseline + slope on the KFRE linear predictor.
#   2-year risk = 1 - THKFRE_S0_2Y ** exp(THKFRE_SLOPE * KFRE_LP)
# Values rounded from 03_model_specifications.json (slope 1.343001, S0 0.994640).
THKFRE_VERSION = "1.0"
THKFRE_SLOPE = 1.3430
THKFRE_S0_2Y = 0.99464

# ---------------------------------------------------------------- screening (flow diagram)
# Counts BEFORE the analysis file (hospital EHR query, before data extraction).
# Confirmed by the PI on 16 Oct 2026 (the 7,027 figure in the earlier thesis is not used).
SCREENING = {
    "identified": 5264,               # CKD ICD-10 N18.3-N18.9, nephrology clinic 2021-2022
    "excluded_no_uacr": 4038,         # no UACR measured
    "excluded_not_ckd": 164,          # did not meet CKD criteria / exclusion criteria
}
