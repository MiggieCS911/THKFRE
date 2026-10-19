"""Run the whole TH-KFRE analysis from the raw data file, in order, and stop at the first error.

Usage (from analysis_v2/):
    python run_all.py            # full run (bootstrap ~15 min on 2 cores, faster with more)
    python run_all.py --quick    # 50 bootstrap resamples, for a fast smoke test only

Each step writes its own log to outputs/logs/. A summary with run times is written to
outputs/logs/run_all_log.txt.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    "01_clean_data.py",
    "02_validate_original.py",
    "03_update_models.py",
    "04_bootstrap_validation.py",
    "05_clinical_utility.py",
    "06_subgroups_sensitivity.py",
    "07_figures.py",
    "08_tables.py",
    "manuscript/build_manuscript.py",
    "manuscript/build_tripod_checklist.py",
]


def main():
    env = dict(os.environ)
    if "--quick" in sys.argv:
        env["THKFRE_NBOOT"] = "50"
        print("QUICK MODE: 50 bootstrap resamples (results are NOT the published ones)")
    env["PYTHONPATH"] = str(HERE) + os.pathsep + env.get("PYTHONPATH", "")
    lines = [f"Python {sys.version.split()[0]}"]
    t_all = time.time()
    for step in STEPS:
        if not (HERE / step).exists():
            print(f"\n=== {step} === not present, skipped")
            lines.append(f"{step:42s} SKIPPED (not in this copy)")
            continue
        t0 = time.time()
        print(f"\n=== {step} ===", flush=True)
        r = subprocess.run([sys.executable, step], cwd=HERE, env=env)
        dt = time.time() - t0
        lines.append(f"{step:42s} {'OK' if r.returncode == 0 else 'FAILED'}  {dt:7.1f} s")
        if r.returncode != 0:
            print(f"\nFAILED at {step}")
            break
    lines.append(f"{'total':42s}      {time.time() - t_all:7.1f} s")
    (HERE / "outputs" / "logs").mkdir(parents=True, exist_ok=True)
    (HERE / "outputs" / "logs" / "run_all_log.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    sys.exit(1 if any("FAILED" in x for x in lines) else 0)


if __name__ == "__main__":
    main()
