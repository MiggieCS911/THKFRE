"""Step 8a - Publication figures (BMC Nephrology format).

Figure 1  Patient flow diagram                                     170 mm wide
Figure 2  Cumulative incidence of kidney failure and death          85 mm wide
Figure 3  Calibration of KFRE-NA, KFRE-nonNA and TH-KFRE           170 mm wide
Figure 4  Decision curves                                           85 mm wide
Figure 5  Predicted vs observed risk in clinical risk categories   170 mm wide

Each figure is saved as PDF (vector, preferred by BMC), TIFF (300 dpi, LZW) and PNG
in outputs/figures/. Figure titles and legends belong in the manuscript text.
Uncertainty: 95% percentile bootstrap intervals (calibration curves: 200 resamples;
risk-category observed risk: 1,000 resamples).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from thkfre.config import (DERIVED_DIR, TABLE_DIR, FIG_DIR, HORIZON_DAYS, RANDOM_SEED, N_BOOTSTRAP,
                           KFRE_S0_2Y_NORTH_AMERICA, KFRE_S0_2Y_NON_NORTH_AMERICA, SCREENING)
from thkfre.kfre import kfre_risk, thkfre_risk
from thkfre import metrics as M
from thkfre import figstyle as S

H = HORIZON_DAYS
N_BOOT_CURVE = 200
MODELS = ["KFRE-NA", "KFRE-nonNA", "TH-KFRE v1.0"]


def predictions(df):
    return {"KFRE-NA": kfre_risk(df, KFRE_S0_2Y_NORTH_AMERICA),
            "KFRE-nonNA": kfre_risk(df, KFRE_S0_2Y_NON_NORTH_AMERICA),
            "TH-KFRE v1.0": thkfre_risk(df)}


# ======================================================================== Figure 1
def figure1(df):
    flow = pd.read_csv(TABLE_DIR / "flow_counts.csv").set_index("step")["n"]
    n_file = int(flow["Patients in analysis file"])
    assert SCREENING["identified"] - SCREENING["excluded_no_uacr"] - SCREENING["excluded_not_ckd"] == n_file
    n_kf, n_death = int((df.status == 1).sum()), int((df.status == 2).sum())
    n_alive = len(df) - n_kf - n_death

    fig, ax = plt.subplots(figsize=(S.DOUBLE, 112 * S.MM))
    ax.set_xlim(0, 170); ax.set_ylim(0, 112); ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    def box(x, y, w, h, text, bold=False, left=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0", lw=0.6,
                                    ec=S.INK_2, fc="white"))
        ax.text(x + 2.5 if left else x + w / 2, y + h / 2, text, ha="left" if left else "center",
                va="center", fontsize=7, fontweight="bold" if bold else "normal", linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color=S.INK_2, mutation_scale=7))

    cx, w = 6, 74            # main column
    box(cx, 98, w, 12, f"Patients with CKD (ICD-10 N18.3–N18.9)\nattending the nephrology clinic, 2021–2022\nn = {SCREENING['identified']}")
    box(96, 92, 72, 16, f"Excluded (n = {SCREENING['excluded_no_uacr'] + SCREENING['excluded_not_ckd']})\n"
                        f"• No UACR measured: n = {SCREENING['excluded_no_uacr']}\n"
                        f"• Did not meet CKD or eligibility\n   criteria: n = {SCREENING['excluded_not_ckd']}", left=True)
    box(cx, 74, w, 10, f"Eligible patients with UACR\nn = {n_file}")
    no_uacr = int(flow["Excluded: no UACR result"])
    egfr60 = int(flow["Excluded: eGFR >= 60"])
    box(96, 66, 72, 14, f"Excluded at data cleaning (n = {no_uacr + egfr60})\n"
                        f"• No UACR result on record: n = {no_uacr}\n"
                        f"• Baseline eGFR ≥60 mL/min/1.73 m²: n = {egfr60}", left=True)
    box(cx, 48, w, 12, f"Analysis cohort\nn = {len(df)}\n(index visits: 2021 n = {(df.index_year == 2021).sum()}; "
                       f"2022 n = {(df.index_year == 2022).sum()})", bold=False)
    yb, hb, wb = 6, 22, 46
    xs = [6, 62, 118]
    texts = [f"Kidney failure\nwithin 2 years\nn = {n_kf} ({n_kf / len(df):.1%})\n"
             f"HD {int((df.krt_modality == 'HD').sum())}, PD {int((df.krt_modality == 'PD').sum())}",
             f"Death before\nkidney failure\nn = {n_death} ({n_death / len(df):.1%})",
             f"Alive without\nkidney failure at 2 years*\nn = {n_alive} ({n_alive / len(df):.1%})"]
    for x, t in zip(xs, texts):
        box(x, yb, wb, hb, t)
    arrow(cx + w / 2, 98, cx + w / 2, 84)
    arrow(cx + w / 2, 74, cx + w / 2, 60)
    ax.plot([cx + w / 2, 95], [91, 91], lw=0.6, c=S.INK_2); arrow(95, 91, 96, 91)
    ax.plot([cx + w / 2, 95], [67, 67], lw=0.6, c=S.INK_2); arrow(95, 67, 96, 67)
    ax.plot([cx + w / 2, cx + w / 2], [48, 36], lw=0.6, c=S.INK_2)
    ax.plot([xs[0] + wb / 2, xs[2] + wb / 2], [36, 36], lw=0.6, c=S.INK_2)
    for x in xs:
        arrow(x + wb / 2, 36, x + wb / 2, yb + hb)
    n_cens = int(((df.status == 0) & (df.time_days < H)).sum())
    ax.text(6, 1.5, f"*Includes {n_cens} patients censored before 2 years. HD, haemodialysis; PD, peritoneal dialysis; "
                    "UACR, urine albumin-to-creatinine ratio.", fontsize=6, color=S.INK_2, va="bottom")
    S.save(fig, FIG_DIR / "Figure1_flow")
    plt.close(fig)


# ======================================================================== Figure 2
def figure2(df):
    t, s = df.time_days.to_numpy(float), df.status.to_numpy(int)
    fig = plt.figure(figsize=(S.SINGLE - 3 * S.MM, 72 * S.MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 0.7], hspace=0.38)
    ax = fig.add_subplot(gs[0])
    for code, key in [(1, "KF"), (2, "Death")]:
        tt, cc = M.aalen_johansen_curve(t, s, code, H)
        tt, cc = np.append(tt, H), np.append(cc, cc[-1])
        st = S.OUTCOME_STYLE[key]
        ax.step(tt / 365.25 * 12, cc * 100, where="post", color=st["color"], ls=st["ls"], lw=1.2)
        ax.text(H / 365.25 * 12 + 0.4, cc[-1] * 100, f"{st['label']}\n{cc[-1]:.1%}", fontsize=6.5,
                va="center", ha="left", color=S.INK)
    ax.set_xlim(0, 24); ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_ylim(0, 10); ax.set_ylabel("Cumulative incidence (%)")
    ax.set_xlabel("Months from index visit")
    # numbers at risk
    axr = fig.add_subplot(gs[1])
    axr.axis("off")
    axr.set_xlim(ax.get_xlim())
    axr.text(-0.5, 0.75, "Number at risk", fontsize=6.5, fontweight="bold", va="bottom", ha="left",
             transform=axr.transData)
    for m in [0, 6, 12, 18, 24]:
        day = m * 365.25 / 12
        n_risk = int((t >= min(day, H)).sum()) if m < 24 else int(((t >= H) & (s == 0)).sum())
        axr.text(m, 0.25, f"{n_risk}", ha="center", va="center", fontsize=6.5)
    axr.set_ylim(0, 1.2)
    fig.subplots_adjust(right=0.70)
    S.save(fig, FIG_DIR / "Figure2_cumulative_incidence")
    plt.close(fig)


# ======================================================================== Figure 3
def figure3(df, rng):
    t, s = df.time_days.to_numpy(float), df.status.to_numpy(int)
    preds = predictions(df)
    cuts = [0, 0.02, 0.06, 0.10, 0.20, 0.40, np.inf]
    grid = np.linspace(0.002, 0.95, 120)
    boots = [rng.integers(0, len(df), len(df)) for _ in range(N_BOOT_CURVE)]

    fig = plt.figure(figsize=(S.DOUBLE, 82 * S.MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[4, 1], hspace=0.08, wspace=0.28)
    order = ["KFRE-NA", "KFRE-nonNA", "TH-KFRE v1.0"]
    summary = []
    for j, name in enumerate(order):
        p = preds[name]
        st = S.MODEL_STYLE[name]
        ax = fig.add_subplot(gs[0, j])
        axh = fig.add_subplot(gs[1, j], sharex=ax)
        hi_p = np.quantile(p, 0.99)
        g = grid[grid <= hi_p]
        _, obs, cal = M.flexible_calibration(t, s, p, H, grid=g)
        band = np.array([M.flexible_calibration(t[i], s[i], p[i], H, grid=g)[1] for i in boots])
        lo, hi = np.percentile(band, [2.5, 97.5], axis=0)
        ax.fill_between(g, lo, hi, color=st["color"], alpha=0.18, lw=0)
        ax.plot(g, obs, color=st["color"], ls=st["ls"], lw=1.4)
        ax.plot([0, 1], [0, 1], color=S.INK_2, lw=0.6, ls=(0, (2, 2)))
        # grouped points with bootstrap CI
        grp = np.digitize(p, cuts[1:-1])
        for k in range(len(cuts) - 1):
            m = grp == k
            if m.sum() < 10:
                continue
            o = M.aalen_johansen(t[m], s[m], H)
            idx = np.where(m)[0]
            bo = [M.aalen_johansen(t[ii], s[ii], H) for ii in
                  (rng.choice(idx, len(idx)) for _ in range(300))]
            ax.errorbar(p[m].mean(), o, yerr=[[o - np.percentile(bo, 2.5)], [np.percentile(bo, 97.5) - o]],
                        fmt=st["marker"], ms=4, color=st["color"], mfc="white", mew=1.0, elinewidth=0.8, capsize=0)
        oe = M.observed_expected(t, s, p, H)
        slope = M.calibration_slope(t, s, p)[0]
        ax.text(0.03, 0.97, f"O/E {oe:.2f}\nSlope {slope:.2f}\nICI {cal['ICI']:.3f}", transform=ax.transAxes,
                va="top", ha="left", fontsize=6.5, color=S.INK)
        ax.set_title(st["label"], fontsize=8, color=S.INK)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(0, 1.01, 0.2)); ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_aspect("equal", adjustable="box")
        plt.setp(ax.get_xticklabels(), visible=False)
        if j == 0:
            ax.set_ylabel("Observed 2-year risk")
        S.panel_label(ax, "ABC"[j], x=-0.22)
        # histogram of predicted risk
        axh.hist(p, bins=np.arange(0, 1.0001, 0.02), color="#9a9890", lw=0)
        axh.set_yscale("log"); axh.set_ylim(0.8, 1000)
        axh.set_yticks([1, 10, 100, 1000]); axh.set_yticklabels(["1", "10", "100", "1000"])
        axh.grid(False)
        axh.set_xlabel("Predicted 2-year risk")
        if j == 0:
            axh.set_ylabel("n (log)", fontsize=7)
        summary.append({"model": name, "O_E": oe, "slope": slope, "ICI": cal["ICI"]})
    # align histogram width with square calibration panel
    fig.canvas.draw()
    for j in range(3):
        a, h = fig.axes[2 * j], fig.axes[2 * j + 1]
        pa, ph = a.get_position(), h.get_position()
        h.set_position([pa.x0, ph.y0, pa.width, ph.height])
    S.save(fig, FIG_DIR / "Figure3_calibration")
    plt.close(fig)
    return pd.DataFrame(summary)


# ======================================================================== Figure 4
def figure4():
    dca = pd.read_csv(TABLE_DIR / "05_decision_curve.csv")
    fig, ax = plt.subplots(figsize=(S.SINGLE, 66 * S.MM))
    ax.plot(dca.threshold, dca.treat_all, color="#9a9890", lw=0.9, label="Refer all")
    ax.axhline(0, color=S.INK_2, lw=0.8, label="Refer none")
    for name in ["KFRE-NA", "KFRE-nonNA", "TH-KFRE v1.0"]:
        st = S.MODEL_STYLE[name]
        ax.plot(dca.threshold, dca[name], color=st["color"], ls=st["ls"], lw=1.3, label=st["label"])
    for x in (0.10, 0.40):
        ax.axvline(x, color=S.INK_2, lw=0.5, ls=(0, (1, 2)))
        ax.text(x, 0.088, f"{x:.0%}", ha="center", va="bottom", fontsize=6.5, color=S.INK_2)
    ax.set_xlim(0, 0.5); ax.set_ylim(-0.01, 0.09)
    ax.set_xticks(np.arange(0, 0.51, 0.1)); ax.set_xticklabels([f"{v:.0%}" for v in np.arange(0, 0.51, 0.1)])
    ax.set_xlabel("Threshold 2-year risk of kidney failure"); ax.set_ylabel("Net benefit")
    ax.legend(loc="upper right", fontsize=6.5, handlelength=2.6)
    S.save(fig, FIG_DIR / "Figure4_decision_curve")
    plt.close(fig)


# ======================================================================== Figure 5
def figure5(df, rng):
    t, s = df.time_days.to_numpy(float), df.status.to_numpy(int)
    preds = predictions(df)
    cats = [(0, 0.10, "<10%"), (0.10, 0.40, "10% to <40%"), (0.40, 1.01, "≥40%")]
    order = ["KFRE-NA", "KFRE-nonNA", "TH-KFRE v1.0"]
    fig, axes = plt.subplots(1, 3, figsize=(S.DOUBLE, 62 * S.MM), sharey=True)
    rows = []
    for ax, (lo_c, hi_c, lab) in zip(axes, cats):
        for y, name in enumerate(order):
            p = preds[name]
            m = (p >= lo_c) & (p < hi_c)
            idx = np.where(m)[0]
            st = S.MODEL_STYLE[name]
            pred = p[m].mean() * 100
            obs = M.aalen_johansen(t[m], s[m], H) * 100
            bo = np.array([M.aalen_johansen(t[ii], s[ii], H) for ii in
                           (rng.choice(idx, len(idx)) for _ in range(N_BOOTSTRAP))]) * 100
            olo, ohi = np.percentile(bo, [2.5, 97.5])
            ax.plot([pred, obs], [y, y], color=st["color"], lw=0.8, alpha=0.6)
            ax.errorbar(obs, y, xerr=[[obs - olo], [ohi - obs]], fmt=st["marker"], ms=5, color=st["color"],
                        elinewidth=0.9, capsize=0, zorder=3)
            ax.plot(pred, y, marker=st["marker"], ms=5, mfc="white", mec=st["color"], mew=1.0, ls="none", zorder=4)
            ax.text(0.98, y + 0.28, f"n = {m.sum()}", transform=ax.get_yaxis_transform(), fontsize=6,
                    va="center", ha="right", color=S.INK_2)
            rows.append({"category": lab, "model": name, "n": int(m.sum()), "events": int(((s == 1) & m).sum()),
                         "predicted_pct": pred, "observed_pct": obs, "observed_lo": olo, "observed_hi": ohi})
        ax.set_title(f"Predicted risk {lab}", fontsize=8)
        ax.set_xlabel("2-year risk of kidney failure (%)")
        ax.set_ylim(-0.6, 2.6)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(left=0)
    axes[0].set_yticks(range(3))
    axes[0].set_yticklabels([S.MODEL_STYLE[n]["label"] for n in order])
    axes[2].set_xlim(0, 100)
    # key inside the figure
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="none", mfc="white", mec=S.INK_2, ms=5, label="Mean predicted"),
               Line2D([], [], marker="o", ls="none", color=S.INK_2, ms=5, label="Observed (95% CI)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=6.5, handletextpad=0.3,
               bbox_to_anchor=(0.6, -0.04))
    fig.subplots_adjust(wspace=0.12, left=0.2, bottom=0.27)
    S.save(fig, FIG_DIR / "Figure5_risk_categories")
    plt.close(fig)
    return pd.DataFrame(rows)


def main():
    S.apply()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DERIVED_DIR / "analysis_cohort.csv")
    rng = np.random.default_rng(RANDOM_SEED)
    figure1(df); print("Figure 1 done")
    figure2(df); print("Figure 2 done")
    cal = figure3(df, rng); print("Figure 3 done"); print(cal.to_string(index=False))
    figure4(); print("Figure 4 done")
    cats = figure5(df, rng); print("Figure 5 done")
    cats.to_csv(TABLE_DIR / "07_figure5_values.csv", index=False)
    print(cats.to_string(index=False, float_format=lambda x: f"{x:.1f}"))


if __name__ == "__main__":
    main()
