"""Performance measures at a fixed horizon with death as a competing event.

Conventions: status 1 = kidney failure, 2 = death (competing), 0 = censored.
Methods follow van Geloven et al., BMJ 2022;377:e069249 ("Validation of
prediction models in the presence of competing risks").
"""
import numpy as np


# ------------------------------------------------------------------ estimators
def aalen_johansen(time, status, horizon, event_code=1):
    """Aalen-Johansen cumulative incidence of `event_code` at `horizon`."""
    time = np.asarray(time, float)
    status = np.asarray(status, int)
    order = np.argsort(time, kind="stable")
    t, s = time[order], status[order]
    uniq, first = np.unique(t, return_index=True)
    n_at_risk = len(t) - first
    d_event = np.add.reduceat((s == event_code).astype(float), first)
    d_any = np.add.reduceat((s > 0).astype(float), first)
    keep = uniq <= horizon
    surv_before = np.concatenate([[1.0], np.cumprod(1 - d_any / n_at_risk)[:-1]])
    return float(np.sum((surv_before * d_event / n_at_risk)[keep]))


def censoring_survival(time, status):
    """Return a function G(t-) = P(censoring time >= t), Kaplan-Meier with reversed status."""
    time = np.asarray(time, float)
    status = np.asarray(status, int)
    ct = np.unique(time[status == 0])
    g, vals = 1.0, []
    for u in ct:
        g *= 1 - ((time == u) & (status == 0)).sum() / (time >= u).sum()
        vals.append(g)
    vals = np.array(vals)

    def G_left(x):
        idx = np.searchsorted(ct, np.asarray(x, float), side="left") - 1
        return np.where(idx < 0, 1.0, vals[np.clip(idx, 0, None)] if len(vals) else 1.0)
    return G_left


def ipcw_weights(time, status, horizon):
    """IPCW weights for a binary 'event 1 by horizon' outcome.
    Known status: event (1 or 2) before horizon -> 1/G(T-); event-free at horizon -> 1/G(h-).
    Censored before horizon -> 0."""
    time = np.asarray(time, float)
    status = np.asarray(status, int)
    G = censoring_survival(time, status)
    w = np.zeros(len(time))
    ev = (status > 0) & (time <= horizon)
    alive = time >= horizon
    alive &= ~ev
    w[ev] = 1 / G(time[ev])
    w[alive] = 1 / G(np.full(alive.sum(), horizon))
    return w


# ------------------------------------------------------------------ discrimination
def harrell_c(time, status, risk, horizon, event_code=1):
    """Harrell's C truncated at horizon, other events treated as censoring (as in KFRE papers)."""
    time, status, risk = map(np.asarray, (time, status, risk))
    case = (status == event_code) & (time <= horizon)
    ti, ri = time[case][:, None], risk[case][:, None]
    comparable = time[None, :] > ti                      # j still event-free when i fails
    conc = (ri > risk[None, :]) + 0.5 * (ri == risk[None, :])
    return float((conc * comparable).sum() / comparable.sum())


def auc_at_horizon(time, status, risk, horizon, event_code=1):
    """Cumulative/dynamic AUC(t) with competing risks (Blanche 2013, definition 2), IPCW.
    Cases: event 1 by horizon. Controls: everyone else with known status (event-free at
    horizon, or competing event before horizon)."""
    time, status, risk = map(np.asarray, (time, status, risk))
    w = ipcw_weights(time, status, horizon)
    case = (status == event_code) & (time <= horizon)
    ctrl = (w > 0) & ~case
    rc, wc = risk[case][:, None], w[case][:, None]
    rk, wk = risk[ctrl][None, :], w[ctrl][None, :]
    num = (wc * wk * ((rc > rk) + 0.5 * (rc == rk))).sum()
    return float(num / (wc.sum() * wk.sum()))


# ------------------------------------------------------------------ accuracy
def brier_at_horizon(time, status, risk, horizon, event_code=1):
    time, status, risk = map(np.asarray, (time, status, risk))
    w = ipcw_weights(time, status, horizon)
    y = ((status == event_code) & (time <= horizon)).astype(float)
    return float(np.sum(w * (y - risk) ** 2) / len(time))


def scaled_brier(time, status, risk, horizon, event_code=1):
    """1 - Brier(model) / Brier(null model predicting the marginal AJ CIF for everyone)."""
    null = np.full(len(time), aalen_johansen(time, status, horizon, event_code))
    return 1 - brier_at_horizon(time, status, risk, horizon, event_code) / \
        brier_at_horizon(time, status, null, horizon, event_code)


# ------------------------------------------------------------------ calibration
def observed_expected(time, status, risk, horizon, event_code=1):
    return aalen_johansen(time, status, horizon, event_code) / float(np.mean(risk))


def calibration_slope(time, status, risk):
    """Fine-Gray regression of the outcome on cloglog(predicted risk); ideal slope = 1."""
    from thkfre.finegray import fit_finegray
    from thkfre.kfre import cloglog
    res = fit_finegray(cloglog(np.asarray(risk)), time, status)
    return float(res.beta[0]), float(res.se[0])


def rcs_basis(x, knots):
    """Restricted cubic spline basis (Harrell), linear term + k-2 nonlinear terms."""
    x = np.asarray(x, float)
    k = np.asarray(knots, float)
    scale = (k[-1] - k[0]) ** 2
    cols = [x]
    for j in range(len(k) - 2):
        term = (np.clip(x - k[j], 0, None) ** 3
                - np.clip(x - k[-2], 0, None) ** 3 * (k[-1] - k[j]) / (k[-1] - k[-2])
                + np.clip(x - k[-1], 0, None) ** 3 * (k[-2] - k[j]) / (k[-1] - k[-2]))
        cols.append(term / scale)
    return np.column_stack(cols)


def flexible_calibration(time, status, risk, horizon, n_knots=3, grid=None):
    """Calibration curve from a Fine-Gray model of the outcome on a restricted cubic
    spline of cloglog(predicted risk) (Austin, Putter et al., Stat Med 2022).
    Returns grid, observed CIF on grid, and ICI/E50/E90 over patients."""
    from thkfre.finegray import fit_finegray
    from thkfre.kfre import cloglog
    risk = np.asarray(risk, float)
    x = cloglog(risk)
    knots = np.quantile(x, {3: [0.10, 0.50, 0.90], 4: [0.05, 0.35, 0.65, 0.95]}[n_knots])
    B = rcs_basis(x, knots)
    centre = B.mean(axis=0)
    fit = fit_finegray(B - centre, time, status)
    obs_patient = fit.predict_cif(B - centre, horizon)
    err = np.abs(obs_patient - risk)
    stats = {"ICI": float(err.mean()), "E50": float(np.median(err)), "E90": float(np.quantile(err, 0.9))}
    if grid is None:
        grid = np.quantile(risk, np.linspace(0.005, 0.995, 200))
    obs_grid = fit.predict_cif(rcs_basis(cloglog(grid), knots) - centre, horizon)
    return grid, obs_grid, stats


def grouped_calibration(time, status, risk, horizon, cuts, event_code=1):
    """Observed AJ CIF vs mean predicted risk within predicted-risk groups."""
    rows = []
    labels = [f"{lo:.0%}-<{hi:.0%}" if np.isfinite(hi) else f">={lo:.0%}"
              for lo, hi in zip(cuts[:-1], cuts[1:])]
    grp = np.digitize(risk, cuts[1:-1], right=False)
    for g, lab in enumerate(labels):
        m = grp == g
        n = int(m.sum())
        rows.append({"group": lab, "n": n,
                     "events": int(((status == event_code) & (time <= horizon) & m).sum()),
                     "predicted": float(risk[m].mean()) if n else np.nan,
                     "observed_AJ": aalen_johansen(time[m], status[m], horizon, event_code) if n else np.nan})
    return rows


# ------------------------------------------------------------------ summary
def performance_summary(time, status, risk, horizon):
    """All headline measures for one model on one dataset."""
    _, _, cal = flexible_calibration(time, status, risk, horizon)
    return {"Harrell_C": harrell_c(time, status, risk, horizon),
            "AUC_2y": auc_at_horizon(time, status, risk, horizon),
            "O_E": observed_expected(time, status, risk, horizon),
            "cal_slope": calibration_slope(time, status, risk)[0],
            "Brier": brier_at_horizon(time, status, risk, horizon),
            "scaled_Brier": scaled_brier(time, status, risk, horizon),
            "ICI": cal["ICI"], "E90": cal["E90"],
            "mean_predicted": float(np.mean(risk))}


# ------------------------------------------------------------------ clinical utility
def threshold_metrics(time, status, risk, cutoff, horizon, event_code=1):
    """Classification at 'risk >= cutoff' with death as competing event.
    Sensitivity/specificity use IPCW (cases = KF by horizon; non-cases = everyone else
    with known status). PPV/NPV use Aalen-Johansen incidence within the flagged/unflagged
    groups. NNR (number needed to refer to find one future KF) = 1 / PPV."""
    time, status, risk = map(np.asarray, (time, status, risk))
    flag = risk >= cutoff
    w = ipcw_weights(time, status, horizon)
    case = (status == event_code) & (time <= horizon)
    ctrl = (w > 0) & ~case
    sens = np.sum(w[case] * flag[case]) / np.sum(w[case])
    spec = np.sum(w[ctrl] * ~flag[ctrl]) / np.sum(w[ctrl])
    ppv = aalen_johansen(time[flag], status[flag], horizon, event_code) if flag.any() else np.nan
    npv = 1 - aalen_johansen(time[~flag], status[~flag], horizon, event_code) if (~flag).any() else np.nan
    return {"n_flagged": int(flag.sum()), "pct_flagged": float(flag.mean()),
            "events_captured": int((case & flag).sum()), "events_total": int(case.sum()),
            "sensitivity": float(sens), "specificity": float(spec),
            "PPV": float(ppv), "NPV": float(npv), "NNR": float(1 / ppv) if ppv > 0 else np.inf}


def net_benefit(time, status, risk, pt, horizon, event_code=1):
    """Net benefit at threshold pt, outcome = cumulative incidence with competing risks
    (Vickers 2008 extension; observed incidence in the flagged group by Aalen-Johansen)."""
    time, status, risk = map(np.asarray, (time, status, risk))
    flag = risk >= pt
    if not flag.any():
        return 0.0
    cif = aalen_johansen(time[flag], status[flag], horizon, event_code)
    return float(flag.mean() * (cif - (1 - cif) * pt / (1 - pt)))


def net_benefit_treat_all(time, status, pt, horizon, event_code=1):
    cif = aalen_johansen(time, status, horizon, event_code)
    return float(cif - (1 - cif) * pt / (1 - pt))


def aalen_johansen_curve(time, status, event_code=1, horizon=None):
    """Step function of the Aalen-Johansen cumulative incidence: (times, cif) starting at (0, 0)."""
    time = np.asarray(time, float)
    status = np.asarray(status, int)
    order = np.argsort(time, kind="stable")
    t, s = time[order], status[order]
    uniq, first = np.unique(t, return_index=True)
    n_at_risk = len(t) - first
    d_event = np.add.reduceat((s == event_code).astype(float), first)
    d_any = np.add.reduceat((s > 0).astype(float), first)
    surv_before = np.concatenate([[1.0], np.cumprod(1 - d_any / n_at_risk)[:-1]])
    cif = np.cumsum(surv_before * d_event / n_at_risk)
    if horizon is not None:
        keep = uniq <= horizon
        uniq, cif = uniq[keep], cif[keep]
    return np.concatenate([[0.0], uniq]), np.concatenate([[0.0], cif])
