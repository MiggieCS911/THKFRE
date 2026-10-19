"""Fine-Gray subdistribution hazard model in pure numpy.

Implementation follows R `survival::finegray()` + `coxph(..., weights = fgwt)`:

1. Data expansion: subjects with the competing event stay in the risk set after
   their event time, with weight G(t) / G(t_j), where G is the Kaplan-Meier
   estimate of the censoring distribution and t_j their competing-event time.
   Their follow-up is split at each later censoring time so the weight can change.
2. Weighted Cox partial likelihood on (start, stop] data, Efron tie handling,
   maximised by Newton-Raphson with the analytic information matrix.

Verified against R 4.3 (`tests/test_finegray.py`): coefficients agree to 1e-6.
Standard errors are model-based (as `coxph` without `robust = TRUE`).
"""
from dataclasses import dataclass

import numpy as np


def _censoring_km(time, status):
    """Kaplan-Meier of the censoring distribution. Returns jump times and G after each jump."""
    cens_times = np.unique(time[status == 0])
    g, out = 1.0, []
    for u in cens_times:
        n_risk = (time >= u).sum()
        n_cens = ((time == u) & (status == 0)).sum()
        g *= 1.0 - n_cens / n_risk
        out.append(g)
    return cens_times, np.array(out)


def _g_left(x, jump_times, g_vals):
    """G(x-): censoring survival just before x."""
    idx = np.searchsorted(jump_times, x, side="left") - 1
    return 1.0 if idx < 0 else g_vals[idx]


def expand_finegray(time, status, event_code=1):
    """Return start, stop, event, weight, row index arrays (counting-process form)."""
    time = np.asarray(time, float)
    status = np.asarray(status, int)
    jt, gv = _censoring_km(time, status)
    end = time.max() + 1.0
    start, stop, event, weight, row = [], [], [], [], []
    for i, (t, s) in enumerate(zip(time, status)):
        if s == 0 or s == event_code:
            start.append(0.0); stop.append(t); event.append(int(s == event_code))
            weight.append(1.0); row.append(i)
            continue
        # competing event: at risk until t with weight 1, then weighted pieces
        start.append(0.0); stop.append(t); event.append(0); weight.append(1.0); row.append(i)
        g_tj = _g_left(t + 1e-9, jt, gv)
        cuts = [u for u in jt if u > t] + [end]
        a = t
        for c in cuts:
            w = _g_left(a + 1e-9, jt, gv) / g_tj if g_tj > 0 else 0.0
            start.append(a); stop.append(c); event.append(0); weight.append(w); row.append(i)
            a = c
    return (np.array(start), np.array(stop), np.array(event), np.array(weight), np.array(row))


def _efron_terms(beta, X, start, stop, event, weight, event_times):
    """Log partial likelihood, score and information (weighted, Efron ties)."""
    eta = X @ beta
    r = weight * np.exp(eta)
    p = X.shape[1]
    ll, score, info = 0.0, np.zeros(p), np.zeros((p, p))
    for u in event_times:
        risk = (start < u) & (stop >= u)
        D = (stop == u) & (event == 1)
        rR, XR = r[risk], X[risk]
        S0, S1, S2 = rR.sum(), rR @ XR, (XR * rR[:, None]).T @ XR
        rD, XD = r[D], X[D]
        SD0, SD1, SD2 = rD.sum(), rD @ XD, (XD * rD[:, None]).T @ XD
        d = D.sum()
        wbar = weight[D].mean()
        ll += weight[D] @ eta[D]
        score += weight[D] @ XD
        for k in range(d):
            f = k / d
            den = S0 - f * SD0
            mean = (S1 - f * SD1) / den
            ll -= wbar * np.log(den)
            score -= wbar * mean
            info += wbar * ((S2 - f * SD2) / den - np.outer(mean, mean))
    return ll, score, info


@dataclass
class FineGrayResult:
    beta: np.ndarray
    cov: np.ndarray
    names: list
    base_times: np.ndarray        # distinct event times
    base_cumhaz: np.ndarray       # Breslow cumulative subdistribution hazard at eta = 0
    loglik: float
    n_iter: int

    @property
    def se(self):
        return np.sqrt(np.diag(self.cov))

    def cumhaz_at(self, t):
        idx = np.searchsorted(self.base_times, t, side="right") - 1
        return 0.0 if idx < 0 else self.base_cumhaz[idx]

    def predict_cif(self, X, t):
        """Cumulative incidence of the event of interest by time t for covariate rows X."""
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X[:, None]
        eta = X @ self.beta
        return 1.0 - np.exp(-self.cumhaz_at(t) * np.exp(eta))


def loglik_at(beta, X, time, status, event_code=1):
    """Fine-Gray log partial likelihood at a fixed beta (for likelihood-ratio tests)."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    start, stop, event, weight, row = expand_finegray(time, status, event_code)
    event_times = np.unique(stop[event == 1])
    return _efron_terms(np.atleast_1d(np.asarray(beta, float)), X[row], start, stop,
                        event, weight, event_times)[0]


def fit_cox(X, time, event, names=None):
    """Standard Cox model (Efron ties) = Fine-Gray with no competing events.
    `event`: 1 = event of interest, 0 = censored (competing events must be coded 0)."""
    return fit_finegray(X, time, np.asarray(event, int), event_code=1, names=names)


def fit_finegray(X, time, status, event_code=1, names=None, tol=1e-9, max_iter=50):
    """Fit a Fine-Gray model. X: (n, p) covariates; status 0=censored, event_code=event, other=competing."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    start, stop, event, weight, row = expand_finegray(time, status, event_code)
    Xe = X[row]
    event_times = np.unique(stop[event == 1])
    beta = np.zeros(X.shape[1])
    ll_old = None
    for it in range(1, max_iter + 1):
        ll, score, info = _efron_terms(beta, Xe, start, stop, event, weight, event_times)
        step = np.linalg.solve(info, score)
        # step halving if the likelihood decreased
        new = beta + step
        ll_new = _efron_terms(new, Xe, start, stop, event, weight, event_times)[0]
        h = 1.0
        while ll_new < ll - 1e-12 and h > 1e-6:
            h /= 2
            new = beta + h * step
            ll_new = _efron_terms(new, Xe, start, stop, event, weight, event_times)[0]
        beta = new
        if ll_old is not None and abs(ll_new - ll_old) < tol and np.max(np.abs(h * step)) < 1e-7:
            break
        ll_old = ll_new
    ll, score, info = _efron_terms(beta, Xe, start, stop, event, weight, event_times)
    cov = np.linalg.inv(info)

    # Breslow baseline cumulative subdistribution hazard (at eta = 0)
    r = weight * np.exp(Xe @ beta)
    inc = []
    for u in event_times:
        risk = (start < u) & (stop >= u)
        D = (stop == u) & (event == 1)
        inc.append(weight[D].sum() / r[risk].sum())
    return FineGrayResult(beta=beta, cov=cov,
                          names=names or [f"x{j}" for j in range(X.shape[1])],
                          base_times=event_times, base_cumhaz=np.cumsum(inc),
                          loglik=ll, n_iter=it)
