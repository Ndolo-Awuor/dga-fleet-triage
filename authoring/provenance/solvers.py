"""
Solvers and scoring for the fleet DGA task (authoring).

solve_a     : expert workflow. Estimate the fleet-wide lab step from common-mode changes, convert lab A to the
              reference lab, split each unit's record at top-up and degassing events, estimate generation rates
              over the last 3 years from within-segment robust pairwise slopes, judge significance against a
              bootstrap standard error, classify the fault from the gas pattern of the increments, and apply the
              stated action rule to the estimated rates.
solve_naive : snapshot method on the latest sample only (typical-level screen + Duval triangle 1 style zones).
solve_b     : independent route. Raw last-3-year slopes with no lab or event correction (a careful but
              simplistic trend analysis).
score       : compares predictions with ground truth.
"""
import datetime as dt
import numpy as np

GASES = ["H2", "CH4", "C2H6", "C2H4", "C2H2", "CO", "CO2"]


def dec_year(s):
    d = dt.date.fromisoformat(s)
    return d.year + (d - dt.date(d.year, 1, 1)).days / 365.25


def parse(fleet):
    units = {}
    reg = {u: r for u, _, _, r in fleet["units"]}
    for u, date, lab, *vals in fleet["samples"]:
        e = units.setdefault(u, {"t": [], "lab": [], "x": [], "ev": [], "region": reg[u]})
        e["t"].append(dec_year(date)); e["lab"].append(lab); e["x"].append(vals)
    for u, date, kind, detail in fleet["events"]:
        units[u]["ev"].append((dec_year(date), kind, detail))
    for u, e in units.items():
        o = np.argsort(e["t"])
        e["t"] = np.array(e["t"])[o]; e["lab"] = np.array(e["lab"])[o]; e["x"] = np.array(e["x"], float)[o]
    return units


def action_from_rates(rate_sum, c2h2):
    if rate_sum > 250 or c2h2 > 5:
        return "remove_from_service"
    if rate_sum > 60 or c2h2 > 1:
        return "investigate"
    return "increase_sampling"


# ------------------------------------------------------------------ expert workflow (lab column NOT used)
def _resid(M, y):
    """Residual of y after regressing on the columns of M (per-unit intercept and trend)."""
    beta, *_ = np.linalg.lstsq(M, y, rcond=None)
    return y - M @ beta


def _series(units, w0=2021.8, gas_idx=(0, 1, 2, 5)):
    """Clean log-series per region: units with no top-up/degassing inside the window, gases above 20 ppm."""
    out = {}
    for e in units.values():
        m = e["t"] >= w0
        if m.sum() < 6 or any(te >= w0 - 0.2 for te, _, _ in e["ev"]):
            continue
        t = e["t"][m]
        for g in gas_idx:
            x = e["x"][m, g]
            if np.median(x) < 20 or np.any(x < 3):
                continue
            out.setdefault(e["region"], []).append((t, np.log(x)))
    return out


def fit_steps_joint(units, cand=np.arange(2023.0, 2024.95, 0.05), iters=4):
    """One shared step size s = log(old reading / new reading) for the fleet (a lab's bias is common to its regions)
    and one step date per region. Per series: log x = intercept + trend*t + s*1[t < D_region]. Alternates between
    the pooled step size (Frisch-Waugh across all series) and each region's date (least residual on identical data)."""
    ser = _series(units)
    regions = [r for r, v in ser.items() if len(v) >= 6]
    if not regions:
        return {}, 1.0
    D = {r: 2023.9 for r in regions}
    s_hat = -0.15
    for _ in range(iters):
        # pooled step size given the dates
        num = den = 0.0
        for r in regions:
            for t, y in ser[r]:
                step = (t < D[r]).astype(float)
                if step.sum() < 1 or (1 - step).sum() < 1:
                    continue
                M = np.column_stack([np.ones_like(t), t - D[r]])
                st, yt = _resid(M, step), _resid(M, y)
                num += float(yt @ st); den += float(st @ st)
        if den > 0:
            s_hat = num / den
        # each region's date given the step size
        for r in regions:
            best = (np.inf, D[r])
            for Dc in cand:
                rss = 0.0
                for t, y in ser[r]:
                    step = (t < Dc).astype(float)
                    M = np.column_stack([np.ones_like(t), t - Dc])
                    rss += float(np.sum(_resid(M, y - s_hat * step) ** 2))
                if rss < best[0]:
                    best = (rss, Dc)
            D[r] = best[1]
    return D, s_hat


def detect_steps(units):
    D, _ = fit_steps_joint(units)
    return D


def step_factors(units, steps):
    _, s_hat = fit_steps_joint(units)
    return np.full(len(GASES), np.exp(-s_hat))


def segment_ids(t, events):
    cuts = sorted(te for te, k, _ in events if k in ("top_up", "degassing"))
    return np.searchsorted(cuts, t, side="right")


def pair_slopes(t, x, seg, idx, min_dt=0.7):
    out = [[] for _ in range(x.shape[1])]
    for ii in range(len(idx)):
        for jj in range(ii + 1, len(idx)):
            i, j = idx[ii], idx[jj]
            if seg[i] == seg[j] and t[j] - t[i] >= min_dt:
                for g in range(x.shape[1]):
                    out[g].append((x[j, g] - x[i, g]) / (t[j] - t[i]))
    return out


def rates_for_unit(e, xc, now, window, rng=None, repeat=0.05, rho=0.08):
    """Robust slope per gas over the window from within-segment pairs, with a standard error that never falls below
    the lab repeatability (about 5 % of the level plus a small absolute floor). If a degassing leaves too few samples
    for a slope, use the accumulated level: rate = (level - retained fraction of the pre-degassing level) / time since."""
    seg = segment_ids(e["t"], e["ev"])
    ng = len(GASES)
    for w in (window, 5.0):
        idx = np.where(e["t"] >= now - w)[0]
        sl = pair_slopes(e["t"], xc, seg, idx)
        if len(sl[0]) >= 3:
            break
    if len(sl[0]) < 3:
        degs = [te for te, k, _ in e["ev"] if k == "degassing" and te < now]
        if degs:
            te = max(degs)
            after = np.where(e["t"] > te)[0]; before = np.where(e["t"] <= te)[0]
            if len(after) and len(before):
                dtm = max(e["t"][after[-1]] - te, 0.25)
                x_last = xc[after[-1]]; x_prev = xc[before[-1]]
                rates = np.maximum((x_last - rho * x_prev) / dtm, 0.0)
                se = (repeat * np.maximum(x_last, 1) + np.array([1.5, 1.5, 1.5, 1.5, 0.4, 1.5, 1.5])) / dtm
                return rates, se
    if len(sl[0]) < 1:
        return np.zeros(ng), np.full(ng, np.inf)
    rates = np.array([np.median(x) if x else 0.0 for x in sl])
    t = e["t"][idx]; sg = seg[idx]
    sxx = 0.0
    for sid in np.unique(sg):
        ts = t[sg == sid]
        sxx += float(np.sum((ts - ts.mean()) ** 2))
    se = np.full(ng, np.inf)
    if sxx > 0:
        for g in range(ng):
            resid = []
            for sid in np.unique(sg):
                m = sg == sid
                xs = xc[idx][m, g]; ts = t[m]
                off = np.median(xs - rates[g] * ts)
                resid.extend(xs - rates[g] * ts - off)
            resid = np.array(resid)
            sigma = 1.4826 * np.median(np.abs(resid - np.median(resid))) if len(resid) > 2 else np.inf
            floor = repeat * float(np.median(xc[idx][:, g])) + (0.3 if g == 4 else 1.0)
            se[g] = max(sigma, floor, 0.3) / np.sqrt(sxx)
    return rates, se


K_SIG, FLOOR = 4.0, 16.0


def classify(rates, se, floor=None, k=None):
    floor = FLOOR if floor is None else floor
    k = K_SIG if k is None else k
    rate_sum = float(np.sum(rates[:5])); c2 = float(rates[4])
    se_sum = float(np.sqrt(np.sum(np.square(se[:5])))) if np.all(np.isfinite(se[:5])) else np.inf
    acetylene = c2 > 0.7 and c2 > k * se[4]
    fault = rate_sum > max(floor, k * se_sum) or acetylene
    if not fault:
        return {"condition": "benign", "action": "routine"}, rate_sum, c2
    if acetylene and c2 / max(rate_sum, 1e-9) > 0.05:
        cond = "arcing"
    elif rates[0] / max(rate_sum, 1e-9) > 0.75 and (rates[3] + rates[4]) / max(rate_sum, 1e-9) < 0.05:
        cond = "partial_discharge"
    else:
        cond = "thermal"
    return {"condition": cond, "action": action_from_rates(rate_sum, c2)}, rate_sum, c2


def refine_unit_dates(units, D_reg, s_hat, span=0.9, step=0.05, gain=0.25, w0=2021.8):
    """Given the shared step size, let each unit take its own step date when that clearly lowers its residual."""
    dates = {}
    for u, e in units.items():
        d0 = D_reg.get(e["region"])
        dates[u] = d0
        if d0 is None or any(te >= w0 - 0.2 for te, _, _ in e["ev"]):
            continue
        m = e["t"] >= w0
        if m.sum() < 6:
            continue
        t = e["t"][m]
        ys = [np.log(e["x"][m, g]) for g in (0, 1, 2, 5) if np.median(e["x"][m, g]) >= 20 and not np.any(e["x"][m, g] < 3)]
        if not ys:
            continue

        def rss(d):
            stp = (t < d).astype(float)
            if stp.sum() < 1 or (1 - stp).sum() < 1:
                return np.inf
            M = np.column_stack([np.ones_like(t), t - d])
            return sum(float(np.sum(_resid(M, y - s_hat * stp) ** 2)) for y in ys)
        base = rss(d0)
        best = min(((rss(d), d) for d in np.arange(d0 - span, d0 + span + 1e-9, step)), key=lambda z: z[0])
        if np.isfinite(base) and best[0] < (1 - gain) * base:
            dates[u] = best[1]
    return dates


def screen_outliers(units, dates, f):
    """Drop samples where every gas (including the very stable CO2) is low by a common factor relative to the
    neighbouring samples of the same segment: a sampling error, not a trend."""
    keep = {}
    for u, e in units.items():
        xc = e["x"].copy()
        d = dates.get(u)
        if d is not None:
            xc[e["t"] < d] *= f
        seg = segment_ids(e["t"], e["ev"])
        ok = np.ones(len(e["t"]), bool)
        for i in range(len(e["t"])):
            nb = [j for j in range(max(0, i - 2), min(len(e["t"]), i + 3)) if j != i and seg[j] == seg[i]]
            if len(nb) < 2:
                continue
            med = np.median(xc[nb], axis=0)
            if med[5] < 50 or med[6] < 500:
                continue
            r_co, r_co2 = xc[i, 5] / med[5], xc[i, 6] / med[6]
            if r_co < 0.78 and r_co2 < 0.78:              # CO and CO2 both low: gas lost at the sampling visit
                ok[i] = False
        keep[u] = ok
    return keep


def D_units(units, D):
    return [(u, D.get(e['region'])) for u, e in units.items()]


def solve_a(fleet, window=3.0, seed=0, refine=True, screen=True, correct=True):
    units = parse(fleet)
    now = max(e["t"].max() for e in units.values())
    D, s_hat = fit_steps_joint(units)
    dates = refine_unit_dates(units, D, s_hat) if refine else dict(D_units(units, D))
    keep = screen_outliers(units, dates, np.exp(-s_hat)) if screen else {u: np.ones(len(e['t']), bool) for u, e in units.items()}
    clean = {}
    for u, e in units.items():
        k = keep[u]
        e2 = dict(e); e2["t"] = e["t"][k]; e2["x"] = e["x"][k]; e2["lab"] = e["lab"][k]
        clean[u] = e2
    D, s_hat = fit_steps_joint(clean)
    dates = refine_unit_dates(clean, D, s_hat) if refine else dict(D_units(clean, D))
    f = np.exp(-s_hat) if correct else 1.0
    out = {}
    for u, e in clean.items():
        xc = e["x"].copy()
        d = dates.get(u)
        if d is not None:
            xc[e["t"] < d] *= f
        rates, se = rates_for_unit(e, xc, now, window)
        out[u], _, _ = classify(rates, se)
    return out


# ------------------------------------------------------------------ independent route: raw slopes, no corrections
def solve_b(fleet, window=3.0):
    units = parse(fleet)
    now = max(e["t"].max() for e in units.values())
    out = {}
    for u, e in units.items():
        idx = np.where(e["t"] >= now - window)[0]
        rates = np.zeros(len(GASES))
        if len(idx) >= 3:
            tt = e["t"][idx]
            for g in range(len(GASES)):
                # Theil-Sen slope over all pairs, no event or lab handling
                s = [(e["x"][j, g] - e["x"][i, g]) / (tt[jj] - tt[ii]) for ii, i in enumerate(idx) for jj, j in enumerate(idx) if jj > ii and tt[jj] - tt[ii] >= 0.5]
                rates[g] = np.median(s) if s else 0.0
        out[u], _, _ = classify(rates, se=np.zeros(len(GASES)), floor=25.0)
    return out


def solve_c(fleet, window=3.0):
    """Detect the regional step, split each record there (and at events); no factor correction."""
    units = parse(fleet)
    now = max(e["t"].max() for e in units.values())
    steps = detect_steps(units)
    out = {}
    for u, e in units.items():
        ev = list(e["ev"])
        tc = steps.get(e["region"])
        if tc is not None:
            ev.append((tc, "top_up", ""))
        e2 = dict(e); e2["ev"] = ev
        rates, se = rates_for_unit(e2, e["x"], now, window)
        out[u], _, _ = classify(rates, se)
    return out


# ------------------------------------------------------------------ naive snapshot baseline
def duval_zone(ch4, c2h4, c2h2):
    s = ch4 + c2h4 + c2h2
    if s <= 0:
        return "none"
    p4, p2, p1 = 100 * ch4 / s, 100 * c2h4 / s, 100 * c2h2 / s
    if p4 >= 98:
        return "PD"
    if p1 >= 13:
        return "arcing"
    return "thermal"


def solve_naive(fleet):
    units = parse(fleet)
    out = {}
    for u, e in units.items():
        h2, ch4, c2h6, c2h4, c2h2, co, co2 = e["x"][-1]
        flagged = h2 > 100 or ch4 > 75 or c2h6 > 75 or c2h4 > 50 or c2h2 > 3
        if not flagged:
            out[u] = {"condition": "benign", "action": "routine"}
            continue
        z = duval_zone(ch4, c2h4, c2h2)
        cond = {"PD": "partial_discharge", "arcing": "arcing", "thermal": "thermal", "none": "benign"}[z]
        tdcg = h2 + ch4 + c2h6 + c2h4 + c2h2 + co
        act = "remove_from_service" if tdcg > 1920 else "investigate"
        out[u] = {"condition": cond, "action": act}
    return out


# ------------------------------------------------------------------ scoring
def score(pred, truth):
    misses = false_alarms = type_err = act_err = 0
    n_fault = sum(1 for t in truth.values() if t["condition"] != "benign")
    for u, t in truth.items():
        p = pred.get(u, {"condition": "benign", "action": "routine"})
        tf, pf = t["condition"] != "benign", p["condition"] != "benign"
        if tf and not pf: misses += 1
        if pf and not tf: false_alarms += 1
        if tf and pf and p["condition"] != t["condition"]: type_err += 1
        if p["action"] != t["action"]: act_err += 1
    return {"misses": misses, "false_alarms": false_alarms, "type_err": type_err, "act_err": act_err, "n_fault": n_fault, "n": len(truth)}
