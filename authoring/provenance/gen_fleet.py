"""
Seeded synthetic DGA fleet generator (authoring only; every number is a plausible, disclosed choice,
not data from any utility).

A fleet of oil-filled transformers is sampled roughly twice a year (2012-2026). Concentrations are simulated by
integrating gas generation and applying discrete events (top-up dilution, vacuum degassing), then the fleet-wide
laboratory change (lab A reports with per-gas bias relative to lab B; lab B is the reference), sampling errors,
noise and missing samples are applied.

Ground truth per unit
  condition : benign | thermal | partial_discharge | arcing
  action    : routine for benign units, otherwise by the stated rule from the mean generation rates over the last 3 years
              (reference-lab units): remove_from_service if rate_sum > 250 or C2H2 rate > 5 ppm/yr;
              investigate if rate_sum > 60 or C2H2 rate > 1 ppm/yr; else increase_sampling.
  rate_sum  = generation rate of H2 + CH4 + C2H6 + C2H4 + C2H2 (ppm/yr).
"""
import datetime as dt
import numpy as np

GASES = ["H2", "CH4", "C2H6", "C2H4", "C2H2", "CO", "CO2"]
T_START, T_END = 2012.0, 2026.0
STEP = 1.0 / 24.0

# share of the combustible generation rate carried by H2, CH4, C2H6, C2H4, C2H2
FRACS = {
    "thermal_low": [0.10, 0.42, 0.33, 0.149, 0.001],
    "thermal_high": [0.18, 0.20, 0.10, 0.517, 0.003],
    "partial_discharge": [0.88, 0.09, 0.03, 0.0, 0.0],
    "arcing": [0.55, 0.10, 0.03, 0.15, 0.17],
}
COND = {"thermal_low": "thermal", "thermal_high": "thermal", "partial_discharge": "partial_discharge", "arcing": "arcing"}


def action_from_rates(rate_sum, c2h2):
    if rate_sum > 250 or c2h2 > 5:
        return "remove_from_service"
    if rate_sum > 60 or c2h2 > 1:
        return "investigate"
    return "increase_sampling"


def to_date(y):
    year = int(np.floor(y))
    base = dt.date(year, 1, 1)
    return base + dt.timedelta(days=int((y - year) * 365.25))


def draw_target(r, cls):
    """target mean generation rates over the last 3 years, in reference units, well inside each action band"""
    sev = r.choice(["low", "mid", "high"], p=[0.35, 0.40, 0.25])
    if cls == "arcing":
        c2 = {"low": r.uniform(1.4, 2.8), "mid": r.uniform(3.0, 4.2), "high": r.uniform(8.0, 22.0)}[sev]
        return c2 / FRACS["arcing"][4], sev
    R = {"low": r.uniform(30, 42), "mid": r.uniform(90, 170), "high": r.uniform(350, 600)}[sev]
    return R, sev


def make_fleet(seed, n_units=24):
    r = np.random.default_rng(seed)
    n_th, n_pd, n_ar = int(r.integers(3, 5)), int(r.integers(2, 4)), int(r.integers(2, 4))
    n_stray = int(r.integers(3, 5))
    kinds = (["thermal"] * n_th + ["partial_discharge"] * n_pd + ["arcing"] * n_ar + ["stray"] * n_stray
             + ["normal"] * (n_units - n_th - n_pd - n_ar - n_stray))
    r.shuffle(kinds)
    ids = [f"T{i + 1:02d}" for i in range(n_units)]

    D_reg = {"North": float(r.uniform(2023.2, 2024.6)), "South": float(r.uniform(2023.2, 2024.6))}   # regional lab change (A -> B)
    c_common = float(r.uniform(0.62, 0.80))                                   # old lab A reads lower than reference lab B
    lab_factor = {g: float(c_common * np.exp(r.normal(0, 0.07))) for g in GASES}  # A reports true * factor

    # oil type: stray-gassing units are type B; keep at least 3 non-stray type-B units so type B is not a shortcut
    oil = {}
    nb_non_stray = 0
    for u, k in zip(ids, kinds):
        if k == "stray":
            oil[u] = "type_B"
        else:
            oil[u] = "type_B" if r.random() < 0.45 else "type_A"
            nb_non_stray += oil[u] == "type_B"
    non_stray = [u for u, k in zip(ids, kinds) if k != "stray"]
    while nb_non_stray < 3:
        u = non_stray[int(r.integers(0, len(non_stray)))]
        if oil[u] == "type_A":
            oil[u] = "type_B"; nb_non_stray += 1

    samples, units_rows, events_rows, truth = [], [], [], {}
    region = {u: ("North" if r.random() < 0.5 else "South") for u in ids}
    for u, k in zip(ids, kinds):
        D = float(np.clip(D_reg[region[u]] + r.normal(0, 0.30), 2022.9, 2024.9))   # staggered adoption per unit
        young_stray = (k == "stray" and r.random() < 0.3)
        tc = float(r.uniform(2011.0, 2012.6)) if young_stray else float(r.uniform(1990, 2010) if k != "stray" else r.uniform(1996, 2004))
        t0 = max(tc, T_START)
        tg = np.arange(t0, T_END + STEP / 2, STEP)
        n = len(tg)
        co0 = r.uniform(120, 320)
        L0 = np.array([r.uniform(6, 35), r.uniform(3, 18), r.uniform(2, 12), r.uniform(1, 7), r.uniform(0.0, 0.4), co0, co0 * r.uniform(8, 13)])
        base = np.array([r.uniform(0.3, 2.0), r.uniform(0.1, 1.0), r.uniform(0.1, 0.8), r.uniform(0.05, 0.5), 0.0, r.uniform(6, 20), r.uniform(60, 220)])

        # stray gassing: saturating generation of H2, CH4, C2H6 (and a little C2H4), no acetylene
        if k == "stray":
            xinf = np.array([r.uniform(60, 220), r.uniform(40, 140), r.uniform(30, 110), r.uniform(2, 12), 0.0, 0.0, 0.0])
            tau = r.uniform(1.2, 2.5) if young_stray else r.uniform(1.5, 4.0)
            L0 = L0 + xinf * (1 - np.exp(-(t0 - tc) / tau))
        # fault
        fault_gen = None
        if k in ("thermal", "partial_discharge", "arcing"):
            cls = {"thermal": r.choice(["thermal_low", "thermal_high"]), "partial_discharge": "partial_discharge", "arcing": "arcing"}[k]
            R_target, sev = draw_target(r, cls)
            t_on = float(r.uniform(T_END - 8.0, T_END - 3.3))
            t_on = max(t_on, t0 + 0.5)
            acc = float(r.uniform(0.0, 0.10))
            grid3 = np.linspace(T_END - 3.0, T_END, 200)
            avg3 = float(np.mean(1 + acc * (grid3 - t_on)))
            R0 = R_target / avg3
            cellulose = (k == "thermal") and r.random() < 0.35
            co_rate = float(r.uniform(30, 90)) if cellulose else 0.0
            fractions = np.array(FRACS[cls])

            def fault_gen(t, R0=R0, acc=acc, t_on=t_on, fractions=fractions, co_rate=co_rate, cellulose=cellulose):
                if t < t_on:
                    return np.zeros(7)
                R = R0 * (1 + acc * (t - t_on))
                g = np.zeros(7)
                g[:5] = fractions * R
                if cellulose:
                    g[5] = co_rate * (1 + acc * (t - t_on))
                    g[6] = g[5] * r_cell
                return g
            r_cell = float(r.uniform(3, 7))
            true_rates = fractions * R_target
            cond = COND[cls]
        else:
            true_rates = np.zeros(5); cond = "benign"

        # events
        tops, degs = [], []
        if r.random() < 0.40:
            tops.append((float(r.uniform(2019.0, 2025.4)), float(r.uniform(0.06, 0.16))))
        if r.random() < 0.12 + (0.10 if k in ("thermal", "partial_discharge", "arcing") else 0.0):
            degs.append((float(r.uniform(2019.0, 2025.0)), r.uniform(0.04, 0.12, size=7)))
        for tt, f in tops:
            events_rows.append((u, to_date(tt).isoformat(), "top_up", f"added_percent={100 * f:.1f}"))
        for tt, _ in degs:
            events_rows.append((u, to_date(tt).isoformat(), "degassing", "vacuum degassing"))

        # integrate
        c = np.zeros((n, 7)); c[0] = L0
        for i in range(1, n):
            t = tg[i - 1]
            gen = base.copy()
            if k == "stray":
                gen[:4] += xinf[:4] / tau * np.exp(-(t - tc) / tau)
            if fault_gen is not None:
                gen = gen + fault_gen(t)
            c[i] = c[i - 1] + gen * STEP
            for tt, f in tops:
                if tg[i - 1] < tt <= tg[i]:
                    c[i] *= (1 - f)
            for tt, rho in degs:
                if tg[i - 1] < tt <= tg[i]:
                    c[i] *= rho

        # sampling times
        ts = []
        t = t0 + r.uniform(0.0, 0.5)
        while t < T_END - 0.45:
            if r.random() > 0.12:
                ts.append(t)
            t += r.uniform(0.35, 0.65)
        ts.append(T_END - r.uniform(0.05, 0.4))
        ts = np.array(ts)
        prev_bad = False
        for tsi in ts:
            val = np.array([np.interp(tsi, tg, c[:, j]) for j in range(7)])
            lab = "A" if tsi < D else "B"
            if lab == "A":
                val = val * np.array([lab_factor[g] for g in GASES])
            bad = (r.random() < 0.05) or (prev_bad and r.random() < 0.5)             # sampling errors often come in pairs
            prev_bad = bad
            if bad:
                val = val * r.uniform(0.35, 0.7)                          # gas loss at the sampling visit
            val = val * np.exp(r.normal(0, 0.045, size=7)) + r.normal(0, 1.0, size=7) * np.array([1, 1, 1, 1, 0.3, 3, 20])
            val = np.maximum(np.round(val), 0)
            samples.append((u, to_date(tsi).isoformat(), lab, *[int(v) for v in val]))
        units_rows.append((u, oil[u], int(tc), region[u]))
        rs = float(np.sum(true_rates)); c2 = float(true_rates[4])
        truth[u] = {"condition": cond, "action": "routine" if cond == "benign" else action_from_rates(rs, c2),
                    "rate_sum": rs, "c2h2_rate": c2, "kind": k}
    return {"samples": samples, "units": units_rows, "events": sorted(events_rows), "truth": truth,
            "meta": {"lab_change_year": D_reg, "lab_factor": lab_factor, "seed": seed}}
