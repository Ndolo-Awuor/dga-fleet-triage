import sys, time, numpy as np, pickle
import gen_fleet as gf, solvers as sv
routes = {
 "A full (joint step + per-unit dates + outlier screen + correction)": lambda fl: sv.solve_a(fl),
 "A without per-unit dates (regional date only)": lambda fl: sv.solve_a(fl, refine=False),
 "A without outlier screen": lambda fl: sv.solve_a(fl, screen=False),
 "A without step correction (split at fitted dates only)": lambda fl: sv.solve_a(fl, correct=False),
 "raw last-3-year slopes (no lab or event handling)": sv.solve_b,
 "naive latest-sample screen + Duval zones": sv.solve_naive,
}
res = {n: [] for n in routes}
t0 = time.time()
for s in range(1, 61):
    fl = gf.make_fleet(s)
    for n, f in routes.items():
        res[n].append(sv.score(f(fl), fl["truth"]))
print("done in %.0fs" % (time.time() - t0))
pickle.dump(res, open("../evidence/ablate_60.pkl", "wb"))
gates = [(1, 1, 1, 3), (1, 1, 1, 2), (1, 2, 1, 3), (0, 1, 1, 3), (1, 2, 2, 4)]
print(f"{'route':66s} " + " ".join(f"g{g}" for g in gates))
for n, rows in res.items():
    cells = []
    for g in gates:
        ok = sum(r["misses"] <= g[0] and r["false_alarms"] <= g[1] and r["type_err"] <= g[2] and r["act_err"] <= g[3] for r in rows)
        cells.append(f"{ok:>2d}/60".rjust(len(f"g{g}")))
    print(f"{n:66s} " + " ".join(cells))
print("gate tuple = (misses, false_alarms, type_err, act_err) maxima")
