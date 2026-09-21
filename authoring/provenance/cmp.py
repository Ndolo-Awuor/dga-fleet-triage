import time, numpy as np
import gen_fleet as gf, solvers as sv
n = 40; t0 = time.time()
routes = {"A full": sv.solve_a, "raw slopes": sv.solve_b, "naive": sv.solve_naive}
res = {k: [] for k in routes}
for s in range(1, n + 1):
    fl = gf.make_fleet(s)
    for k, f in routes.items():
        res[k].append(sv.score(f(fl), fl["truth"]))
print("done %.0fs" % (time.time() - t0))
for g in [(1, 1, 1, 3), (1, 1, 1, 2), (1, 2, 2, 4)]:
    print("gate", g, {k: sum(r["misses"] <= g[0] and r["false_alarms"] <= g[1] and r["type_err"] <= g[2] and r["act_err"] <= g[3] for r in v) for k, v in res.items()}, "of", n)
for k, v in res.items():
    print(f"{k:11s} mean FA {np.mean([r['false_alarms'] for r in v]):.2f} misses {np.mean([r['misses'] for r in v]):.2f} act_err {np.mean([r['act_err'] for r in v]):.2f}")
