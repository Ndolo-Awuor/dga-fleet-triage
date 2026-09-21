import sys, time, pickle, numpy as np
import gen_fleet as gf, solvers as sv
lo, hi = int(sys.argv[1]), int(sys.argv[2])
routes = {
 "A full": lambda fl: sv.solve_a(fl),
 "A without outlier screen": lambda fl: sv.solve_a(fl, screen=False),
 "A without step correction": lambda fl: sv.solve_a(fl, correct=False),
 "raw slopes": sv.solve_b,
 "naive snapshot": sv.solve_naive,
}
res = {n: [] for n in routes}; t0 = time.time()
for s in range(lo, hi):
    fl = gf.make_fleet(s)
    for n, f in routes.items():
        res[n].append(sv.score(f(fl), fl["truth"]))
pickle.dump(res, open(f"../evidence/validate_{lo}_{hi}.pkl", "wb"))
print("done %.0fs" % (time.time() - t0))
N = hi - lo
for g in [(1, 1, 1, 3), (1, 2, 2, 4)]:
    print("gate misses<=%d FA<=%d type<=%d act<=%d:" % g, {n: sum(r["misses"] <= g[0] and r["false_alarms"] <= g[1] and r["type_err"] <= g[2] and r["act_err"] <= g[3] for r in v) for n, v in res.items()}, "of", N)
for n, v in res.items():
    print(f"  {n:26s} mean misses {np.mean([r['misses'] for r in v]):.2f}  FA {np.mean([r['false_alarms'] for r in v]):.2f}  type {np.mean([r['type_err'] for r in v]):.2f}  act {np.mean([r['act_err'] for r in v]):.2f}")
