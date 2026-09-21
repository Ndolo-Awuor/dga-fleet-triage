import sys, time, numpy as np
import gen_fleet as gf, solvers as sv
seeds = range(int(sys.argv[1]), int(sys.argv[2]))
rows = []; t0 = time.time()
for s in seeds:
    fl = gf.make_fleet(s)
    tr = fl["truth"]
    res = {n: sv.score(f(fl), tr) for n, f in (("A", sv.solve_a), ("C", sv.solve_c), ("B", sv.solve_b), ("naive", sv.solve_naive))}
    rows.append(res)
print(f"{len(rows)} fleets in {time.time()-t0:.0f}s; units/fleet 24; faults per fleet {np.mean([r['A']['n_fault'] for r in rows]):.1f}")
for n in ("A", "C", "B", "naive"):
    m = {k: np.array([r[n][k] for r in rows]) for k in ("misses", "false_alarms", "type_err", "act_err")}
    print(f"{n:6s} mean per fleet: misses {m['misses'].mean():.2f}  false_alarms {m['false_alarms'].mean():.2f}  type_err {m['type_err'].mean():.2f}  act_err {m['act_err'].mean():.2f}"
          f"   | max: {m['misses'].max()} {m['false_alarms'].max()} {m['type_err'].max()} {m['act_err'].max()}")
    for gate in [(1, 1, 1, 3), (0, 1, 1, 3), (1, 2, 2, 4)]:
        ok = (m['misses'] <= gate[0]) & (m['false_alarms'] <= gate[1]) & (m['type_err'] <= gate[2]) & (m['act_err'] <= gate[3])
        print(f"        gate misses<={gate[0]}, FA<={gate[1]}, type<={gate[2]}, act<={gate[3]}: pass {ok.sum()}/{len(rows)}")
