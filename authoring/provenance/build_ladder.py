"""Ladder + cheats table for one shipped seed, matching evidence/shipped_ladder_and_cheats.json shape.
Usage: python build_ladder.py <seed>"""
import json, sys
import gen_fleet as gf, solvers as sv

seed = int(sys.argv[1])
fl = gf.make_fleet(seed)
truth = fl["truth"]
n = len(truth)
n_fault = sum(1 for t in truth.values() if t["condition"] != "benign")

def gate(r):
    return r["misses"] <= 1 and r["false_alarms"] <= 1 and r["type_err"] <= 1 and r["act_err"] <= 3

routes = {
    "expert workflow (reference)": lambda f: sv.solve_a(f),
    "expert without the outlier screen": lambda f: sv.solve_a(f, screen=False),
    "expert without the laboratory-step correction": lambda f: sv.solve_a(f, correct=False),
    "raw last-3-year slopes": sv.solve_b,
    "latest sample only (level screen + Duval-style zones)": sv.solve_naive,
}

out = {}
for name, fn in routes.items():
    r = sv.score(fn(fl), truth)
    r["n_fault"], r["n"] = n_fault, n
    r["passes"] = gate(r)
    out[name] = r

# cheats built directly against truth (no solver involved)
units = list(truth.keys())
def cheat_score(pred):
    r = sv.score(pred, truth)
    r["n_fault"], r["n"] = n_fault, n
    r["passes"] = gate(r)
    return r

out["cheat: every unit benign / routine"] = cheat_score({u: {"condition": "benign", "action": "routine"} for u in units})
out["cheat: every unit thermal / investigate"] = cheat_score({u: {"condition": "thermal", "action": "investigate"} for u in units})
out["cheat: every unit arcing / remove_from_service"] = cheat_score({u: {"condition": "arcing", "action": "remove_from_service"} for u in units})

import random
rng = random.Random(12345)
conds = ["benign", "thermal", "partial_discharge", "arcing"]
acts = ["routine", "increase_sampling", "investigate", "remove_from_service"]
out["cheat: random labels"] = cheat_score({u: {"condition": rng.choice(conds), "action": rng.choice(acts)} for u in units})

json.dump(out, open("../evidence/shipped_ladder_and_cheats.json", "w"), indent=1)
for name, r in out.items():
    print(f"{name:55s} miss{r['misses']} fa{r['false_alarms']} type{r['type_err']} act{r['act_err']} pass={r['passes']}")
