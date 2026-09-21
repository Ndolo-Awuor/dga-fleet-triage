"""Write the shipped instance: agent-visible CSVs (lab column hidden) and the sealed ground truth.
Usage: python build_instance.py <seed> <task_root>"""
import csv, json, sys, os
import gen_fleet as gf
seed, root = int(sys.argv[1]), sys.argv[2]
fl = gf.make_fleet(seed)
d = os.path.join(root, "environment", "data"); os.makedirs(d, exist_ok=True)
with open(os.path.join(d, "samples.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["unit", "date"] + gf.GASES)
    for u, date, lab, *vals in sorted(fl["samples"], key=lambda r: (r[0], r[1])):
        w.writerow([u, date] + vals)
with open(os.path.join(d, "units.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["unit", "oil_type", "commissioned_year", "region"])
    for row in sorted(fl["units"]): w.writerow(row)
with open(os.path.join(d, "events.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["unit", "date", "event", "detail"])
    for row in sorted(fl["events"], key=lambda r: (r[0], r[1])): w.writerow(row)
os.makedirs(os.path.join(root, "tests"), exist_ok=True)
truth = {u: {"condition": t["condition"], "action": t["action"], "rate_sum": round(t["rate_sum"], 2),
             "c2h2_rate": round(t["c2h2_rate"], 3), "kind": t["kind"]} for u, t in fl["truth"].items()}
json.dump(truth, open(os.path.join(root, "tests", "truth.json"), "w"), indent=1)
from collections import Counter
print("samples:", len(fl["samples"]), "| units:", len(fl["units"]), "| events:", len(fl["events"]))
print("conditions:", dict(Counter(t["condition"] for t in truth.values())), "| actions:", dict(Counter(t["action"] for t in truth.values())))
