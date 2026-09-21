"""Reference solution: read the CSVs, run the expert workflow, write /app/diagnosis.json.
Usage: python solve.py [data_dir] [out_path]"""
import csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solvers as sv

data = sys.argv[1] if len(sys.argv) > 1 else "/app/data"
out = sys.argv[2] if len(sys.argv) > 2 else "/app/diagnosis.json"
rd = lambda name: list(csv.DictReader(open(os.path.join(data, name), newline="")))
samples = [(r["unit"], r["date"], "?", *[int(r[g]) for g in sv.GASES]) for r in rd("samples.csv")]
units = [(r["unit"], r["oil_type"], int(r["commissioned_year"]), r["region"]) for r in rd("units.csv")]
events = [(r["unit"], r["date"], r["event"], r["detail"]) for r in rd("events.csv")]
diagnosis = sv.solve_a({"samples": samples, "units": units, "events": events})
json.dump({u: {"condition": d["condition"], "action": d["action"]} for u, d in sorted(diagnosis.items())}, open(out, "w"), indent=1)
print("wrote", out, len(diagnosis), "units")
