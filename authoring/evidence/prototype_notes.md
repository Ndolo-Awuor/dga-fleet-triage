# Prototype results: fleet DGA trend diagnosis (synthetic, seeded)

Generator: authoring/provenance/gen_fleet.py (24 units per fleet, ~8.7 faults, semi-annual samples 2012-2026).
Confounders: regional lab change (old lab reads ~15-30% lower, shared across gases) inside the last 3-year rate window, lab column hidden;
top-up dilution and degassing events (listed); stray-gassing units (benign, plateaued); sampling-error outliers; missing samples; noise.
Truth: condition (benign/thermal/partial_discharge/arcing) and action from the stated generation-rate rule over the last 3 years, reference = latest lab.

Gate used for the comparison: misses <= 1, false alarms <= 1, wrong fault type <= 1, wrong action <= 3 (of 24 units). 60 seeded fleets.

| route | pass |
|---|---|
| naive: latest sample only (typical-level screen + Duval-style zones) | 0/60 |
| raw last-3-year slopes, no lab or event handling | 26/60 |
| slopes split at events and at the detected lab step, no factor correction | 39/60 |
| expert: joint step fit (shared size, regional dates), correct, split at events, residual-based significance | 50/60 (56/60 with FA<=2, type<=2, act<=4) |

Step fit accuracy: date within 0.25 y in 82% of regions; shared step-size error 90th percentile |log| = 0.088.
Development history (what did not work): bootstrap SE inflated with 5-6 samples; step detection by max-statistic mislocated the date;
criteria compared on different data subsets preferred edge dates; per-gas step estimates were noisy (pooled to one shared size).
