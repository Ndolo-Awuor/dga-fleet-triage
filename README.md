# dga-fleet-triage

Diagnose a fleet of 24 oil-filled transformers from multi-year dissolved-gas records: which units have a developing
fault, of what kind, which are benign, and what action each needs. The records are **synthetic**, from a seeded
generator (`authoring/provenance/gen_fleet.py`, shipped instance seed 110); every gas-generation figure is a plausible,
disclosed choice and not data from any utility. The action rule is this task's own convention, stated in the data README.

## Difficulty

The obvious approach is to read the latest sample of each transformer: elevated gas levels, ratio methods, a
Duval-style zone. That fails here for a real reason. Benign units with oil-related ("stray") gassing sit at high but
stable levels and get flagged, real faults of different types get the wrong label, and the generation rate that decides
the action is not visible in a single sample. On the shipped instance it flags 3 benign units, gets 3 fault types wrong
and makes 7 wrong actions; on 60 fresh generated fleets it never passed the gate.

The generation rate has to come from the trends, and the trends are corrupted in ways a practitioner must recognise:

* **Laboratory change.** The laboratory changed within the last three years of the record, reads consistently higher
  than the previous one, hit different units on different dates, and the dates are not recorded. It appears as a
  common jump across a region's units and has to be found and corrected. Left in, it makes every high-level unit look like
  it is generating gas.
* **Top-ups and degassings** (listed, but their effect must be handled): dilution and reset change concentrations
  without changing generation. After a recent degassing the accumulated level, not a slope, carries the rate.
* **Sampling errors that come in pairs** (gas lost at the sampling visit) defeat a simple median. They show as a common
  drop in the stable gases (CO and CO2), which no fault of these types moves quickly.
* **Stray gassing** must be told from a low-temperature thermal fault by whether generation is still going on.

Wrong routes and what they produce on the shipped instance are in the ablation ladder below. Honest limits: the raw-slope
route is not hopeless (it passes about a quarter of generated fleets), and how often frontier agents solve this is
measured only by the platform's probes.

## Reference solution

`solution/solve.py` with `solution/solvers.py` (numpy only) reads the CSVs and writes `/app/diagnosis.json`:

1. Fit one shared laboratory step size and a step date per region, on identical data for every candidate date, with a
   separate trend for each unit and gas so real fault trends do not contaminate the step; then let each unit take its own
   date if that clearly lowers its residual.
2. Screen out samples where CO and CO2 both fall by a common factor against neighbours of the same segment, refit, and
   convert the older laboratory's readings to the newest laboratory's units.
3. Split each record at top-ups and degassings; take the median of within-segment pairwise slopes over the last three
   years; judge significance against a noise level that never falls below normal laboratory repeatability (about 5 % of the
   level); test acetylene on its own noise floor. After a recent degassing with too few samples, use the accumulated level.
4. Classify by the gas pattern of the rates and apply the stated action rule to the estimated rates.

Other valid strategies exist and the verifier does not require this one.

## Verification

`tests/test_triage.py` reads `/app/diagnosis.json`, normalises key case and spacing, and requires all 24 units with
allowed values. It compares with `tests/truth.json`, the seeded generator's fault or benign source for each unit and the
action computed from its true generation rates by the stated rule (not any solver's output). Four checks must all pass:

| check | limit (of 24 units, 10 real faults) |
|---|---|
| real faults called benign | at most 1 |
| benign units flagged as faults | at most 1 |
| wrong fault type among detected faults | at most 1 |
| wrong action, any unit | at most 3 |

The limits allow a competent expert to miss a borderline unit: the generator's noise means some low-rate faults are
genuinely hard. They were chosen after looking at the calibration results. The ground truth and gates are only in the
verifier image, and the histories are not published, so the answers cannot be looked up. Grading is deterministic.

**Calibration.** The generator and the reference were developed on seeds 1-40 (thresholds tuned there), then tested on 60
fresh fleets, seeds 101-160, never used for tuning. The shipped instance was chosen from those fresh seeds because the
reference is perfect on it and the weaker routes fail it. Results are in `authoring/evidence/`.

**Disclosures.** (1) The gates were set after seeing calibration results. (2) The instance was selected. (3) The
reference passes 46 of 60 fresh fleets (77 %), not all; its misses are marginal false alarms or low-rate faults. (4) The
weaker routes in the ladder are ablations of the reference, not independent solvers; an independent second solver by a
different route was not built. (5) The gas signatures, noise, and the action rule are the author's plausible choices and
have not been checked against a utility's practice.

## Ablation ladder

60 fresh generated fleets (seeds 101-160), gate as above:

| route | passes | mean misses | mean false alarms | mean action errors |
|---|---|---|---|---|
| expert workflow (reference) | 46 / 60 | 0.57 | 0.55 | 1.40 |
| without the outlier screen | 36 / 60 | 0.88 | 1.05 | 2.38 |
| without the laboratory-step correction | 16 / 60 | 0.37 | 2.63 | 3.97 |
| raw last-3-year slopes | 16 / 60 | 0.50 | 1.95 | 3.47 |
| latest sample only (level screen + Duval-style zones) | 0 / 60 | 0.33 | 2.78 | 5.62 |

Shipped instance (seed 110):

| route | missed | false alarms | wrong type | wrong action | passes |
|---|---|---|---|---|---|
| expert workflow (reference) | 0 | 0 | 0 | 0 | yes |
| without the outlier screen | 0 | 2 | 0 | 2 | no |
| without the laboratory-step correction | 0 | 3 | 0 | 4 | no |
| raw last-3-year slopes | 0 | 3 | 0 | 3 | no |
| latest sample only | 1 | 3 | 3 | 7 | no |
| cheat: every unit benign | 10 | 0 | 0 | 10 | no |
| cheat: every unit thermal / investigate | 0 | 14 | 6 | 18 | no |
| cheat: every unit arcing / remove_from_service | 0 | 14 | 7 | 22 | no |
| cheat: random labels | 2 | 9 | 5 | 18 | no |

The oracle scores 1 and the nop scores 0 on the real harness. Not done: agent trials (the platform's probes are the first
measurement) and an independent second solver.
