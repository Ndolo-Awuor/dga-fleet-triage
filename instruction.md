# Transformer fleet diagnosis

## Context

Multi-year dissolved-gas records for a fleet of 24 oil-filled transformers. Some units have a developing fault.

## Inputs

`/app/data/samples.csv`, `/app/data/units.csv`, `/app/data/events.csv` - the records.

`/app/data/README.txt` - the file definitions, the conventions to follow, the allowed values, and the action rule.

## Deliverable

`/app/diagnosis.json` - a condition and an action for each of the 24 units:

```
{"T01": {"condition": "...", "action": "..."}, "T02": {"condition": "...", "action": "..."}}
```

## What is graded

Each unit's condition and action are compared with the true state of the fleet. Checked: real faults that are missed, benign units flagged as faults, wrong fault types, and wrong actions.

You have 14400 seconds to complete this task. Do not cheat by using online solutions or hints specific to this task.
