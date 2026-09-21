"""
Sealed verifier. Reads /app/diagnosis.json and compares each unit's condition and action with the ground truth
of the seeded generator (tests/truth.json). Four gates must all hold (misses of real faults, false alarms on benign
units, wrong fault type among detected faults, wrong action anywhere). The file is read defensively: key case, spacing
and hyphens are normalised. A missing file, a missing unit, or a value outside the allowed sets fails.
Diagnostics are written to /logs/verifier/metrics.json.
"""
import json
import re
from pathlib import Path

import pytest

ARTIFACT = Path("/app/diagnosis.json")
HERE = Path(__file__).parent
TRUTH = json.loads((HERE / "truth.json").read_text())
GATES = json.loads((HERE / "gates.json").read_text())
CONDITIONS = {"benign", "thermal", "partial_discharge", "arcing"}
ACTIONS = {"routine", "increase_sampling", "investigate", "remove_from_service"}


def norm(v):
    return re.sub(r"[\s\-]+", "_", str(v).strip().lower())


@pytest.fixture(scope="module")
def result():
    assert ARTIFACT.exists(), f"missing {ARTIFACT}"
    raw = json.loads(ARTIFACT.read_text())
    assert isinstance(raw, dict), "diagnosis.json must be a JSON object keyed by unit"
    pred = {}
    for k, v in raw.items():
        assert isinstance(v, dict), f"entry for {k} must be an object"
        v = {norm(a): b for a, b in v.items()}
        assert "condition" in v and "action" in v, f"entry for {k} needs condition and action"
        pred[str(k).strip().upper()] = {"condition": norm(v["condition"]), "action": norm(v["action"])}
    for u in TRUTH:
        assert u in pred, f"missing unit {u}"
        assert pred[u]["condition"] in CONDITIONS, f"{u}: unknown condition {pred[u]['condition']!r}"
        assert pred[u]["action"] in ACTIONS, f"{u}: unknown action {pred[u]['action']!r}"
    m = {"misses": 0, "false_alarms": 0, "type_errors": 0, "action_errors": 0}
    detail = []
    for u, t in TRUTH.items():
        p = pred[u]
        tf, pf = t["condition"] != "benign", p["condition"] != "benign"
        if tf and not pf: m["misses"] += 1; detail.append((u, "missed fault"))
        if pf and not tf: m["false_alarms"] += 1; detail.append((u, "false alarm"))
        if tf and pf and p["condition"] != t["condition"]: m["type_errors"] += 1; detail.append((u, "wrong type"))
        if p["action"] != t["action"]: m["action_errors"] += 1; detail.append((u, "wrong action"))
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier") / "metrics.json").write_text(json.dumps({"metrics": m, "gates": GATES, "detail": detail}, indent=2))
    return m


def test_missed_faults(result):
    assert result["misses"] <= GATES["misses"], f"{result['misses']} real faults called benign (limit {GATES['misses']})"


def test_false_alarms(result):
    assert result["false_alarms"] <= GATES["false_alarms"], f"{result['false_alarms']} benign units flagged (limit {GATES['false_alarms']})"


def test_fault_types(result):
    assert result["type_errors"] <= GATES["type_errors"], f"{result['type_errors']} wrong fault types (limit {GATES['type_errors']})"


def test_actions(result):
    assert result["action_errors"] <= GATES["action_errors"], f"{result['action_errors']} wrong actions (limit {GATES['action_errors']})"
