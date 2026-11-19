"""Tool-call health of a run: separates serving/parser trouble from model mistakes.

    python3 -m aa_evals.automationbench.inspect_run /data/aai/aa_evals/runs/automationbench/<run_id>

Per item: score, objectives, guardrails broken, turns, tool calls, and the tool errors grouped by
kind. Stdlib only, so any python3 can run it.
- `bad_json`: params/body that are not JSON. The string "None" means the history-rendering bug
  that the `xml_null_args` adapter fixes.
- `not_connected`: a 401 for an app the task does not have. A model mistake.
- `no_handler`: a 404 for an unknown endpoint path. A model mistake.
- `leaked`: tool-call markup left in the assistant text. A parser failure.
"""

from __future__ import annotations

import collections
import io
import json
import sys
from pathlib import Path
from aa_evals.common import jsonl


def classify(content: str) -> str | None:
    head = content[:200]
    if '"error"' not in head and "Error" not in head:
        return None
    if "Invalid JSON" in head:
        return "bad_json"
    if "account is connected" in head:
        return "not_connected"
    if "No handler" in head:
        return "no_handler"
    return "other_error"


def main(run_dir: str) -> None:
    run = Path(run_dir)
    items = jsonl.read(run / "items.jsonl", required=True)
    totals: collections.Counter = collections.Counter()
    none_args = 0
    tpath = run / "transcripts.jsonl"
    with (tpath.open("rb") if tpath.exists() else io.BytesIO()) as f:
        for r in items:
            if "transcript_offset" not in r:  # cutoff rows have no transcript
                print(f"{r['item_id']:45s} {r['status']}")
                continue
            f.seek(r["transcript_offset"])
            t = json.loads(f.readline())
            calls, errs = {}, collections.Counter()
            leaked = 0
            for m in t["messages"]:
                if m.get("role") == "assistant":
                    c = m.get("content")
                    if isinstance(c, str) and ("<tool_call>" in c or "<function=" in c):
                        leaked += 1
                    for tc in m.get("tool_calls") or []:
                        calls[tc["id"]] = tc
                        try:
                            args = json.loads(tc.get("arguments") or "{}")
                        except ValueError:
                            args = {}
                        none_args += sum(1 for v in args.values() if v == "None")
                elif m.get("role") == "tool":
                    c = m.get("content")
                    kind = classify(c if isinstance(c, str) else json.dumps(c))
                    if kind:
                        errs[kind] += 1
            totals.update(errs)
            totals["tool_calls"] += len(calls)
            totals["leaked"] += leaked
            sc = "-" if r.get("score") is None else f"{r['score']:.2f}"
            print(f"{r['item_id']:45s} {r['status']:11s} score={sc} obj={r.get('n_objectives_met')}/{r.get('n_objectives')} "
                  f"gb={r.get('n_guardrails_broken')} turns={r['turns']:2d} calls={len(calls):3d} "
                  f"stop={r.get('stop_condition')} errs={dict(errs)}" + (f" LEAKED={leaked}" if leaked else ""))
    print(f"\ntotals: {dict(totals)}; string 'None' args: {none_args}")


if __name__ == "__main__":
    main(sys.argv[1])
