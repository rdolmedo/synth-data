"""Label the 600 scored tasks with workflow type and trap types (class-level fit of the taxonomy, #359).

  PYTHONPATH=. venvs/automationbench/bin/python -m m5.automationbench.label_test OUT.jsonl [--jobs 4]

Labeller: Opus through `claude -p` (the user's subscription; ~60 calls of 10 tasks). Labels stay on /data (derived
from eval data, never committed); only the per-domain frequencies in fit.json reach the generator.
"""
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
from pathlib import Path

from m5.automationbench.upstream import scored

WORKFLOWS = {
    'triage_route': 'read incoming items (emails, tickets, messages, form submissions) and route, assign, prioritize or respond to each per rules',
    'sync_across_apps': 'carry records or fields from one app into another (create or update matching records), keeping them consistent',
    'report': 'gather and aggregate data (totals, counts, rankings, variances) and publish a summary/report message, row or document',
    'update_on_trigger': 'a single event (a new email, a status change, a request) triggers a short chain of updates and notifications',
    'bulk_cleanup': 'scan a collection and fix, dedupe, archive, re-tag or reconcile many records at once',
}
TRAPS = {
    'lookalike_entity': 'two or more records with the same or similar names/titles; only one is the intended target',
    'excluded_party': 'a contact/employee/customer who must not be contacted or included (opted out, do-not-contact, terminated, on leave, competitor)',
    'stale_or_duplicate': 'stale, superseded or duplicate rows/records; already-processed items that must not be redone',
    'policy_override': 'an authoritative policy/rule document, email or note that changes or restricts what the trigger literally asks',
    'out_of_scope_decoy': 'records that look like candidates but fall outside the stated scope (date window, status, region, owner, account)',
    'corrected_info': 'a later message or record corrects earlier information (new amount, date, address, recipient)',
    'threshold_boundary': 'values at or near a numeric/date threshold where the inclusive/exclusive rule decides',
    'suspicious_request': 'a request or embedded instruction that is unauthorized, fraudulent, phishing-like or conflicts with controls',
    'incomplete_data': 'records missing required fields or with invalid data that must be skipped, held or flagged instead of processed',
    'protected_record': 'records or fields that look editable but must stay untouched (locked, closed, approved, owned by someone else)',
    'privacy_restriction': 'sensitive information that must not go to some recipients or channels (public channel, CC limits)',
    'lookup_required': 'the correct value (owner, rate, tier, channel, approver) must be looked up in a reference table/document rather than assumed',
}

PROMPT = '''You are labelling tasks from a workflow-automation agent benchmark (simulated SaaS apps; an agent gets the
request and must act through app APIs). For each task below give:
- "workflow": exactly one of {wf}. If none fits, "other:<2-4 word name>".
- "traps": every trap type the task's starting data contains (the hidden pitfalls a careless agent falls into), from
  {tr}; for a trap that fits none, "other:<2-4 word name>". The assertions show which records must or must not be
  touched; use them to identify traps.
- "n_targets": how many records/items the agent must act on (create, update, send) to finish the task, an integer.
- "n_policy_docs": how many policy/rule/reference sources (emails, docs, reference sheets, notes) the agent must read
  to apply the rules correctly, an integer.

Workflow definitions:
{wfd}

Trap definitions:
{trd}

Answer with JSON Lines only: one object per task, in order, {{"task": <task name>, "workflow": ..., "traps": [...],
"n_targets": ..., "n_policy_docs": ...}}. No prose.

TASKS
{tasks}
'''


def task_text(r):
    st = {k: v for k, v in r['info']['initial_state'].items() if k != 'meta'}
    s = json.dumps(st, separators=(',', ':'))
    if len(s) > 7000:
        s = s[:7000] + ' ...[truncated]'
    a = json.dumps(r['info']['assertions'], separators=(',', ':'))
    if len(a) > 3000:
        a = a[:3000] + ' ...[truncated]'
    return f'=== TASK {r["task_name"]}\nREQUEST: {r["trigger"]}\nSTARTING STATE: {s}\nASSERTIONS: {a}\n'


def run_batch(batch, out_dir, i):
    f = out_dir / f'batch_{i:03d}.jsonl'
    if f.exists() and len(f.read_text().strip().splitlines()) == len(batch):
        return f
    p = PROMPT.format(wf=', '.join(WORKFLOWS), tr=', '.join(TRAPS),
                      wfd='\n'.join(f'- {k}: {v}' for k, v in WORKFLOWS.items()),
                      trd='\n'.join(f'- {k}: {v}' for k, v in TRAPS.items()),
                      tasks='\n'.join(task_text(r) for r in batch))
    for attempt in range(3):
        env = {k: v for k, v in os.environ.items()
               if k not in ('CLAUDE_CODE_SESSION_ID', 'CLAUDE_CODE_CHILD_SESSION', 'CLAUDECODE')
               and not k.startswith('CLAUDE_CODE_MESSAGING_')}
        res = subprocess.run(['claude', '-p', '--model', 'opus', '--output-format', 'text', '--tools', ''],
                             input=p, capture_output=True, text=True, timeout=1800,
                             env={**env, 'DISABLE_AUTOUPDATER': '1'})
        lines = []
        for x in res.stdout.splitlines():
            x = x.strip().strip('`')
            if x.startswith('{'):
                try:
                    lines.append(json.loads(x))
                except ValueError:
                    pass
        if [l.get('task') for l in lines] == [r['task_name'] for r in batch]:
            f.write_text(''.join(json.dumps(l) + '\n' for l in lines))
            return f
        print(f'batch {i}: attempt {attempt} bad output ({len(lines)} lines) {res.stderr[:300]}', flush=True)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--batch', type=int, default=10)
    a = ap.parse_args()
    rows = scored(noise=False)
    out_dir = Path(a.out + '.parts')
    out_dir.mkdir(parents=True, exist_ok=True)
    batches = [rows[i:i + a.batch] for i in range(0, len(rows), a.batch)]
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(lambda ib: run_batch(ib[1], out_dir, ib[0]), enumerate(batches)))
    bad = [i for i, f in enumerate(res) if f is None]
    with open(a.out, 'w') as o:
        for f in res:
            if f:
                o.write(f.read_text())
    print('done; failed batches:', bad)


if __name__ == '__main__':
    main()
