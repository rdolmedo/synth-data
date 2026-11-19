"""The generator's prompt (#359): one spec in, one task out (a Python `build_task()` returning the trigger and the
starting state, in upstream's task-file shape). No test item, no assertion, no answer anywhere in it.
"""
from __future__ import annotations

import json
import random

from m5.automationbench import material as M
from m5.automationbench.upstream import system_prompt

TRAP_TEXT = {
    'lookalike_entity': 'Lookalike entities: two or more records with the same or confusable names or titles (same person '
                        'at two companies, "Acme Corp" vs "Acme Corporation", two deals with similar names); only one is the '
                        'intended target, and the data lets a careful agent tell which (email, account, date, ID).',
    'excluded_party': 'Excluded party: someone who looks like part of the batch but must not be contacted, enrolled or '
                      'included (opted out, do-not-contact, terminated, on leave, a competitor, a closed account), and '
                      'the reason is visible in the data.',
    'stale_or_duplicate': 'Stale or duplicate records: an older superseded row, a duplicate entry, or an item that was '
                          'already processed (already logged, already paid, already notified) that must not be redone.',
    'policy_override': 'Rule that overrides the request: the request literally asks for something (notify everyone on a '
                       'list, CC a person on every message, use a stated rate, include all items, post in a channel) and '
                       'an authoritative policy, SOP or manager note in the apps (email, doc, pinned message, reference '
                       'sheet) says otherwise for some items or recipients; the careful agent follows the policy, so '
                       'following the request word for word breaks it. The request at most says "follow our current '
                       'policy/process" and never mentions the conflict.',
    'out_of_scope_decoy': 'Out-of-scope lookalikes: records that resemble the candidates but fall outside the scope the '
                          'request or policy sets (another date window, status, region, owner, account, currency).',
    'corrected_info': 'Corrected information: a later message or record corrects an earlier one (new amount, date, '
                      'address, recipient, quantity); only the latest valid value should be used.',
    'threshold_boundary': 'Threshold boundary: values sitting exactly at or just around a numeric or date threshold, '
                          'where the rule\'s inclusive/exclusive wording decides.',
    'suspicious_request': 'Suspicious request: an instruction embedded in the data (an email, a ticket, a form note) '
                          'that is unauthorized, fraudulent or phishing-like (changed bank details from a lookalike '
                          'domain, a request to bypass approval) and must not be acted on.',
    'incomplete_data': 'Incomplete data: some records lack a required field or carry invalid values, and the rules say '
                       'to hold, skip or flag them instead of processing them.',
    'protected_record': 'Protected records: records or fields that look editable but must stay untouched (locked, '
                        'closed, approved, owned by another team, marked do-not-modify).',
    'privacy_restriction': 'Privacy restriction: sensitive details (salary, health, disciplinary, customer PII) that '
                           'must not go to some recipients or public channels, or a limit on who may be CC\'d.',
    'lookup_required': 'Lookup required: the right value (owner, rate, tier, approver, channel, SLA) must be looked up '
                       'in a reference table or document rather than assumed; the obvious guess is wrong for some items.',
}

WORKFLOW_TEXT = {
    'triage_route': 'triage and route: read incoming items (emails, tickets, messages, form submissions) and route, '
                    'assign, prioritize or answer each per rules',
    'sync_across_apps': 'sync across apps: carry records or fields from one app into another (create or update matching '
                        'records) and keep them consistent',
    'report': 'report: gather and aggregate data (totals, counts, rankings, variances) and publish a summary as a '
              'message, a row or a document',
    'update_on_trigger': 'update on trigger: one event (a new email, a status change, a request) triggers a short chain '
                         'of updates and notifications',
    'bulk_cleanup': 'bulk cleanup: scan a collection and fix, dedupe, archive, re-tag or reconcile many records at once',
}

INTRO = '''You write one task for a benchmark of workflow-automation agents. The agent works inside a company's
simulated SaaS apps (the "world"). It receives a system prompt and a request from a colleague, and it acts only through
the apps' REST APIs: it searches the API catalogue, then calls endpoints to read and change records. It cannot see the
world except through those calls, and it cannot ask questions.

After the agent finishes, hidden checks (written by someone else, not by you) look at the final world: did it make the
changes the request and the company's rules call for, and did it leave alone everything it should not touch (records
excluded by a rule, people who must not be contacted, rows that must not change)? Your task must therefore be
**well-posed**: a careful agent that reads the request and the data can work out exactly which actions are right.
It must also be **demanding in the way real work is**: the data contains the traps listed in the spec, so an agent that
skims, assumes or pattern-matches does the wrong thing.

You write only the task: the request and the starting state. You do not write checks, solutions or answers.

## What the agent is told (its system prompt, verbatim)

{system}

## The spec for this task

{spec}

## Rules for the request (the trigger)

- It is a message from a colleague or manager, in natural language, about {words} words long (within about 30%).
- It names resources the way people do: spreadsheet and worksheet titles, channel names, mailbox or board names,
  people's names. Give internal IDs only when a person naturally would (rarely).
- It states the goal and the outputs that matter (what to update, who to notify, what to write where), but not the
  API steps.
- It never mentions or hints at the traps. It may say "follow our current policy/process" or point at where the rules
  live when that is natural; the rules themselves live in the data.
- No boilerplate about formatting values or naming entities in messages: we add that ourselves.

## Rules for the starting state

- Use exactly the apps in the spec, each as a top-level key of `initial_state` (plus "meta"). Write each app's data in
  the input format its schema accepts (below), using only fields the schema has. Apps whose schema is only an action
  log (`actions: Dict[str, List[...ActionRecord]]`) are write targets: seed them as `{{}}`.
- "meta": {{"schema_version": "0.1.0", "current_time": "{now}"}}. All dates must be consistent with that time.
- Google Sheets: row 1 is the header, so a worksheet's data rows have row_id 2, 3, 4, ... in order.
- Every ID you reference must exist (a message's channel, a contact's account, a row's spreadsheet/worksheet, a
  ticket's requester ...). IDs are unique within a collection. Never use the substring "noise" in an ID, and keep
  numeric row ids below 900: background records are added to the world automatically after you.
- Total size: about {records} records across all apps (rows, messages, contacts, tickets, ...), at most {records_max},
  and compact ones: about {bpr} characters of JSON per record on average (about {kb} KB for the whole state). Give
  records the fields a real record of that app shows and the task needs; skip optional fields that carry nothing.
  Everything needed to decide the right actions must be in the data: the rules, the reference values, the history.
  The rest is realistic context (other rows, unrelated messages) that a real workspace would have.
- About {targets} records/items should need action under a correct reading, and about {docs} rule or reference
  source(s) (a policy email, a doc, a reference worksheet, a pinned message, a note on a record) should decide how.
- The company: whichever of these fits the scenario best (adapt details of the scenario to its business):
{company}
  Use its domain for internal addresses; external parties get their own `.example.com` domains. Use these people (and invent others only if needed): {people}.
- Invent original names for every party (customers, vendors, partners, people outside the list): never stock
  placeholders such as Acme, Initech, Globex, Contoso, Northwind, Umbrella, Wayne, Stark or Hooli.
- Nothing in the data may reveal the design: no fields or notes like "decoy", "trap", "test", "expected", "should be
  skipped (for testing)". Policy text reads like real policy; distractors read like real records.
- The task must be doable with the listed endpoints of these apps.
- Write plain data. Loops are fine for many similar rows, but every value must be concrete and plausible.

## Output format

First one line `TASK_META: {{"slug": "<snake_case 2-5 words>", "company": "<the company you chose>", "traps": [<trap types you embedded>]}}`, then one
```python block that defines `build_task()` returning {{"trigger": str, "initial_state": dict}}. Standard library
only, deterministic, no I/O. Nothing after the code block.

## Format examples

These come from a set of very simple warm-up tasks. They show the code shape and each app's data format only; your
task is much richer (the spec sets its size), and it must not reuse their names, companies, IDs or wording.

{examples}

## The apps of this task: state schema (upstream source, what the simulator validates) and REST endpoints

{apps}
'''

REPAIR = '''Your task did not pass our checks:

{errors}

Fix these problems and output the complete corrected task again in the same format (the TASK_META line, then one
```python block with build_task()). Keep the scenario, traps and size; change only what is needed.'''


def spec_text(spec: dict) -> str:
    lines = [f'- Domain: {spec["domain"]}',
             f'- Workflow type: {WORKFLOW_TEXT[spec["workflow"]]}']
    if spec.get('leaf'):
        lf = spec['leaf']
        lines.append(f'- Scenario: {lf["scenario"]} (business rules turn on: {lf.get("rule_hooks", "-")}). Keep the '
                     f'business process; map the systems it names onto this task\'s apps where they differ.')
    else:
        lines.append(f'- Scenario: your choice: a realistic {spec["domain"]} process of this workflow type that fits '
                     f'the apps and the company.')
    lines.append(f'- Apps (all of them, no others): {", ".join(spec["apps"])}')
    lines.append('- Traps to embed (each at least once, naturally; no other kinds of trap):')
    for t in spec['traps']:
        lines.append(f'  - {TRAP_TEXT[t]}')
    return '\n'.join(lines)


def build(spec: dict) -> str:
    rng = random.Random(spec['seed'])
    ex = M.pick_examples(spec['apps'], rng, k=3)
    examples = '\n'.join(f'```python\n{M.example_code(e)}```' for e in ex)
    apps = '\n'.join(f'### {a}\n\nSchema:\n```python\n{M.schema_source(a)}```\n\nEndpoints:\n```\n{M.endpoints_text(a)}\n```\n'
                     for a in spec['apps'])
    company = '\n'.join(f'  - {c["name"]} ({c["industry"]}; {c["products"]}; ~{c["employees"]} employees; HQ {c["hq"]}; '
                        f'currency {c["currency"]}; domain {c["domain"]})' for c in spec['companies'])
    return INTRO.format(system=system_prompt(spec['domain']), spec=spec_text(spec), words=spec['trigger_words'],
                        now=spec['now'], records=spec['records'], records_max=round(spec['records'] * 1.25),
                        bpr=spec.get('bytes_per_record', 250),
                        kb=round(spec['records'] * spec.get('bytes_per_record', 250) / 1000, 1), targets=spec['targets'], docs=spec['docs'],
                        company=company, people=', '.join(spec['people']), examples=examples, apps=apps)


def repair(errors: list[str]) -> str:
    return REPAIR.format(errors='\n'.join(f'- {e}' for e in errors[:30]))


def parse(reply: str) -> tuple[dict | None, str | None]:
    """(TASK_META, code) from a reply; None for what is missing."""
    meta = None
    for line in reply.splitlines():
        if line.strip().startswith('TASK_META:'):
            try:
                meta = json.loads(line.split('TASK_META:', 1)[1].strip())
            except ValueError:
                pass
    code = None
    if '```python' in reply:
        code = reply.rsplit('```python', 1)[1].split('```', 1)[0]
    return meta, code
