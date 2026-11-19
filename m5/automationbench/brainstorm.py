"""Brainstorm the taxonomy's leaves (scenarios) per domain x workflow type, and the company / people pools.

  PYTHONPATH=. venvs/automationbench/bin/python -m m5.automationbench.brainstorm leaves OUT_DIR [--per-call 40 --calls 2]
  ... brainstorm companies OUT_DIR [--calls 25]

Opus via `claude -p`. The prompts carry only the domain, the workflow type's definition and the app list: no test
item, no upstream task text (the leaf rule of #359: no leaf from a single test item).
"""
import argparse
import concurrent.futures as cf
import json
from pathlib import Path

from m5.automationbench.opus import json_lines, opus

DOMAIN_DESC = {
    'sales': 'sales and revenue operations (CRM hygiene, leads, opportunities, quotes, contracts, meetings, pipeline reporting, account management)',
    'marketing': 'marketing (campaigns, email lists, ads, social media, events and webinars, content, lead capture and attribution)',
    'operations': 'business operations (vendors, procurement, inventory and orders, facilities, IT and project coordination, scheduling, compliance logistics)',
    'support': 'customer support (helpdesk tickets, escalations, SLAs, customer communication, refunds and returns, knowledge, support reporting)',
    'finance': 'finance and accounting (AP/AR, invoices, expenses, reconciliation, budgeting, journal entries, tax, treasury, audits)',
    'hr': 'human resources (recruiting, onboarding and offboarding, time off, payroll inputs, benefits, training and compliance, performance, employee communications)',
}
WORKFLOWS = {
    'triage_route': 'read incoming items (emails, tickets, messages, form submissions) and route, assign, prioritize or respond to each per rules',
    'sync_across_apps': 'carry records or fields from one app into another (create or update matching records), keeping them consistent',
    'report': 'gather and aggregate data (totals, counts, rankings, variances) and publish a summary/report message, row or document',
    'update_on_trigger': 'a single event (a new email, a status change, a request) triggers a short chain of updates and notifications',
    'bulk_cleanup': 'scan a collection and fix, dedupe, archive, re-tag or reconcile many records at once',
}

LEAF_PROMPT = '''We are building realistic practice scenarios for an AI agent that automates business workflows across
SaaS apps (Gmail, Google Sheets, Slack, Salesforce, HubSpot, helpdesks such as Zendesk/Freshdesk/Intercom,
accounting apps such as QuickBooks/Xero, calendars, Zoom, DocuSign, project tools, ...). A colleague sends the agent a
short request; the agent reads data in the apps and makes the changes and notifications.

Domain: {domain} — {ddesc}
Workflow type: {wf} — {wfdesc}

List {n} distinct, concrete scenarios of this workflow type that a real {domain} team at a mid-size company would
automate. Each should be specific (name the business process, what triggers it, what gets read and what gets
changed), not a generic category, and the scenarios should spread over the whole domain: different sub-processes,
document types, stakeholders, and business rules. Avoid near-duplicates of each other{avoid}.

Output JSON Lines only, one object per line: {{"slug": "<snake_case, 2-5 words>", "scenario": "<one or two sentences>",
"rule_hooks": "<the kind of business rule or reference data that decides what to do, a few words>"}}
'''


def leaves(out: Path, per_call: int, calls: int, jobs: int):
    out.mkdir(parents=True, exist_ok=True)

    def cell(d, w):
        f = out / f'{d}.{w}.jsonl'
        got = [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []
        for i in range(calls):
            if len(got) >= (i + 1) * (per_call - 8):
                continue
            avoid = ''
            if got:
                avoid = ' and of these, which we already have: ' + '; '.join(g['slug'] for g in got)
            p = LEAF_PROMPT.format(domain=d, ddesc=DOMAIN_DESC[d], wf=w, wfdesc=WORKFLOWS[w], n=per_call, avoid=avoid)
            new = [x for x in json_lines(opus(p)) if x.get('slug') and x.get('scenario')]
            have = {g['slug'] for g in got}
            got += [x for x in new if x['slug'] not in have]
            f.write_text(''.join(json.dumps(g) + '\n' for g in got))
        return d, w, len(got)

    with cf.ThreadPoolExecutor(jobs) as ex:
        for r in ex.map(lambda dw: cell(*dw), [(d, w) for d in DOMAIN_DESC for w in WORKFLOWS]):
            print(*r, flush=True)


COMPANY_PROMPT = '''Invent {n} fictional companies for realistic business-software test data. Spread them over many
industries (B2B and B2C, services, manufacturing, healthcare, education, logistics, retail, software, media,
nonprofits, hospitality, construction, energy, agriculture, finance, legal, ...), sizes (20 to 5,000 employees) and
regions (North America, Europe, Latin America, Asia-Pacific, Africa, Middle East). Names must be original and
plausible, not famous brands, and not generic placeholders like Acme, Globex, Initech, Contoso or Northwind; vary the
naming style (surnames, invented words, places, descriptive). Batch {i}: start from industries beginning with
letters {letters}.

Output JSON Lines only: {{"name": ..., "domain": "<lowercase web domain ending in .example.com>", "industry": ...,
"products": "<what they sell, a few words>", "employees": <int>, "hq": "<city, country>", "currency": "<ISO code>"}}
'''


def companies(out: Path, calls: int, jobs: int):
    out.mkdir(parents=True, exist_ok=True)
    letters = ['A-B', 'C-D', 'E-F', 'G-H', 'I-L', 'M-N', 'O-P', 'R-S', 'T-V', 'W-Z']

    def one(i):
        f = out / f'companies_{i:03d}.jsonl'
        if f.exists():
            return len(f.read_text().splitlines())
        rows = [x for x in json_lines(opus(COMPANY_PROMPT.format(n=60, i=i, letters=letters[i % len(letters)]),
                                           effort='medium')) if x.get('name')]
        f.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        return len(rows)

    with cf.ThreadPoolExecutor(jobs) as ex:
        print(sum(ex.map(one, range(calls))))


PEOPLE_PROMPT = '''List {n} {what} for realistic test data about employees, customers and vendors of companies around
the world. Mix origins broadly (Anglo, Hispanic, Lusophone, Francophone, German, Nordic, Slavic, Italian, Greek,
Turkish, Arabic, Persian, South Asian, East Asian, Southeast Asian, African, Caribbean, Pacific, Jewish, ...) roughly
in proportion to a large multinational's workforce. Plain ASCII spelling, no duplicates, no famous people. Batch {i}.

Output JSON Lines only: {{"name": ...}}
'''


def people(out: Path, calls: int, jobs: int):
    out.mkdir(parents=True, exist_ok=True)

    def one(args):
        what, i = args
        f = out / f'{what.split()[0]}_{i:02d}.jsonl'
        if not f.exists():
            rows = [x for x in json_lines(opus(PEOPLE_PROMPT.format(n=150, what=what, i=i), effort='medium')) if x.get('name')]
            f.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        return len(f.read_text().splitlines())

    with cf.ThreadPoolExecutor(jobs) as ex:
        print(sum(ex.map(one, [(w, i) for w in ('first names (given names, any gender)', 'surnames (family names)')
                                for i in range(calls)])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['leaves', 'companies', 'people'])
    ap.add_argument('out')
    ap.add_argument('--per-call', type=int, default=40)
    ap.add_argument('--calls', type=int, default=2)
    ap.add_argument('--jobs', type=int, default=3)
    a = ap.parse_args()
    if a.what == 'leaves':
        leaves(Path(a.out), a.per_call, a.calls, a.jobs)
    elif a.what == 'companies':
        companies(Path(a.out), a.calls, a.jobs)
    else:
        people(Path(a.out), a.calls, a.jobs)


if __name__ == '__main__':
    main()
