"""Brainstorm taxonomy leaves below each fitted (partition, category, subject node) with Opus (`claude -p`).

Only class-level information reaches the brainstorm: category and subject names, the partition, and the general HLE
style. No HLE question is ever shown. Output: m5/hle/data/leaves/<partition>__<category>__<subject>.json.

    python -m m5.hle.brainstorm [--total 14000] [--per-leaf 12] [--jobs 6] [--only <substring>]
"""
import argparse
import concurrent.futures as cf
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'

HLE = """Humanity's Last Exam (HLE) is a benchmark of extremely hard, expert-written, closed-ended questions at the frontier
of human knowledge. Each question has one short, unambiguous, verifiable answer (an exact short answer, or one letter of
a multiple-choice list), cannot be answered by a quick web search, and needs real expertise. Questions come in two
partitions:
- reasoning: the solver must work the answer out (a derivation, computation, construction, proof-like argument,
  game analysis, puzzle) from what the question states plus standard expert knowledge;
- knowledge: the answer rests on an obscure specialist fact (from scholarly books, papers, editions, manuscripts,
  catalogues, case law, field guides...) PLUS a reasoning step on top of that fact (combine, compare, count, infer)."""

REASONING_SPEC = """Each leaf is a specific problem area for ORIGINAL problems with one exact answer, written by an expert
(not the result of a specific paper). JSON fields:
  "leaf": short specific name (3-8 words)
  "scope": one or two sentences on what problems in this leaf ask and why they are hard
  "techniques": 4-6 specific techniques, constructions or ideas a problem in this leaf could hinge on
  "answer_forms": 2-4 answer forms that fit (e.g. integer, rational number, closed-form expression, polynomial,
                  ordered tuple, small set, name of an object, a word/move/sequence, a letter for multiple choice)"""

KNOWLEDGE_SPEC = """Each leaf is a specific specialist topic where obscure but documented facts live, so that a question can
be written from a source document we give to the writer. Our source library holds: open-access academic books (DOAB),
Project Gutenberg and pre-1929 public-domain books, Library of Congress digitized books, Biodiversity Heritage Library
(natural history), LibreTexts textbooks, peer-reviewed open-access papers (peS2o, all fields), arXiv papers, and US case
law. Prefer topics that these sources cover in depth. JSON fields:
  "leaf": short specific name (3-8 words)
  "scope": one or two sentences on what obscure facts this leaf holds and what reasoning step a question could add
  "queries": 4-6 short keyword queries (2-5 words each, distinctive terms) that would retrieve relevant source documents
             by full-text search
  "sources": 1-3 of [doab, project_gutenberg, pre_1929_books, library_of_congress, biodiversity_heritage_library,
             libretexts, peS2o, arxiv, caselaw_access_project]"""


def prompt(part, cat, subj, n, other_subjects, existing):
    node = (f'the subject "{subj}" within the category "{cat}"' if subj != 'broad' else
            f'the category "{cat}", covering its areas OTHER than these subjects, which have their own lists: '
            f'{", ".join(other_subjects) or "(none)"}')
    ex = ''
    if existing:
        ex = '\nThese leaves already exist; do not repeat or near-repeat them:\n' + '\n'.join(f'- {e}' for e in existing)
    return f"""{HLE}

We are building a taxonomy to generate new training questions in HLE's style (never copies of HLE questions). Give
{n} distinct leaves for the {part} partition under {node}. Spread the leaves over the whole breadth of the area,
from mainstream graduate topics to niche corners, at the level of a specialist; leaves must be distinct from one
another and specific enough that two questions from the same leaf would still differ. Do not base any leaf on a
famous puzzle, a well-known exam question or a specific published problem.

{REASONING_SPEC if part == 'reasoning' else KNOWLEDGE_SPEC}
{ex}
Reply with only a JSON array of {n} objects, no prose."""


def call(p, timeout=1800):
    import os
    env = {k: v for k, v in os.environ.items() if not (k in ('CLAUDE_CODE_SESSION_ID', 'CLAUDE_CODE_CHILD_SESSION', 'CLAUDECODE')
                                                       or k.startswith('CLAUDE_CODE_MESSAGING_'))}
    r = subprocess.run(['claude', '-p', '--model', 'opus', p], capture_output=True, text=True, timeout=timeout,
                       cwd='/tmp', env=env)
    s = r.stdout
    m = re.search(r'\[.*\]', s, re.S)
    if not m:
        raise ValueError(f'no JSON array: {s[:300]} {r.stderr[:300]}')
    return json.loads(m.group(0))


def node_jobs(total, per_leaf, per_leaf_r):
    fit = json.loads((DATA / 'fit.json').read_text())
    jobs = []
    for part in ('reasoning', 'knowledge'):
        for cat, cw in fit['category'][part].items():
            subs = fit['subjects'][cat][part]
            for subj, sw in subs.items():
                exp = total * fit['partition'][part] * cw * sw
                n = int(min(300, max(8, round(exp / (per_leaf if part == 'knowledge' else per_leaf_r)))))
                jobs.append((part, cat, subj, n, [s for s in subs if s != 'broad'], exp))
    return jobs


def run_node(part, cat, subj, n, others, exp):
    key = f'{part}__{cat}__{subj}'.replace('/', '-').replace(' ', '_')
    out = DATA / 'leaves' / f'{key}.json'
    leaves = json.loads(out.read_text())['leaves'] if out.exists() else []
    tries = 0
    while len(leaves) < n and tries < 12:
        tries += 1
        k = min(50, n - len(leaves))
        try:
            got = call(prompt(part, cat, subj, k, others, [l['leaf'] for l in leaves]))
        except Exception as e:
            print(key, 'error', e, flush=True)
            continue
        seen = {l['leaf'].lower() for l in leaves}
        for g in got:
            if isinstance(g, dict) and g.get('leaf') and g['leaf'].lower() not in seen:
                leaves.append(g)
                seen.add(g['leaf'].lower())
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({'partition': part, 'category': cat, 'subject': subj, 'target_leaves': n,
                                   'expected_problems': round(exp), 'leaves': leaves}, indent=1) + '\n')
    print(key, len(leaves), '/', n, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--total', type=int, default=14000)
    ap.add_argument('--per-leaf', type=float, default=12, help='knowledge')
    ap.add_argument('--per-leaf-r', type=float, default=8, help='reasoning')
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--only')
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    jobs = node_jobs(a.total, a.per_leaf, a.per_leaf_r)
    if a.only:
        jobs = [j for j in jobs if a.only in f'{j[0]}__{j[1]}__{j[2]}']
    print(len(jobs), 'nodes', sum(j[3] for j in jobs), 'leaves')
    if a.dry:
        for j in jobs:
            print(j[0], j[1], j[2], j[3], round(j[5]))
        return
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(lambda j: run_node(*j), jobs))


if __name__ == '__main__':
    main()
