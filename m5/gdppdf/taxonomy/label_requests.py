"""Class-level labels of task requests (task type, asks, inline material, output format, ...), with DeepSeek.

Used twice with the same prompt: on the GDP.pdf test prompts (only to fit class-level frequencies; nothing from a
test item reaches the generator) and on generated problems (spread check against the targets).

  python -m m5.gdppdf.taxonomy.label_requests --test --out <labels.jsonl>
  python -m m5.gdppdf.taxonomy.label_requests --problems <problems.jsonl> --out <labels.jsonl>
"""
import argparse
import asyncio
import json
import re
from pathlib import Path

from m5.gdppdf.llm import Server

TASK_TYPES = {
    'extract_compile': 'locate specific facts, figures, items or passages spread over the document and compile them '
                       '(list, table, summary) under the stated conditions',
    'calculate': 'compute values from the document\'s tables or figures (changes, totals, ratios, payouts, '
                 'premiums, conversions, derived quantities)',
    'apply_rules': 'apply the document\'s rules, criteria, guideline, policy or contract terms to a concrete case '
                   '(eligibility, coverage, compliance, clinical or technical decision)',
    'reconcile_check': 'cross-check items against each other or against requirements: find what is missing, '
                       'inconsistent, non-compliant or discrepant',
    'assess_recommend': 'weigh evidence from the document to assess a situation, select among options or recommend a '
                        'course of action with justification',
}
FORMATS = ['table', 'bulleted_list', 'numbered_list', 'short_answer', 'prose_summary', 'memo_or_report',
           'worked_calculation', 'mixed']

PROMPT = """Below is a request that a professional sent to an AI assistant together with one attached PDF document \
(the document is not shown here). Classify the request.

Request:
<<<
{prompt}
>>>

Return one JSON object with:
- "task_type": the main kind of work, one of {types}
- "secondary_type": another of those types if the request clearly also needs it, else null
- "n_asks": number of distinct deliverables or questions the request asks for (integer)
- "inline_material": true if the request itself supplies case facts, a scenario, a vignette, data or rules beyond \
the question (more than one sentence of such material)
- "inline_kind": if inline_material, one of "case_scenario", "client_or_patient_facts", "added_rules_or_checklist", \
"data_or_list", "other"; else null
- "output_format": one of {formats}
- "persona": true if the requester states their role or situation (e.g. "I'm a claims adjuster", "our HR team")
- "points_to_location": true if the request names specific pages, sections, tables or figures of the document
- "constraints": number of explicit constraints on content or format (exclusions, sorting, limits, "do not ...")
- "specificity": 1-3 (1 = broad summary request, 3 = narrowly specified items or numbers)

Definitions of task types:
{defs}"""


def build(prompt):
    return PROMPT.format(prompt=prompt, types=', '.join(f'"{t}"' for t in TASK_TYPES),
                         formats=', '.join(f'"{f}"' for f in FORMATS),
                         defs='\n'.join(f'- {k}: {v}' for k, v in TASK_TYPES.items()))


def parse_json(text):
    m = re.search(r'\{.*\}', text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except ValueError:
        return None


async def label(items, out_path, server, concurrency=32):
    done = set()
    if Path(out_path).exists():
        done = {json.loads(l)['id'] for l in open(out_path) if l.strip()}
    todo = [it for it in items if it['id'] not in done]
    out = open(out_path, 'a')
    sem = asyncio.Semaphore(concurrency)

    async def one(it):
        async with sem:
            body = {'messages': [{'role': 'user', 'content': build(it['prompt'])}], 'temperature': 0.2,
                    'max_tokens': 400, 'chat_template_kwargs': ({'enable_thinking': False} if 'super' in server.name
                                                                else {'thinking': False})}
            for _ in range(3):
                try:
                    res = await server.chat(body, affinity=it['id'])
                except (RuntimeError, ValueError) as e:
                    print(e, flush=True)
                    continue
                lab = parse_json(res['content'])
                if lab:
                    out.write(json.dumps({'id': it['id'], **{k: v for k, v in it.items() if k not in ('prompt',)},
                                          'words': len(it['prompt'].split()), 'label': lab}) + '\n')
                    out.flush()
                    return

    await asyncio.gather(*(one(it) for it in todo))
    out.close()


async def amain(a):
    if a.test:
        from aa_evals.gdppdf import data
        items = [{'id': t.task_id, 'domain': t.domain, 'text_only': t.task_id not in data.TEXT_ONLY_EXCLUDED,
                  'prompt': t.prompt} for t in data.load_tasks()]
    else:
        items = [json.loads(l) for l in open(a.problems) if l.strip()]
        items = [{'id': p['problem_id'], 'domain': p.get('domain'), 'prompt': p['prompt']} for p in items]
    server = await Server(a.server, 'super' if 'super' in a.server else 'ds', cap=a.concurrency).start()
    try:
        await label(items, a.out, server, a.concurrency)
    finally:
        await server.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--problems', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--server', default='m5-ds')
    ap.add_argument('--concurrency', type=int, default=32)
    asyncio.run(amain(ap.parse_args()))


if __name__ == '__main__':
    main()
