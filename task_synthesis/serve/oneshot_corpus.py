"""Replay corpus of a one-shot round's DeepSeek requests, for synth.bench / synth.tune (issue #163).

  PYTHONPATH=<the round's frozen checkout> python task_synthesis/serve/oneshot_corpus.py --round m1-hero \
      --out /data/aai/tbench/tasksynth/m1-spec163/corpus --size 3000

A one-shot round keeps no request bodies: <round>/work/<spec>/record.json holds the spec, each attempt's usage and
check errors, attempt-<k>.reply.txt holds each reply's content, and attempt-<k>.check.json the check the judge saw.
This rebuilds every generator and judge request the round sent from those files with the round's own prompt code
(run it with the round's frozen checkout first on PYTHONPATH, the `code` in <round>/config.json), samples `--size` of
them uniformly, and writes one chat-completions body per request plus <out>/corpus.json in synth.bench's corpus-cache
format: one single-request trajectory per body, `recorded` = the round's prompt/completion tokens. Each rebuilt
prompt is counted with the teacher's encoder and compared with the recorded prompt tokens (`prompt_check`).

Bodies carry the production sampling and extra_body (the endpoint's `request.overrides`) and no max_tokens, as the
round sent them (`--max-tokens recorded` caps each at its recorded completion instead).
"""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import statistics
import sys

ROUNDS = Path('/data/aai/tbench/tasksynth/rounds')


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def requests_of(wd, rec, cfg):
    """[(kind, messages, recorded usage)] for one spec, in the order the round sent them."""
    from task_synthesis.pipeline import prompts, specs as specs_mod
    from task_synthesis.pipeline.judge import prompt as judge_prompt
    from task_synthesis.pipeline.package import parse
    from task_synthesis.pipeline.round import asset_files, system_prompt
    from task_synthesis.pipeline import explore
    spec = rec['spec']
    if cfg.get('author', 'oneshot') != 'oneshot' or cfg.get('order', 'spec') != 'spec':
        raise SystemExit('only one-shot rounds with --order spec are supported')
    _, asset_gen, asset_judge = asset_files(spec)
    messages = [{'role': 'system', 'content': system_prompt(spec['minutes'])},
                {'role': 'user', 'content': specs_mod.render(spec, asset_gen, agentic=False, asset_first=False,
                                                             workspace=False, tokens=cfg.get('author_tokens', 0),
                                                             lean=False, commands=explore.MAX_COMMANDS)}]
    out, content = [], None
    for att in rec['attempts']:
        k = att['k']
        if k > 0:
            prev = rec['attempts'][k - 1]
            errors = '\n'.join(f'- {e}' for e in prev.get('errors') or [])
            messages = messages + [{'role': 'assistant', 'content': content},
                                   {'role': 'user', 'content': prompts.REPAIR.format(errors=errors)}]
        try:
            content = (wd / f'attempt-{k}.reply.txt').read_text()
        except OSError:
            return out   # nothing later can be rebuilt
        if not att.get('replayed'):
            out.append((f'author{min(k, 1)}', messages, att.get('usage') or {}))
    j = rec.get('judge') or {}
    if rec['status'] == 'valid' and j.get('usage'):
        k = rec['attempts'][-1]['k']
        chk = read(wd / f'attempt-{k}.check.json')
        if chk is not None:
            out.append(('judge', judge_prompt(spec, parse(content), chk, asset_judge), j['usage']))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--round', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--size', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--endpoint', default='', help='serving name whose endpoints.json request overrides to use '
                                                    '(default: the round\'s first --gen endpoint)')
    ap.add_argument('--teacher', default='deepseek-v41-flash-1017')
    ap.add_argument('--max-tokens', choices=['none', 'recorded'], default='none')
    a = ap.parse_args()
    import task_synthesis
    rd = ROUNDS / a.round
    if (rd / 'packed' / 'index.json').exists():   # reads replies by path; unpack the specs first (roundpack, #245)
        raise SystemExit(f'{rd} is packed (roundpack): this tool reads loose work/ directories only')
    cfg = read(rd / 'config.json')
    code = Path(task_synthesis.__file__).resolve().parents[1]
    if Path(cfg['code']).resolve() != code:
        print(f'warning: importing task_synthesis from {code}, the round ran {cfg["code"]}', file=sys.stderr)
    from tbshared import serving   # $TASKSYNTH_SERVING_ROOT, else production (as serve_teacher)
    endpoint = read(serving.path(a.endpoint or cfg['gen'].split('+')[0]))
    overrides = dict(endpoint['request']['overrides'])
    sampling = {k: overrides.pop(k) for k in ('temperature', 'top_p') if k in overrides}

    def load(wd):
        rec = read(wd / 'record.json')
        if not rec or not rec.get('attempts'):
            return []
        try:
            return [(wd.name, i, *r) for i, r in enumerate(requests_of(wd, rec, cfg))]
        except Exception as e:  # a reply the round could parse must parse here too; report, do not stop
            print(f'{wd.name}: {e!r}'[:300], file=sys.stderr)
            return []

    dirs = sorted(p for p in (rd / 'work').iterdir() if p.is_dir())
    with ThreadPoolExecutor(32) as ex:
        reqs = [r for rs in ex.map(load, dirs) for r in rs]
    reqs = [r for r in reqs if r[4].get('completion')]
    kinds = collections.Counter(r[2] for r in reqs)
    print(f'{len(reqs)} recorded requests in {len(dirs)} spec dirs: {dict(kinds)}')
    random.Random(a.seed).shuffle(reqs)
    sample = sorted(reqs[:a.size], key=lambda r: (r[0], r[1]))

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from synth.config import load_teacher
    from synth.serializers import serializer_for
    ser = serializer_for(load_teacher(a.teacher))
    out = Path(a.out)
    (out / 'requests').mkdir(parents=True, exist_ok=True)
    trajectories, ratios = [], []
    for spec_id, i, kind, messages, usage in sample:
        body = {'model': endpoint['served_name'], 'messages': messages, **sampling, **overrides}
        if a.max_tokens == 'recorded':
            body['max_tokens'] = usage['completion']
        path = out / 'requests' / f'{spec_id}-{i}.json'
        path.write_text(json.dumps(body, ensure_ascii=False))
        counted = ser.count(ser.render(messages, None))
        ratios.append(counted / usage['prompt'])
        trajectories.append({'run': spec_id, 'kind': kind, 'requests': [str(path)], 'gaps': [None],
                             'recorded': {'prompt': usage['prompt'], 'completion': usage['completion'],
                                          'finish': usage.get('finish'), 'counted_prompt': counted}})
    q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))]
    check = {'ratio_p01': round(q(ratios, .01), 4), 'ratio_p50': round(q(ratios, .5), 4),
             'ratio_p99': round(q(ratios, .99), 4),
             'exact': sum(t['recorded']['counted_prompt'] == t['recorded']['prompt'] for t in trajectories)}
    comp = [t['recorded']['completion'] for t in trajectories]
    summary = {'round': a.round, 'code': str(code), 'population': dict(kinds), 'size': len(trajectories),
               'kinds': dict(collections.Counter(t['kind'] for t in trajectories)), 'prompt_check': check,
               'completion_mean': round(statistics.mean(comp)), 'completion_p50': q(comp, .5),
               'completion_p90': q(comp, .9), 'max_tokens': a.max_tokens, 'sampling': sampling, 'extra': overrides}
    # synth.bench.build_corpus returns a cached corpus when pattern, size and seed match: a tuning campaign points
    # [tuning] corpus at this file with corpus_size = size and corpus_seed = seed.
    (out / 'corpus.json').write_text(json.dumps({'pattern': str(out / 'corpus.json'), 'size': len(trajectories),
                                                 'seed': a.seed, 'summary': summary, 'trajectories': trajectories}))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
