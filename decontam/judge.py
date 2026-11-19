"""Step 3: judge (TB2.1 task, CalibForge task) pairs on the self-hosted DeepSeek-V4.1-Flash-1017 server.

  python -m decontam.judge --work W --out W/verdicts.jsonl [--pairs calibration | --pairs FILE] [--limit N]
  python -m decontam.judge ... --rubric decontam/rubric_reuse.md --labels REUSE,WEAK_REUSE,NO_REUSE --endpoint E
      --fingerprints F
      (the variant pass: is the candidate a variant of this test task? F holds each TB2.1 task's genre and distinctive
      choices, written before any candidate was seen; its block follows the TB2.1 bundle, inside the shared prefix)

Single-turn, everything in context: system = rubric.md, user = TB2.1 bundle -> overlap signals -> candidate bundle.
Requests go out grouped by TB2.1 task (the rubric + TB2.1 prefix is shared by its ~100+ candidates) with a short
concurrency ramp, so the first requests of a group fill the prefix cache before the rest arrive. Each pair gets
--samples independent samples (thinking on); a sample whose JSON does not parse is retried. Appends one line per
sample to --out and skips samples already there, so a rerun resumes.

--serving NAME[+NAME...] judges on task_synthesis DeepSeek servers instead of one endpoint.json (issue #119: the
production generation engines). Each test task's group is pinned to one (server, DP rank), the one with the fewest pairs
pinned so far, and sent there with X-data-parallel-rank (vLLM's own balancer would scatter the shared prefix over
cold ranks); servers that leave are dropped and their groups re-pinned when the serving files are re-read.
A connection failure (a server being released: its endpoints.json outlives vLLM by up to a minute) does not use up a
sample's retries: after 3 in a row the target is set aside for DOWN_SECONDS and its groups re-pinned at once, so no
sample ends without a verdict because a server left mid-pass (#234). --load-file writes the pinned load per server
(the production driver keeps the busiest server for decontam at the round's STOP).
"""
import argparse
import asyncio
import collections
import json
import os
from pathlib import Path
import re
import sys
import time

import aiohttp

from . import tasks as T
from tbshared import serving   # the registry reader (#445 item 3)

ENDPOINT = '/data/aai/tbenchtrain/decontam_calibforge/serve/endpoint.json'
# The serving registry root: $TASKSYNTH_SERVING_ROOT (tbshared.serving, every reader), else production. The old
# DECONTAM_SERVING_ROOT is a deprecated alias, still read for running campaigns; set both only to the same root.
LEGACY_ROOT_ENV = 'DECONTAM_SERVING_ROOT'
RUBRIC = Path(__file__).with_name('rubric.md')
LABELS = ('DUPLICATE', 'REWRITE', 'RELATED', 'UNRELATED')
CONTEXT = 132000
DOWN_SECONDS = 600        # a target with CONNECT_FAILS connection failures in a row is left out this long
CONNECT_FAILS = 3
CONNECT_DEADLINE = 6 * 3600   # a sample waits this long for a live target before it is written without a verdict
CONNECTION_ERRORS = (aiohttp.ClientConnectionError, asyncio.TimeoutError)
TAIL = '\n\nGive your verdict on this candidate as the JSON object.'
_warned: list = []   # the deprecation note, once per process


def serving_root() -> Path:
    """$TASKSYNTH_SERVING_ROOT, else the deprecated $DECONTAM_SERVING_ROOT, else the production registry; resolved on
    every call. Both set to different roots is an error (which one the run meant is unknowable)."""
    old, new = os.environ.get(LEGACY_ROOT_ENV), os.environ.get(serving.ROOT_ENV)
    if old and new and Path(old) != Path(new):
        raise SystemExit(f'{LEGACY_ROOT_ENV}={old} and {serving.ROOT_ENV}={new} disagree: set only {serving.ROOT_ENV}')
    if old and not new and not _warned:
        _warned.append(1)
        print(f'decontam.judge: {LEGACY_ROOT_ENV} is deprecated: set {serving.ROOT_ENV} (every reader honours it)',
              file=sys.stderr, flush=True)
    return serving.registry_root(new or old)


def parse(content):
    """The verdict object in the answer, or None."""
    if not content:
        return None
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', content.strip())
    decoder = json.JSONDecoder()
    candidates = [text] + [content[m.start():] for m in re.finditer(r'\{', content)]
    for candidate in candidates:
        try:
            obj = decoder.raw_decode(candidate)[0]
        except ValueError:
            continue
        if isinstance(obj, dict) and str(obj.get('label', '')).upper() in LABELS:
            obj['label'] = obj['label'].upper()
            obj.setdefault('copied_elements', [])
            obj.setdefault('reason', '')
            return obj
    return None


def load_bundles(work, needed, name='bundles.jsonl'):
    out = {}
    with open(Path(work) / name) as f:
        for line in f:
            row = json.loads(line)
            if (row['side'], row['name']) in needed:
                out[(row['side'], row['name'])] = row
    return out


def fingerprint_text(fp):
    """The fingerprint block of one TB2.1 task (decontam variants pass)."""
    lines = ['## Fingerprint of the test task', '',
             'Written beforehand by a careful reader of the test task who had not seen any candidate.', '',
             f'Genre (never counts): {fp["genre"]}', 'Conventions (never count):']
    lines += [f'- {c}' for c in fp.get('conventions', [])]
    lines += ['Distinctive choices:']
    lines += [f'- {c["id"]} [{c["weight"]}, {c["kind"]}] {c["choice"]} (evidence: {c["evidence"]})' for c in fp['choices']]
    lines += [f'Variant test: {fp["variant_test"]}', '', '']
    return '\n'.join(lines)


def select_pairs(work, which, limit=None, signal_files=('pair_signals.jsonl',)):
    """Pairs with their signal text: all retrieved, the calibration seed, or the (tb, cf) pairs in a JSONL file."""
    rows = list({(r['tb'], r['cf']): r for r in (json.loads(l) for name in signal_files
                                                  for l in open(Path(work) / name))}.values())
    if which == 'calibration':
        cal = T.read_calibration()
        want = set(cal['exclude']) | set(cal['keep']) | set(cal['expected_exclude_unconfirmed'])
        rows = [r for r in rows if (r['tb'], r['cf']) in want]
    elif which != 'all':
        want = {(r['tb'], r['cf']) for r in map(json.loads, open(which))}
        rows = [r for r in rows if (r['tb'], r['cf']) in want]
    rows.sort(key=lambda r: (r['tb'], r['cf']))
    if limit:  # a spread sample: every k-th pair, still grouped by TB2.1 task
        step = max(1, len(rows) // limit)
        rows = rows[::step][:limit]
    return rows


def dp_size(launch):
    """DP ranks behind the one URL of a dp_internal vLLM server (1 if external LB or unknown)."""
    try:
        argv = json.loads(Path(launch).read_text())['argv']
        if any(a in argv for a in ('--data-parallel-external-lb', '--data-parallel-hybrid-lb', '--data-parallel-rank')):
            return 1
        return max(1, int(argv[argv.index('--data-parallel-size') + 1]))
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return 1


def serving_targets(names, root=None, stale=120):
    """task_synthesis serving names joined by '+' -> (served model, [(url, DP rank or None)]) over the healthy ones."""
    root = root or serving_root()
    model, targets = None, []
    for n in names.split('+'):
        path = serving.path(n, root)
        info = serving.endpoint(n, root, stale_sec=stale)   # fresh and healthy, else None (tbshared.serving)
        if info is None:
            continue
        if model and info['served_name'] != model:
            raise SystemExit(f'{names}: members serve different models ({model}, {info["served_name"]})')
        model = info['served_name']
        dp = dp_size(path.parent / 'launch.json') if len(info['urls']) == 1 else 1
        targets += [(u.rstrip('/'), r if dp > 1 else None) for u in info['urls'] for r in range(dp)]
    return model, targets


class Judge:
    def __init__(self, url, model, effort, max_tokens, concurrency, ramp, rubric=RUBRIC, fingerprints=None,
                 serving=None):
        self.url, self.model, self.effort, self.max_tokens = url, model, effort, max_tokens
        self.serving = serving                       # task_synthesis serving names, or None for the one url
        self.targets = [(url, None)] if url else []  # (url, DP rank)
        self.pins, self.pinned = {}, collections.Counter()   # test task -> target; target -> pairs pinned
        self.failures = collections.Counter()
        self.refreshed = time.time()
        self.run_id = f'{os.uname().nodename}-{os.getpid()}'
        self.concurrency, self.ramp = concurrency, ramp
        self.rubric = Path(rubric).read_text()
        self.fingerprints = fingerprints or {}
        self.offset = 0  # first sample index (extra passes over judged pairs number their samples after the first two)
        self.stats = collections.Counter()
        self.started = time.time()
        self.down = {}          # target -> time it was set aside (connection failures)
        self.connect_fails = collections.Counter()   # target -> connection failures in a row
        self.load_file, self.load_written = None, 0.0

    def body(self, tb, signals, cf):
        fp = self.fingerprints.get(tb['name'], '')
        user = tb['text'] + '\n\n' + fp + signals + '\n' + cf['text'] + TAIL
        budget = CONTEXT - (tb['tokens'] + cf['tokens'] + len(fp) // 3 + 3000)
        return {'model': self.model, 'max_tokens': max(2048, min(self.max_tokens, budget)),
                'messages': [{'role': 'system', 'content': self.rubric}, {'role': 'user', 'content': user}],
                'reasoning_effort': self.effort,
                'chat_template_kwargs': {'thinking': True, 'reasoning_effort': self.effort}}

    def refresh(self):
        """Re-read the serving files: new servers join, gone ones (and targets set aside) leave and their test tasks
        are re-pinned."""
        self.refreshed = time.time()
        if not self.serving:
            return
        now = time.time()
        self.down = {t: since for t, since in self.down.items() if now - since < DOWN_SECONDS}
        model, targets = serving_targets(self.serving)
        targets = [t for t in targets if t not in self.down]
        if not targets:
            return
        self.targets = targets
        for tb, t in list(self.pins.items()):
            if t not in targets:
                del self.pins[tb]
        self.pinned = collections.Counter({t: n for t, n in self.pinned.items() if t in targets})
        self.write_load(force=True)

    def set_aside(self, target):
        """CONNECT_FAILS connection failures in a row: leave the target out for DOWN_SECONDS and re-pin its groups."""
        if self.serving and target in self.targets and target not in self.down:
            self.down[target] = time.time()
            self.stats['targets_set_aside'] += 1
            print(f'target {target} set aside after {CONNECT_FAILS} connection failures; its test tasks re-pin',
                  flush=True)
            self.refresh()
            if target in self.targets:   # it was the last target the serving files list: keep it, retry later
                self.targets = [t for t in self.targets if t != target] or self.targets
                for tb, t in list(self.pins.items()):
                    if t == target and len(self.targets) > 1:
                        del self.pins[tb]

    def pin(self, tb, weight=1):
        if tb not in self.pins:
            t = min(self.targets, key=lambda t: (self.pinned[t], t))
            self.pins[tb] = t
            self.pinned[t] += weight
            self.write_load()
        return self.pins[tb]

    def write_load(self, force=False):
        """--load-file: {serving, targets: [[url, rank, pinned weight]], updated_at}, at most every 30 s."""
        if not self.load_file or (not force and time.time() - self.load_written < 30):
            return
        self.load_written = time.time()
        tmp = Path(str(self.load_file) + '.partial')
        try:
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps({'serving': self.serving, 'run_id': self.run_id, 'updated_at': time.time(),
                                       'targets': [[u, r, self.pinned[(u, r)]] for u, r in self.targets]}))
            os.replace(tmp, self.load_file)
        except OSError:
            pass

    async def one(self, session, body, retries=3, tb=None):
        """One sample. API errors and unparseable answers use up `retries`; connection failures (a server leaving) do
        not: the target is set aside after CONNECT_FAILS of them in a row and the sample moves to another one."""
        last = None
        attempt, waited_since = 0, None
        while attempt < retries:
            t0 = time.time()
            if self.serving and (time.time() - self.refreshed > 300 or self.failures[self.pins.get(tb)] >= 3):
                self.failures[self.pins.get(tb)] = 0
                self.refresh()
            url, rank = self.pin(tb) if tb is not None else self.targets[0]
            headers = {'X-Decontam-Judge': self.run_id}
            if rank is not None:
                headers['X-data-parallel-rank'] = str(rank)
            try:
                async with session.post(url + '/chat/completions', json=body, headers=headers,
                                        timeout=aiohttp.ClientTimeout(total=3600)) as r:
                    data = await r.json(content_type=None)
            except CONNECTION_ERRORS as exc:
                last = {'error': repr(exc)}
                self.stats['http_errors'] += 1
                self.failures[(url, rank)] += 1
                self.connect_fails[(url, rank)] += 1
                if self.serving and self.connect_fails[(url, rank)] >= CONNECT_FAILS:
                    self.connect_fails[(url, rank)] = 0
                    self.set_aside((url, rank))
                waited_since = waited_since or time.time()
                if self.serving and time.time() - waited_since < CONNECT_DEADLINE:
                    await asyncio.sleep(5)
                    continue          # not this sample's fault: its retries are kept
                attempt += 1
                await asyncio.sleep(5)
                continue
            except (aiohttp.ClientError, ValueError) as exc:
                last = {'error': repr(exc)}
                self.stats['http_errors'] += 1
                self.failures[(url, rank)] += 1
                attempt += 1
                await asyncio.sleep(5)
                continue
            self.failures[(url, rank)] = 0
            self.connect_fails[(url, rank)] = 0
            waited_since = None
            attempt += 1
            if 'choices' not in data:
                last = {'error': str(data)[:2000]}
                self.stats['api_errors'] += 1
                await asyncio.sleep(5)
                continue
            choice, usage = data['choices'][0], data.get('usage') or {}
            msg = choice['message']
            reasoning = msg.get('reasoning') or msg.get('reasoning_content') or ''
            verdict = parse(msg.get('content'))
            self.stats['prompt_tokens'] += usage.get('prompt_tokens', 0)
            self.stats['cached_tokens'] += (usage.get('prompt_tokens_details') or {}).get('cached_tokens') or 0
            self.stats['completion_tokens'] += usage.get('completion_tokens', 0)
            last = {'verdict': verdict, 'content': msg.get('content'), 'reasoning': reasoning,
                    'finish_reason': choice.get('finish_reason'), 'usage': usage, 'attempts': attempt,
                    'seconds': round(time.time() - t0, 1)}
            if verdict:
                return last
            self.stats['parse_failures'] += 1
        return last

    async def run(self, pairs, bundles, samples, out):
        done = set()
        if Path(out).exists():
            for line in open(out):
                try:
                    r = json.loads(line)
                except ValueError:   # a line cut short by a killed run
                    continue
                if r.get('verdict'):
                    done.add((r['tb'], r['cf'], r['sample']))
        jobs = [(p, s) for p in pairs for s in range(self.offset, self.offset + samples) if (p['tb'], p['cf'], s) not in done]
        print(f'{len(jobs)} samples to judge ({len(done)} already done)', flush=True)
        if self.serving:   # pin the biggest groups first so the targets end up with similar loads
            for tb, n in collections.Counter(p['tb'] for p, _ in jobs).most_common():
                self.pin(tb, n)
            print(f'{len(self.targets)} targets; samples pinned per target: '
                  f'{sorted(self.pinned.values(), reverse=True)}', flush=True)
            self.write_load(force=True)
        sem = asyncio.Semaphore(0)
        released = 0

        async def ramp():
            nonlocal released
            while released < self.concurrency:
                n = min(self.ramp, self.concurrency - released)
                for _ in range(n):
                    sem.release()
                released += n
                await asyncio.sleep(2)

        conn = aiohttp.TCPConnector(limit=0)
        if Path(out).exists() and Path(out).stat().st_size:
            with open(out, 'rb') as f:
                f.seek(-1, 2)
                cut = f.read(1) != b'\n'
            if cut:   # end a line left unfinished by a killed run, so the next record starts on its own line
                with open(out, 'a') as f:
                    f.write('\n')
        with open(out, 'a') as sink:
            async with aiohttp.ClientSession(connector=conn, trust_env=False) as session:
                async def task(pair, sample):
                    async with sem:
                        tb, cf = bundles[('tb', pair['tb'])], bundles[('cf', pair['cf'])]
                        result = await self.one(session, self.body(tb, pair['text'], cf), tb=pair['tb'])
                        row = {'tb': pair['tb'], 'cf': pair['cf'], 'sample': sample, 'effort': self.effort,
                               'tb_truncated': tb['truncated'], 'cf_truncated': cf['truncated'],
                               'prompt_tokens_est': tb['tokens'] + cf['tokens'], **result}
                        sink.write(json.dumps(row) + '\n')
                        sink.flush()
                        self.stats['done'] += 1
                        self.stats[f'label_{(result.get("verdict") or {}).get("label", "NONE")}'] += 1

                async def report():
                    while True:
                        await asyncio.sleep(30)
                        dt = time.time() - self.started
                        s = self.stats
                        print(f'{dt / 60:6.1f} min  done {s["done"]}/{len(jobs)}  prompt {s["prompt_tokens"] / dt:,.0f} tok/s '
                              f'(cached {s["cached_tokens"] / max(1, s["prompt_tokens"]):.0%})  gen {s["completion_tokens"] / dt:,.0f} tok/s  '
                              f'{dict((k, v) for k, v in s.items() if k.startswith(("label_", "parse", "http", "api")))}',
                              flush=True)

                ramper, reporter = asyncio.create_task(ramp()), asyncio.create_task(report())
                await asyncio.gather(*(task(p, s) for p, s in jobs))
                ramper.cancel()
                reporter.cancel()
        self.write_load(force=True)
        dt = time.time() - self.started
        summary = {**self.stats, 'seconds': round(dt), 'prompt_tok_s': round(self.stats['prompt_tokens'] / dt),
                   'gen_tok_s': round(self.stats['completion_tokens'] / dt), 'samples': len(jobs)}
        print(json.dumps(summary), flush=True)
        return summary


def main():
    global LABELS
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pairs', default='all')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--samples', type=int, default=2)
    ap.add_argument('--effort', default='max')
    ap.add_argument('--max-tokens', type=int, default=32768)
    ap.add_argument('--concurrency', type=int, default=512)
    ap.add_argument('--ramp', type=int, default=32)
    ap.add_argument('--summary')
    ap.add_argument('--signals', default='pair_signals.jsonl', help='comma list of signal files under --work')
    ap.add_argument('--bundles', default='bundles.jsonl', help='bundle file under --work')
    ap.add_argument('--sample-offset', type=int, default=0)
    ap.add_argument('--rubric', default=str(RUBRIC), help='system prompt file')
    ap.add_argument('--labels', default=','.join(LABELS), help='comma list of the labels the rubric allows')
    ap.add_argument('--endpoint', default=ENDPOINT, help="the judge server's endpoint.json")
    ap.add_argument('--serving', help="task_synthesis serving names joined by '+' (instead of --endpoint)")
    ap.add_argument('--fingerprints', help='JSONL of TB2.1 fingerprints ({tb, genre, conventions, choices, variant_test})')
    ap.add_argument('--load-file', help='--serving: write the pinned load per target here (JSON, every 30 s)')
    a = ap.parse_args()
    LABELS = tuple(a.labels.split(','))
    if a.serving:
        t0 = time.time()
        while not (served := serving_targets(a.serving))[1]:
            if time.time() - t0 > 6 * 3600:
                raise SystemExit(f'{a.serving}: no healthy server after 6 h')
            time.sleep(60)
        endpoint = {'url': None, 'served_name': served[0], 'serving': a.serving, 'targets': served[1]}
    else:
        endpoint = json.loads(Path(a.endpoint).read_text())
    pairs = select_pairs(a.work, a.pairs, a.limit, a.signals.split(','))
    bundles = load_bundles(a.work, {('tb', p['tb']) for p in pairs} | {('cf', p['cf']) for p in pairs}, a.bundles)
    fps = {}
    if a.fingerprints:
        fps = {r['tb']: fingerprint_text(r) for r in map(json.loads, open(a.fingerprints))}
        missing = sorted({p['tb'] for p in pairs} - set(fps))
        if missing:
            raise SystemExit(f'no fingerprint for {missing}')
    judge = Judge(endpoint['url'], endpoint['served_name'], a.effort, a.max_tokens, a.concurrency, a.ramp, a.rubric, fps,
                  serving=a.serving)
    if a.serving:
        judge.targets = endpoint['targets']
    judge.load_file = a.load_file
    judge.offset = a.sample_offset
    summary = asyncio.run(judge.run(pairs, bundles, a.samples, a.out))
    if a.summary:
        Path(a.summary).write_text(json.dumps({**summary, 'pairs': len(pairs), 'args': vars(a),
                                               'endpoint': endpoint['url'] or a.serving, 'control': endpoint.get('control'),
                                               'targets': [list(t) for t in judge.targets]}, indent=1))


if __name__ == '__main__':
    main()
