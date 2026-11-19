"""m5-omniscience streaming pipeline: passage -> DeepSeek question -> checks -> Super trace (the eval's own harness).

Runs as one HTCondor CPU job (light HTTP loops; bulk writes go to /data from a non-g123 node).
- Generator: DeepSeek V4.1 Flash (servers in alloc.json ds.serving, default m5-ds) at high reasoning effort, no output cap, 6 h client timeout. Each
  worker thread is pinned to one DP rank (prefix cache for the shared prompt head).
- Sampling: subtopic by fitted weight (taxonomy/subtopics.json, via leaves.json), then the subtopic's leaf with the
  fewest calls so far, then that leaf's next unused passage (pool/passages.jsonl.zst). No open draws.
- Checks: gen/checks.py, plus online near-duplicate removal. Accepted problems stream straight into Super.
- Traces: AA-Omniscience's answer prompt (85% exact, 15% paraphrases) as system, the bare question as user, the
  M3.6 Super request settings exactly (aa_evals/omniscience run config of m36-super-vl-omniscience-full), one trace per
  problem, every ending kept.
- Caps: alloc.json re-read every 60 s (DeepSeek in flight; Super sessions per engine). Endpoints re-read; stale
  registries (>120 s) are waited out, never fatal.
- Stop: --n-problems accepted problems (pilot), or --token-target Super trace tokens (production: in-flight work is
  abandoned at the target, i.e. the tails are cut). Resumable: re-running with the same OUT continues.

Usage: python -m m5.omniscience.gen.pipeline OUT --passages P --leaves L [--n-problems N | --token-target T]
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import queue
import random
import re
import socket
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

import zstandard

from aa_evals.omniscience import client as aa_client
from m5.omniscience import fsio
from m5.omniscience.gen import checks, system_prompts

SERVING = Path('/data/aai/tbench/tasksynth/serving')
ALLOC = Path('/opt/aai/m5-coord/alloc.json')
V10 = Path('/data/aai/aa_evals/data/omniscience/public-v1@e4883ed/AA-Omniscience_dataset_public.csv')
PROMPT = (Path(__file__).parent / 'prompt.md').read_text()
CRITIC = (Path(__file__).parent / 'critic.md').read_text()
CRITIC_FLAGS = ('ambiguous', 'unstable', 'off_subtopic', 'not_short', 'garbled')
BENCH = 'omniscience'
# M3.6 Super generation settings (config.json of m36-super-vl-omniscience-full; = registry overrides + auto max_tokens)
SUPER_GEN = {'temperature': 1.0, 'top_p': 0.95,
             'chat_template_kwargs': {'enable_thinking': True, 'medium_effort': False, 'truncate_history_thinking': False},
             'max_tokens': 131072 - 2048}
SUPER_TIMEOUT = 3600
DS_BODY = {'temperature': 1.0, 'top_p': 0.95, 'reasoning_effort': 'high',
           'chat_template_kwargs': {'thinking': True, 'reasoning_effort': 'high'}}
DS_TIMEOUT = 6 * 3600
NOPROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def log(*a):
    print(time.strftime('%m-%d %H:%M:%S'), *a, flush=True)


def read_zst(path):
    with open(path, 'rb') as f:
        r = zstandard.ZstdDecompressor().stream_reader(f)
        buf = b''
        while True:
            b = r.read(1 << 24)
            if not b:
                break
            buf += b
            *lines, buf = buf.split(b'\n')
            for l in lines:
                if l.strip():
                    yield json.loads(l)
        if buf.strip():
            yield json.loads(buf)


class Alloc:
    def __init__(self, ds_name, super_cap_override=None, ds_cap_override=None):
        self.ds_name, self.t, self.data = ds_name, 0, {}
        self.super_override, self.ds_override = super_cap_override, ds_cap_override
        self.lock = threading.Lock()

    def _get(self):
        with self.lock:
            if time.time() - self.t > 60:
                try:
                    self.data = json.loads(ALLOC.read_text())
                except (OSError, ValueError):
                    pass
                self.t = time.time()
            return self.data

    def ds_cap(self):
        c = int(self._get().get('ds', {}).get('max_inflight', {}).get(BENCH, 0))
        return min(c, self.ds_override) if self.ds_override is not None else c

    def super_engines(self):
        return list(self._get().get('super', {}).get('serving', []))

    def super_cap(self):
        c = int(self._get().get('super', {}).get('max_inflight_per_engine', {}).get(BENCH, 0))
        return min(c, self.super_override) if self.super_override is not None else c


class Gate:
    """A semaphore whose size follows a callable (re-evaluated while waiting)."""

    def __init__(self, cap_fn):
        self.cap_fn, self.n, self.cv = cap_fn, 0, threading.Condition()

    def acquire(self, stop):
        with self.cv:
            while not stop.is_set() and self.n >= self.cap_fn():
                self.cv.wait(5)
            if stop.is_set():
                return False
            self.n += 1
            return True

    def release(self):
        with self.cv:
            self.n -= 1
            self.cv.notify()


class DS:
    """DeepSeek client over every server listed in alloc.json ds.serving (or one fixed server name). Each server has
    one URL per DP rank, or one URL + X-data-parallel-rank; requests are pinned to (server, rank) slots. Servers whose
    endpoints.json is older than 120 s are skipped; a request that fails is retried on the current server list, so
    servers can be added to or removed from ds.serving while the pipeline runs."""

    def __init__(self, name, alloc=None):
        self.name, self.alloc, self.t = name, alloc, 0
        self.slot_map, self.urls = [], []
        self.lock = threading.Lock()

    def _names(self):
        if self.alloc is not None:
            names = list(self.alloc._get().get('ds', {}).get('serving', []))
            if names:
                return names
        return [self.name]

    @staticmethod
    def _server(name):
        path = SERVING / name / 'endpoints.json'
        try:
            info = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        if time.time() - float(info.get('updated_at', 0)) > 120 or not info.get('urls'):
            return None
        urls, dp = [u.rstrip('/') for u in info['urls']], 1
        if len(urls) == 1:
            try:
                launch = json.loads((path.parent / 'launch.json').read_text())
                argv = launch.get('argv') or (launch.get('replicas') or [{}])[0].get('argv') or []
                if '--data-parallel-size' in argv and not any(
                        a in argv for a in ('--data-parallel-external-lb', '--data-parallel-hybrid-lb')):
                    dp = int(argv[argv.index('--data-parallel-size') + 1])
            except (OSError, ValueError, KeyError, IndexError):
                pass
        if len(urls) > 1:
            return [(u, {}, info['served_name']) for u in urls]
        return [(urls[0], ({'X-data-parallel-rank': str(r)} if dp > 1 else {}), info['served_name']) for r in range(dp)]

    def refresh(self, force=False):
        with self.lock:
            if not force and time.time() - self.t < 60 and self.slot_map:
                return True
            slot_map = []
            for n in self._names():
                slot_map += self._server(n) or []
            self.slot_map = slot_map
            self.urls = sorted({u for u, _, _ in slot_map})
            if slot_map:
                self.t = time.time()
            return bool(slot_map)

    def slots(self):
        return max(1, len(self.slot_map))

    def chat(self, messages, slot, stop, log_errors):
        """-> response dict. Retries forever on infra trouble (server restarts), until stop."""
        attempt = 0
        while not stop.is_set():
            if not self.refresh():
                time.sleep(min(300, 30 * (1 + attempt)))
                attempt += 1
                continue
            smap = self.slot_map
            if not smap:
                time.sleep(5)
                continue
            s = slot % len(smap)
            url, hdr, model = smap[s]
            body = dict(DS_BODY, model=model, messages=messages, **getattr(self, 'overrides', {}))
            req = urllib.request.Request(url + '/chat/completions', data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json', **hdr}, method='POST')
            t0 = time.time()
            try:
                with NOPROXY.open(req, timeout=DS_TIMEOUT) as r:
                    resp = json.loads(r.read())
                if not resp.get('choices'):
                    raise ValueError('no choices')
                resp['_latency_s'] = round(time.time() - t0, 1)
                resp['_slot'] = s
                return resp
            except urllib.error.HTTPError as e:
                detail = ''
                try:
                    detail = e.read().decode(errors='replace')[:300]
                except Exception:
                    pass
                log_errors.append(f'HTTP {e.code} {detail}')
                if e.code == 400 and 'maximum context' in detail:
                    return {'_error': 'context', 'detail': detail}
            except (urllib.error.URLError, ConnectionError, socket.timeout, TimeoutError, OSError, ValueError) as e:
                log_errors.append(f'{type(e).__name__}: {str(e)[:200]}')
            attempt += 1
            self.refresh(force=True)
            time.sleep(min(300, 2 ** min(attempt, 8)) * (0.5 + random.random()))
        return None


def passage_block(p):
    head = []
    if p.get('title'):
        head.append(f"Title: {p['title']}")
    if p.get('header'):
        head.append(f"Document header: {p['header']}")
    src = {'wiki': 'Wikipedia (2018)', 'web': 'educational web page', 'law': 'US court opinion',
           'sw': 'software documentation / changelog', 'pubmed': 'PubMed abstract', 'edgar': 'SEC 10-K filing'}
    head.append(f"Source type: {src.get(p['source'], p['source'])}")
    out = '\n'.join(head) + '\n\n' + p['text']
    if p.get('avoid'):
        out += ('\n\n[Already asked from this passage; ask about a clearly different detail, or skip: '
                + p['avoid'] + ']')
    return out


class Sampler:
    def __init__(self, leaves, passages, used, rng, quota_total, source_w=None):
        self.rng = rng
        self.source_w = source_w or {}
        self.leaves = {L['leaf_id']: L for L in leaves}
        self.q = collections.defaultdict(list)          # leaf_id -> unused passages
        self.used_pids = used
        for p in passages:
            if p['pid'] not in used:
                self.q[p['leaf_id']].append(p)
        self.calls = collections.Counter()
        self.sub_leaves = collections.defaultdict(list)
        self.sub_w = {}
        for L in leaves:
            key = (L['domain'], L['subtopic'])
            self.sub_leaves[key].append(L['leaf_id'])
            self.sub_w[key] = L['subtopic_weight']
        for pid in used:
            self.calls[pid.split(':', 1)[0]] += 1
        self.reuse = collections.defaultdict(list)       # leaf_id -> passages for a second question (fallback)
        self.acc = collections.Counter()                 # (domain, subtopic) -> accepted problems
        self.sub_calls = collections.Counter()           # (domain, subtopic) -> generator calls
        for pid in used:
            L = self.leaves.get(pid.split(':', 1)[0])
            if L:
                self.sub_calls[(L['domain'], L['subtopic'])] += 1
        self.lock = threading.Lock()

    def note_accept(self, p):
        with self.lock:
            self.acc[(p['domain'], p['subtopic'])] += 1

    def _w(self, k):
        """Call weight: target share x (target / accepted share), so acceptance-rate differences between subtopics
        (case law ~64%, software docs ~41%) do not skew the accepted mix. Smoothed and clamped to [1/3, 3]."""
        n = sum(self.acc.values())
        tot = sum(self.sub_w.values())
        want = self.sub_w[k] / tot
        if n < 500:
            return want
        rate = (self.acc[k] + 2) / (self.sub_calls[k] + 4)          # this subtopic's acceptance rate
        if self.sub_calls[k] >= 50 and rate < 0.12:
            # the pool barely covers this subtopic (e.g. no JUnit docs): extra calls would mostly be skips, so it
            # keeps its plain share instead of being chased with DeepSeek calls (noisy-but-usable pools, ~20-25%
            # acceptance as in several Finance subtopics, are still chased)
            return want
        r = (want * n + 5) / (self.acc[k] + 5)                       # remaining deficit (>1) or surplus (<1)
        return want / max(0.25, rate) * min(2.0, max(0.5, r))

    def offer_reuse(self, p, question):
        """An accepted first-pass passage may serve once more (a different detail) after its leaf runs dry."""
        if p.get('avoid') or p['pid'] + ':r' in self.used_pids:
            return
        with self.lock:
            self.reuse[p['leaf_id']].append(dict(p, pid=p['pid'] + ':r', avoid=question))

    def next(self):
        with self.lock:
            keys = [k for k in self.sub_w if any(self.q[l] or self.reuse[l] for l in self.sub_leaves[k])]
            if not keys:
                return None
            k = self.rng.choices(keys, weights=[self._w(k) for k in keys])[0]
            self.sub_calls[k] += 1
            ls = [l for l in self.sub_leaves[k] if self.q[l]]
            fresh = bool(ls)
            if not fresh:
                ls = [l for l in self.sub_leaves[k] if self.reuse[l]]
            m = min(self.calls[l] for l in ls)
            lid = self.rng.choice([l for l in ls if self.calls[l] == m])
            self.calls[lid] += 1
            if not fresh:
                return self.reuse[lid].pop(0)
            q = self.q[lid]
            if not self.source_w:
                return q.pop(0)
            # among the next few by retrieval rank, prefer sources the models know less well from pretraining
            head = q[:8]
            w = [self.source_w.get((p['domain'], p['source']), self.source_w.get(p['source'], 1.0)) * 0.85 ** r
                 for r, p in enumerate(head)]
            return q.pop(self.rng.choices(range(len(head)), weights=w)[0])


def parse_source_weights(spec):
    """'wiki=0.2,web=0.6,Finance:web=0.2' -> {'wiki': 0.2, 'web': 0.6, ('Finance', 'web'): 0.2}"""
    out = {}
    for x in filter(None, spec.split(',')):
        k, v = x.split('=')
        out[tuple(k.split(':', 1)) if ':' in k else k] = float(v)
    return out


class Pipeline:
    def __init__(self, args):
        self.args, self.out = args, Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.stop, self.gen_stop = threading.Event(), threading.Event()
        self.trace_stop = threading.Event()   # token target reached: start no new traces, let in-flight ones finish
        self.alloc = Alloc(args.ds, args.super_cap, args.ds_cap)
        self.ds = DS(args.ds, self.alloc)
        # critic at low effort: same recall vs the Opus audit (25 vs 26 of 35), less than half the tokens (1.06k vs 2.31k)
        self.ds_critic = DS(args.ds, self.alloc)
        self.ds_critic.overrides = {'reasoning_effort': 'low', 'chat_template_kwargs': {'thinking': True, 'reasoning_effort': 'low'}}
        self.rng = random.Random(args.seed)
        self.wlock = threading.Lock()
        self.super_q = queue.Queue()
        self.stats = collections.Counter()
        self.dedup = checks.DedupIndex()
        self.examples = list(csv.DictReader(open(V10, encoding='utf-8')))
        self.trace_tokens = 0
        self.engine_load = collections.Counter()
        self.engine_eps = {}
        leaves = json.loads(Path(args.leaves).read_text())['leaves']
        used, accepted, traced = set(), {}, set()
        if (self.out / 'gen.jsonl').exists():
            for line in open(self.out / 'gen.jsonl'):
                r = json.loads(line)
                used.add(r['pid'])
                if r.get('accepted'):
                    accepted[r['problem_id']] = r
                    self.dedup.add(r['problem_id'], r['question'], r.get('leaf_id'))
        if (self.out / 'traces.jsonl').exists():
            for line in open(self.out / 'traces.jsonl'):
                r = json.loads(line)
                traced.add(r['problem_id'])
                self.trace_tokens += r.get('total_tokens') or 0
        self.n_accepted = len(accepted)
        for pid, r in accepted.items():
            if pid not in traced and args.super_threads:
                self.super_q.put(r)
        self.n_traced = len(traced)
        log(f'resume: {len(used)} gen calls, {len(accepted)} accepted, {len(traced)} traced, '
            f'{self.trace_tokens} trace tokens')
        passages = list(read_zst(args.passages))
        self.sampler = Sampler(leaves, passages, used, self.rng, args.n_problems, parse_source_weights(args.source_weights))
        by_pid = {p['pid']: p for p in passages}
        for pid, r in accepted.items():
            self.sampler.note_accept(r)
            if r['pid'] in by_pid:
                self.sampler.offer_reuse(by_pid[r['pid']], r['question'])
        self.ds_gate = Gate(lambda: self.alloc.ds_cap())

    def write(self, name, row):
        with self.wlock:
            fsio.append_line(self.out / name, json.dumps(row, ensure_ascii=False))

    def examples_for(self, p, rng):
        pool = [e for e in self.examples if e['subtopic'] != p['subtopic']]
        same = [e for e in pool if e['domain'] == p['domain']]
        other = [e for e in pool if e['domain'] != p['domain']]
        ex = rng.sample(same, 2) + rng.sample(other, 4)
        rng.shuffle(ex)
        return '\n'.join(f"- [{e['domain']} / {e['subtopic']}] Q: {e['question']}\n  A: {e['answer']}" for e in ex), \
            [e['question_id'] for e in ex]

    # ---------------- generator ----------------
    def gen_worker(self, i):
        rng = random.Random(f'{self.args.seed}-{i}')
        while not self.gen_stop.is_set():
            if self.args.n_problems and self.n_accepted >= self.args.n_problems:
                self.gen_stop.set()
                break
            if not self.ds_gate.acquire(self.gen_stop):
                break
            try:
                p = self.sampler.next()
                if p is None:
                    log('passages exhausted')
                    self.gen_stop.set()
                    break
                self.gen_one(p, i, rng)
            except Exception:
                log('gen worker error', traceback.format_exc())
                time.sleep(10)
            finally:
                self.ds_gate.release()

    def gen_one(self, p, slot, rng):
        ex_text, ex_ids = self.examples_for(p, rng)
        prompt = PROMPT.format(domain=p['domain'], subtopic=p['subtopic'], leaf=p['leaf'], examples=ex_text,
                               passage=passage_block(p))
        errs = []
        t0 = time.time()
        resp = self.ds.chat([{'role': 'user', 'content': prompt}], slot, self.gen_stop, errs)
        if resp is None:
            return  # stopping; the passage stays unused for a resume
        row = {'pid': p['pid'], 'leaf_id': p['leaf_id'], 'domain': p['domain'], 'subtopic': p['subtopic'],
               'leaf': p['leaf'], 'source': p['source'], 'doc_id': p['doc_id'], 'examples': ex_ids,
               'errors': errs[-5:], 'latency_s': round(time.time() - t0, 1)}
        if resp.get('_error'):
            row.update(status='error', reason=resp['_error'])
            self.write('gen.jsonl', row)
            self.stats['gen_error'] += 1
            return
        msg = resp['choices'][0]['message']
        content = msg.get('content') or ''
        u = resp.get('usage') or {}
        row.update(usage={'prompt': u.get('prompt_tokens'), 'completion': u.get('completion_tokens')},
                   finish=resp['choices'][0].get('finish_reason'), reasoning_chars=len(msg.get('reasoning_content') or
                                                                                      msg.get('reasoning') or ''),
                   slot=resp.get('_slot'))
        self.stats['ds_prompt_tokens'] += u.get('prompt_tokens') or 0
        self.stats['ds_completion_tokens'] += u.get('completion_tokens') or 0
        d = checks.parse(content)
        if d is None:
            row.update(status='unparsed', raw=content[-2000:])
            self.stats['unparsed'] += 1
        elif d.get('status') != 'ok':
            row.update(status='skip', reason=str(d.get('reason', ''))[:500])
            self.stats['skip'] += 1
        else:
            bad = checks.check(d, p.get('header', '') + '\n' + p['text'])
            row.update(question=d.get('question', '').strip(), answer_check_only=str(d.get('answer', '')).strip(),
                       evidence=str(d.get('evidence', ''))[:2000], answer_type=d.get('answer_type'))
            if not bad and not self.args.no_critic:
                with self.wlock:
                    dup = self.dedup.find(row['question'], row['leaf_id'])
                if not dup:
                    bad = self.critic(row, p, slot, errs)
                    if bad is None:
                        return  # stopping; the passage stays unused for a resume
            with self.wlock:
                dup = None if bad else self.dedup.find(row['question'], row['leaf_id'])
                if dup:
                    bad = ['near_duplicate']
                    row['dup_of'] = dup
                ok = not bad and not (self.args.n_problems and self.n_accepted >= self.args.n_problems)
                if ok:
                    self.n_accepted += 1
                    pid = 'om-' + hashlib.sha1(p['pid'].encode()).hexdigest()[:12]
                    row.update(problem_id=pid, accepted=True)
                    self.dedup.add(pid, row['question'], row['leaf_id'])
            row['status'] = 'ok' if row.get('accepted') else ('rejected' if bad else 'over_quota')
            row['reject'] = bad
            self.stats[row['status']] += 1
            for b in bad:
                self.stats['rej_' + b] += 1
        self.write('gen.jsonl', row)
        if row.get('accepted'):
            self.sampler.offer_reuse(p, row['question'])
            self.sampler.note_accept(p)
        if row.get('accepted') and self.args.super_threads:
            self.super_q.put(row)

    def critic(self, row, p, slot, errs):
        """Problem-side critic on DeepSeek (self-containment, subtopic fit, short answer, well-formed; never difficulty).
        -> list of reject reasons, or None when stopping."""
        prompt = CRITIC.format(domain=p['domain'], subtopic=p['subtopic'], question=row['question'],
                               answer=row['answer_check_only'])
        resp = self.ds_critic.chat([{'role': 'user', 'content': prompt}], slot, self.gen_stop, errs)
        if resp is None:
            return None
        if resp.get('_error'):
            row['critic'] = {'error': resp['_error']}
            return ['critic_error']
        u = resp.get('usage') or {}
        self.stats['ds_prompt_tokens'] += u.get('prompt_tokens') or 0
        self.stats['ds_completion_tokens'] += u.get('completion_tokens') or 0
        row['usage_critic'] = {'prompt': u.get('prompt_tokens'), 'completion': u.get('completion_tokens')}
        d = checks.parse(resp['choices'][0]['message'].get('content') or '')
        if d is None or not isinstance(d.get('flags'), list):
            row['critic'] = {'unparsed': True}
            return ['critic_unparsed']
        flags = [f for f in d['flags'] if f in CRITIC_FLAGS]
        row['critic'] = {'flags': flags, 'referent': str(d.get('referent', ''))[:200], 'note': str(d.get('note', ''))[:200]}
        return ['critic_' + f for f in flags]

    # ---------------- traces ----------------
    def engine(self, name):
        ep = self.engine_eps.get(name)
        try:
            ep = aa_client.resolve(f'serving:{name}')
            self.engine_eps[name] = ep
        except aa_client.InfraError:
            return None
        return ep

    def pick_engine(self):
        """-> engine name with a free session slot (fewest live sessions), else None."""
        cap = self.alloc.super_cap()
        with self.wlock:
            names = [n for n in self.alloc.super_engines() if self.engine_load[n] < cap]
            if not names:
                return None
            n = min(names, key=lambda n: (self.engine_load[n], random.random()))
            self.engine_load[n] += 1
            return n

    def super_worker(self, j):
        rng = random.Random(f'super-{self.args.seed}-{j}')
        while not self.stop.is_set():
            if self.trace_stop.is_set():
                return
            try:
                prob = self.super_q.get(timeout=10)
            except queue.Empty:
                if self.gen_stop.is_set() and self.super_q.empty() and self.inflight_gen() == 0:
                    return
                continue
            name = None
            while not self.stop.is_set():
                name = self.pick_engine()
                if name:
                    break
                time.sleep(2)
            if not name:
                self.super_q.put(prob)
                return
            try:
                self.trace_one(prob, name, rng)
            except Exception:
                log('super worker error', traceback.format_exc())
                self.super_q.put(prob)
                time.sleep(30)
            finally:
                with self.wlock:
                    self.engine_load[name] -= 1

    def inflight_gen(self):
        return self.ds_gate.n

    def trace_one(self, prob, name, rng):
        pr = random.Random(prob['problem_id'])
        prompt_id, system = system_prompts.pick(prob['domain'], prob['subtopic'], pr)
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': prob['question']}]
        errors = []
        t0 = time.time()
        while not self.stop.is_set():
            ep = self.engine(name)
            if ep is None:
                time.sleep(60)
                continue
            try:
                resp = aa_client.chat(ep, dict(SUPER_GEN, messages=messages), timeout=SUPER_TIMEOUT, log=errors,
                                      timeout_is_model=True)
                out = aa_client.split_response(resp)
                break
            except aa_client.ModelTimeout as e:
                out = {'answer': '', 'reasoning': '', 'finish_reason': 'timeout', 'usage': {}}
                break
            except aa_client.InfraError:
                time.sleep(60)
        else:
            self.super_q.put(prob)
            return
        u = out.get('usage') or {}
        row = {'problem_id': prob['problem_id'], 'pid': prob['pid'], 'domain': prob['domain'],
               'subtopic': prob['subtopic'], 'leaf': prob['leaf'], 'prompt_id': prompt_id, 'messages': messages,
               'answer': out['answer'], 'reasoning': out['reasoning'], 'finish_reason': out['finish_reason'],
               'usage': u, 'total_tokens': u.get('total_tokens'), 'engine': name,
               'latency_s': round(time.time() - t0, 1), 'errors': errors[-5:], 'gen_settings': SUPER_GEN}
        with self.wlock:
            self.trace_tokens += u.get('total_tokens') or 0
            self.n_traced += 1
        self.write('traces.jsonl', row)
        if self.args.token_target and self.trace_tokens >= self.args.token_target and not self.trace_stop.is_set():
            # no new traces; in-flight ones finish so the kept set is not biased toward short traces
            log('token target reached', self.trace_tokens, '- draining in-flight traces')
            self.gen_stop.set()
            self.trace_stop.set()

    def metrics_worker(self):
        """Every 5 min: the servers' vLLM counters (tokens served, running requests, prefix-cache hits), for the
        GPU-hour share (my tokens / all tokens served x engine GPUs x wall) and the KV / prefix-hit report."""
        keys = ('vllm:generation_tokens_total', 'vllm:prompt_tokens_total', 'vllm:num_requests_running',
                'vllm:num_requests_waiting', 'vllm:prefix_cache_hits_total', 'vllm:prefix_cache_queries_total',
                'vllm:kv_cache_usage_perc', 'vllm:gpu_cache_usage_perc')
        while not self.stop.is_set():
            rows = []
            targets = [('ds', u) for u in (self.ds.urls or [])]
            for n in self.alloc.super_engines():
                ep = self.engine_eps.get(n)
                if ep:
                    targets += [(n, u) for u in ep.urls]
            for name, u in targets:
                try:
                    with NOPROXY.open(u.rsplit('/v1', 1)[0] + '/metrics', timeout=20) as r:
                        txt = r.read().decode(errors='replace')
                    vals = collections.Counter()
                    for line in txt.splitlines():
                        if line.startswith('#'):
                            continue
                        for k in keys:
                            if line.startswith(k + '{') or line.startswith(k + ' '):
                                try:
                                    vals[k] += float(line.rsplit(' ', 1)[1])
                                except ValueError:
                                    pass
                    rows.append({'server': name, 'url': u, **vals})
                except Exception as e:
                    rows.append({'server': name, 'url': u, 'error': str(e)[:100]})
            self.write('metrics.jsonl', {'t': time.time(), 'servers': rows,
                                         'mine': {'trace_tokens': self.trace_tokens, 'traced': self.n_traced,
                                                  'accepted': self.n_accepted, **self.stats}})
            self.stop.wait(300)

    def run(self):
        threading.Thread(target=self.metrics_worker, daemon=True).start()
        threads = [threading.Thread(target=self.gen_worker, args=(i,), daemon=True) for i in range(self.args.ds_threads)]
        threads += [threading.Thread(target=self.super_worker, args=(j,), daemon=True)
                    for j in range(self.args.super_threads)]
        for t in threads:
            t.start()
        last = 0
        while any(t.is_alive() for t in threads):
            time.sleep(5)
            if self.args.token_target and self.args.gen_headroom:
                # stop generating once accepted problems cover the target with headroom (1.4x oversupply)
                per = self.trace_tokens / max(1, self.n_traced)
                if self.n_traced > 200 and self.n_accepted * per >= self.args.gen_headroom * self.args.token_target:
                    if not self.gen_stop.is_set():
                        log('generation headroom reached', self.n_accepted, per)
                    self.gen_stop.set()
            if time.time() - last > 300:
                last = time.time()
                st = {'accepted': self.n_accepted, 'traced': self.n_traced, 'trace_tokens': self.trace_tokens,
                      'super_q': self.super_q.qsize(), 'ds_inflight': self.ds_gate.n, 'ds_cap': self.alloc.ds_cap(),
                      'super_load': dict(self.engine_load), **self.stats}
                log(json.dumps(st))
                fsio.write_atomic(self.out / 'progress.json', json.dumps({**st, 't': time.time()}))
            if self.gen_stop.is_set() and self.super_q.empty() and self.ds_gate.n == 0 and \
                    sum(self.engine_load.values()) == 0:
                break
        log('done', self.n_accepted, self.n_traced, self.trace_tokens, dict(self.stats))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--passages', required=True)
    ap.add_argument('--leaves', required=True)
    ap.add_argument('--ds', default='m5-ds')
    ap.add_argument('--n-problems', type=int, default=0)
    ap.add_argument('--token-target', type=int, default=0)
    ap.add_argument('--gen-headroom', type=float, default=1.4)
    ap.add_argument('--ds-threads', type=int, default=200)
    ap.add_argument('--super-threads', type=int, default=200)
    ap.add_argument('--ds-cap', type=int, default=None)
    ap.add_argument('--super-cap', type=int, default=None)
    ap.add_argument('--no-super', action='store_true')
    ap.add_argument('--no-critic', action='store_true')
    ap.add_argument('--source-weights', default='wiki=0.2,web=0.6', help='passage choice weights per source, optionally per domain, e.g. wiki=0.35,Finance:web=0.2')
    ap.add_argument('--seed', type=int, default=354)
    args = ap.parse_args()
    if args.no_super:
        args.super_threads = 0
    Pipeline(args).run()


if __name__ == '__main__':
    main()
