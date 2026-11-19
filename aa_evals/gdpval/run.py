"""GDPval-AA (#183): AA's agentic file-deliverable eval on the public GDPval gold set, pairwise-judged into Elo.

  # one command for a model (submits condor CPU workers; the model is served elsewhere)
  python -m aa_evals.gdpval.submit --run-id q38m-r1 -- --model serving:aa-gdpval-qwen --reasoning-effort medium

  python -m aa_evals.gdpval.run --run-id ID --stage init  --model SPEC [--reasoning-effort E] [--limit N] [--repeats K]
  python -m aa_evals.gdpval.run --run-id ID --stage work  [--concurrency C]          # a worker (condor job)
  python -m aa_evals.gdpval.run --run-id ID --stage judge --judge-arm ARM [--judge-endpoint SPEC] [--follow]
  python -m aa_evals.gdpval.run --run-id ID --stage score [--judge-arm ARM]
  python -m aa_evals.gdpval.run --run-id ID --status
  python -m aa_evals.gdpval.run --stage prepare                                        # refs + expert renders, once

Runs live in $AA_RUNS_ROOT/gdpval/<run_id>/ (default /data/aai/aa_evals/runs): config.json, attempts/<key>/
(outcome.json, transcript.json.gz, deliverables/, render/), claims/, matches/<arm>.jsonl, items.jsonl, summary.json.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from aa_evals.common import jsonl, protocol
from aa_evals.gdpval import elo
from aa_evals.gdpval import tasks as T
from aa_evals.gdpval.network import POLICY as NETWORK_POLICY, require_offline

RUNS = Path(os.environ.get('AA_RUNS_ROOT', '/data/aai/aa_evals/runs')) / 'gdpval'
SIF = Path(os.environ.get('AA_GDPVAL_SIF', '/data/aai/aa_evals/sif/gdpval-v1.sif'))
PREP = Path('/data/aai/aa_evals/data/gdpval/render_v1')
LEADERBOARD = Path('/data/aai/tbench/index_audit/aa_leaderboard_2026-11-02.json')
AA_KEY = 'gdpval-aa-normalized'
MAX_INFRA_TRIES = 4
STALE_CLAIM_SEC = 1800
REPEATS = 5   # M3 protocol (AA: 1)
DEFAULT_JUDGE_ARM = 'qwen38-27b-medium-pin'   # the pinned judge (M3.0): the summary headline when several arms judged


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def key(task_id, repeat):
    return f'{task_id}__r{repeat}'


def git_sha():
    here = Path(__file__).resolve().parent
    sha_file = here.parents[1] / 'GIT_SHA'
    if sha_file.exists():
        return sha_file.read_text().strip()
    r = subprocess.run(['git', '-C', str(here), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    d = subprocess.run(['git', '-C', str(here), 'status', '--porcelain', '.'], capture_output=True, text=True)
    return r.stdout.strip() + ('-dirty' if d.stdout.strip() else '')


BAD_LINES = jsonl.Tolerant()   # undecodable lines seen in this run's files: counted in summary.json, which is then invalid


def read_jsonl(p):
    """'\n'-separated (aa_evals.common.jsonl); an unterminated last line is a row a writer is still appending; any
    other bad line is recorded in BAD_LINES."""
    return BAD_LINES.read(p)


# ---------------------------------------------------------------------------------------------------------- config
def endpoint_ctx(spec):
    """max_model_len of the endpoint (from /v1/models), else None."""
    import httpx
    from aa_evals.gdpval.endpoint import resolve
    try:
        ep = resolve(spec)
        r = httpx.get(f'{ep["urls"][0]}/models', timeout=30)
        for m in r.json().get('data', []):
            if m.get('id') == ep['served_name']:
                return m.get('max_model_len')
    except Exception:  # noqa: BLE001
        return None
    return None


def build_config(a, run_dir, tasks):
    from aa_evals.gdpval.endpoint import resolve
    from aa_evals.gdpval.client import history_adapters
    ep = resolve(a.model)
    served, request = ep['served_name'], ep['request']
    sif = Path(a.sif) if a.sif else SIF
    body = copy.deepcopy(request.get('overrides') or {})
    if not body:
        body['temperature'] = 0.6   # AA's default for reasoning models when the lab gives none
    if a.reasoning_effort:
        try:
            body = protocol.apply_effort(body, a.reasoning_effort)
        except ValueError as e:
            sys.exit(f'--reasoning-effort {a.reasoning_effort}: {e}')
    if a.extra_body:
        body.update(json.loads(a.extra_body))
    for k in ('max_tokens', 'max_completion_tokens'):
        body.pop(k, None)
    ctx_ep = endpoint_ctx(a.model)
    win, why = protocol.window(ctx_ep, a.context_window)
    if not win:
        sys.exit('could not read the endpoint context window; pass --context-window')
    parts = [x.strip().rpartition('*')[0] if '*' in x else x.strip() for x in a.model.split(',') if x.strip()]
    seen = {p: protocol.image_inputs(p) for p in parts}   # a pooled spec: every endpoint must agree
    if len({v for v, _ in seen.values()}) > 1:
        sys.exit(f'the pooled endpoints disagree on image inputs: {seen}')
    vision, vwhy = next(iter(seen.values()))
    if a.vision is not None:
        vision, vwhy = a.vision == 'on', f'--vision {a.vision}'
    if vision is None:
        sys.exit(f'image inputs unknown ({vwhy}); pass --vision on|off')
    max_output = a.max_output or win
    aa_name, aa_slug = protocol.aa_model(served, body)
    aa_slug = a.aa_slug or aa_slug
    cfg = {
        'benchmark': 'GDPval-AA', 'version': 'AA v2.1 method, our open reproduction', 'run_id': a.run_id,
        'git_sha': git_sha(), 'started_at': now(), 'argv': sys.argv, 'agent_network': dict(NETWORK_POLICY),
        'model': {'spec': a.model, 'served_name': served, 'body': body, 'effort': protocol.effort_of(body),
                  'vision': bool(vision), 'vision_rule': vwhy, 'max_output': max_output, 'aa_name': aa_name,
                  'adapters': history_adapters(request),
                  'aa_slug': aa_slug},
        'context': {'window': win, 'rule': why, 'endpoint_max_model_len': ctx_ep, 'requested': a.context_window,
                    'summarize_at': 0.7},
        'harness': {'name': 'Stirrup', 'version': '0.2.1', 'commit': 'c092cee80ce09cd1dfb93aef053296a9ecd9dbb6',
                    'max_turns': 250, 'tools': ['code_exec'] + (['view_image'] if vision else []) + ['finish',
                                                                                         'abandon_task_finish'],
                    'prompts': 'AA v2.1 (aa_prompts.py) plus explicit offline environment notice', 'shell_timeout_sec': 600},
        'sandbox': {'sif': str(sif), 'sif_sha256': (sif.with_suffix('.sha256').read_text().split()[0]
                                                  if sif.with_suffix('.sha256').exists() else None),
                    'kind': 'apptainer exec --containall per command; /home/user and /tmp bound per attempt'},
        'limits': {'wall_sec': a.wall_sec, 'max_infra_tries': MAX_INFRA_TRIES},
        'dataset': {'id': 'openai/gdpval', 'revision': T.REVISION, 'parquet_sha256': T.parquet_sha(),
                    'n_tasks_total': T.N_TASKS},
        'repeats': a.repeats, 'selection': {'limit': a.limit, 'task_ids': [t.task_id for t in tasks],
                                            # recorded only for a non-'all' set, so pre-#346 configs still match
                                            **({'task_set': a.task_set} if a.task_set != 'all' else {})},
        'anchor': {'player': elo.EXPERT, 'elo': elo.EXPERT_ELO,
                   'why': "AA v2 human baseline 1000, moved onto the v2.1 scale (-32; #183)"},
    }
    from aa_evals.gdpval.runtime_cache import freeze
    cfg['sandbox']['runtime_packages'] = freeze(getattr(a, 'runtime_cache_manifest', None), tasks,
                                               cfg['sandbox']['sif_sha256'])
    old = run_dir / 'config.json'
    if old.exists():
        prev = json.loads(old.read_text())
        if prev.get('agent_network') != cfg['agent_network']:
            sys.exit(f'{old}: agent network protocol changed; use a new --run-id (not --force)')
        if prev.get('sandbox', {}).get('runtime_packages') != cfg['sandbox']['runtime_packages']:
            sys.exit(f'{old}: runtime package protocol changed; use a new --run-id (not --force)')
        for k in ('model', 'repeats', 'selection', 'context'):
            if prev.get(k) != cfg[k] and not a.force:
                hint = (' (runs initialised before #346 used all 220 tasks: pass --task-set all)'
                        if k == 'selection' and 'task_set' not in prev.get(k, {}) else '')
                sys.exit(f'{old}: {k} changed; use a new --run-id (or --force){hint}')
        cfg['started_at'] = prev.get('started_at', cfg['started_at'])
    return cfg


def select(a):
    tasks = T.task_set(T.stratified(T.load_tasks()), a.task_set)
    if a.task_ids:
        want = a.task_ids.split(',')
        if a.task_set != 'all' and (out := [w for w in want if w in T.TEXT_ONLY_EXCLUDED]):
            sys.exit(f'--task-ids {out} are excluded by --task-set {a.task_set} (#346); pass --task-set all')
        tasks = [t for t in tasks if t.task_id in want]
    elif a.limit:
        tasks = tasks[:a.limit]
    return tasks


def stage_init(a):
    run_dir = RUNS / a.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    tasks = select(a)
    cfg = build_config(a, run_dir, tasks)
    (run_dir / 'config.json').write_text(json.dumps(cfg, indent=1, default=str) + '\n')
    for d in ('attempts', 'claims', 'matches', 'jobs'):
        (run_dir / d).mkdir(exist_ok=True)
    print(json.dumps({k: cfg[k] for k in ('model', 'context', 'repeats')}, indent=1, default=str))
    return cfg


# ---------------------------------------------------------------------------------------------------------- worker
class Claims:
    """Lustre-safe claims (mkdir is atomic; no file locks on Lustre)."""

    def __init__(self, run_dir):
        self.dir = run_dir / 'claims'
        self.dir.mkdir(exist_ok=True)

    def take(self, k):
        try:
            (self.dir / k).mkdir()
        except FileExistsError:
            return False
        (self.dir / k / 'hb').write_text(str(time.time()))
        return True

    def beat(self, k):
        try:
            (self.dir / k / 'hb').write_text(str(time.time()))
        except OSError:
            pass

    def release(self, k):
        shutil.rmtree(self.dir / k, ignore_errors=True)

    def stale(self, k):
        try:
            return time.time() - float((self.dir / k / 'hb').read_text()) > STALE_CLAIM_SEC
        except (OSError, ValueError):
            return False


def attempt_state(run_dir, k):
    """'done' (ok, or infra out of tries), 'retry' (infra, tries left) or 'todo'."""
    d = run_dir / 'attempts' / k
    o = d / 'outcome.json'
    if not o.exists():
        return 'todo'
    try:
        rec = json.loads(o.read_text())
    except ValueError:
        return 'todo'
    if rec.get('status') == 'ok':
        return 'done' if (d / 'render' / 'manifest.json').exists() else 'render'
    if rec.get('harness') == 'pi':   # imported by pi/run.py collect: pi_eval already spent the retries
        return 'done'
    tries = len(list(d.glob('outcome.infra*.json'))) + 1
    return 'retry' if tries < MAX_INFRA_TRIES else 'done'


def render(src: Path, dst: Path, sif: str, code_dir: Path):
    """render.py inside the SIF; src may be missing (no files submitted -> empty manifest)."""
    dst.mkdir(parents=True, exist_ok=True)
    if src is None or not src.exists() or not any(p.is_file() for p in src.rglob('*')):
        (dst / 'manifest.json').write_text(json.dumps({'files': [], 'n_files': 0}))
        return True
    tmp = Path(os.environ.get('_CONDOR_SCRATCH_DIR', '/tmp')) / f'rtmp_{os.getpid()}_{time.time_ns()}'
    tmp.mkdir(parents=True)
    err = ''
    try:
        for iso in (['--containall'], ['--contain', '--ipc']):   # some hosts forbid a new /proc (no PID namespace)
            try:
                r = subprocess.run(['apptainer', 'exec', *iso, '--cleanenv', '-B', f'{src}:/in:ro', '-B',
                                    f'{dst}:/out', '-B', f'{code_dir}:/code:ro', '-B', f'{tmp}:/tmp', '--env',
                                    'HOME=/tmp', sif, 'python', '/code/render.py', '/in', '/out'],
                                   capture_output=True, text=True, timeout=3600)
            except subprocess.TimeoutExpired:
                err = 'render timed out after 3600 s'
                break
            err = r.stderr[-5000:] if r.returncode else ''
            if not (r.returncode == 255 and "can't mount proc" in r.stderr):
                break
        if err:
            (dst / 'render_error.txt').write_text(err)
        if not (dst / 'manifest.json').exists():
            # final, so the attempt doesn't stay in 'render' forever; the judge sees the files exist, not "no files"
            files = [{'name': str(p.relative_to(src)), 'ext': p.suffix.lower(), 'size': p.stat().st_size,
                      'kind': 'other', 'text': '', 'images': [], 'n_pages': None,
                      'notes': ['this file could not be rendered for review']}
                     for p in sorted(src.rglob('*')) if p.is_file()]
            (dst / 'manifest.json').write_text(json.dumps({'files': files, 'n_files': len(files),
                                                            'render_failed': True}))
            return False
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def stage_work(a):
    from aa_evals.gdpval.agent import run_attempt
    run_dir = RUNS / a.run_id
    cfg = json.loads((run_dir / 'config.json').read_text())
    require_offline(cfg)
    if (cfg.get('harness') or {}).get('name') == 'pi':
        sys.exit('a Pi-harness run generates with aa_evals.common.pi_eval (see aa_evals/gdpval/pi/run.py)')
    scratch = Path(os.environ.get('_CONDOR_SCRATCH_DIR') or '/tmp') / f'gdpval_{a.run_id}_{os.getpid()}'
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ['HOME'] = str(scratch)
    # Stirrup's run cache (state + files, written when a run errors) is fixed at import to ~/.cache/stirrup: keep it
    # in node scratch, not /home
    import stirrup.core.cache as stirrup_cache
    stirrup_cache.DEFAULT_CACHE_DIR = scratch / 'stirrup-cache'
    sif_local = scratch / 'gdpval.sif'
    if not sif_local.exists():
        subprocess.run(['dd', f'if={cfg["sandbox"]["sif"]}', f'of={sif_local}', 'bs=64M', 'status=none'], check=True)
    cfg['sandbox']['sif_local'] = str(sif_local)
    code_dir = Path(__file__).resolve().parent
    tasks = {t.task_id: t for t in T.load_tasks()}
    keys = [key(t, r) for t in cfg['selection']['task_ids'] for r in range(cfg['repeats'])]
    claims = Claims(run_dir)
    active: set[str] = set()
    log = lambda *x: print(now(), *x, flush=True)  # noqa: E731
    stop = run_dir / 'STOP_WORKERS'

    async def heartbeat():
        while True:
            for k in list(active):
                claims.beat(k)
            await asyncio.sleep(60)

    async def one(k):
        tid, rep = k.rsplit('__r', 1)
        d = run_dir / 'attempts' / k
        st = attempt_state(run_dir, k)
        if st in ('todo', 'retry'):
            if st == 'retry':
                n = len(list(d.glob('outcome.infra*.json'))) + 1
                os.replace(d / 'outcome.json', d / f'outcome.infra{n}.json')
            out = await run_attempt(tasks[tid], int(rep), cfg=cfg, work=scratch / k, out_dir=d)
            log('done', k, out['status'], out['end'], f'turns={out["turns"]}', f'files={out["n_submitted"]}',
                f'wall={out["wall_sec"]}', (out.get('error') or '')[:200].replace('\n', ' '))
            if out['status'] != 'ok':
                return
        if attempt_state(run_dir, k) == 'render':
            ok = await asyncio.to_thread(render, d / 'deliverables', d / 'render', cfg['sandbox']['sif_local'],
                                         code_dir)
            log('rendered', k, ok)

    sem = asyncio.Semaphore(a.concurrency)
    hb = asyncio.create_task(heartbeat())
    running: set[asyncio.Task] = set()

    async def guarded(k):
        try:
            await one(k)
        except Exception as e:  # noqa: BLE001
            log('worker-error', k, type(e).__name__, str(e)[:500])
        finally:
            active.discard(k)
            claims.release(k)
            sem.release()

    def next_key():
        # chosen when a slot is free, so a claim that went stale (a dead worker) is picked up at once, not on the
        # next pass over the whole list
        pending = False
        for k in keys:
            if attempt_state(run_dir, k) not in ('todo', 'retry', 'render'):
                continue
            pending = True
            if (claims.dir / k).exists():
                if k in active or not claims.stale(k):
                    continue
                log('stale-claim', k)
                claims.release(k)
            if claims.take(k):
                return k, True
        return None, pending

    while not stop.exists():
        await sem.acquire()
        k, pending = await asyncio.to_thread(next_key)   # ~1k Lustre stats: off the event loop
        if k is None:
            sem.release()
            if not pending:
                break
            await asyncio.sleep(30)
            continue
        active.add(k)
        t = asyncio.create_task(guarded(k))
        running.add(t)
        t.add_done_callback(running.discard)
    if running:
        await asyncio.gather(*running, return_exceptions=True)
    hb.cancel()
    shutil.rmtree(scratch, ignore_errors=True)
    log('worker exit')


# ---------------------------------------------------------------------------------------------------------- prepare
def stage_prepare(a):
    """Render every task's reference files and the human expert's deliverables once (CPU job, inside the SIF)."""
    code_dir = Path(__file__).resolve().parent
    sif = a.sif or str(SIF)
    import concurrent.futures as cf
    tasks = T.load_tasks()

    def one(t):
        out = PREP / t.task_id
        res = {}
        for name, files in (('refs', t.reference_files), ('expert', t.deliverable_files)):
            dst = out / name
            if (dst / 'manifest.json').exists() and not a.force:
                res[name] = True
                continue
            src = Path(os.environ.get('_CONDOR_SCRATCH_DIR', '/tmp')) / f'prep_{t.task_id}_{name}'
            shutil.rmtree(src, ignore_errors=True)
            src.mkdir(parents=True)
            for f in files:
                shutil.copy2(T.DATA / f, src / Path(f).name)
            res[name] = render(src, dst, sif, code_dir)
            shutil.rmtree(src, ignore_errors=True)
        return t.task_id, res

    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        for tid, res in ex.map(one, tasks):
            print(now(), tid, res, flush=True)
    bad = [t.task_id for t in tasks if not (PREP / t.task_id / 'expert' / 'manifest.json').exists()]
    print('missing expert renders:', bad)


# ---------------------------------------------------------------------------------------------------------- judge
def judge_spec(a):
    from aa_evals.judges.arms import ARMS
    arm = ARMS[a.judge_arm]
    spec = a.judge_endpoint or f'serving:{arm.server}'
    return arm, spec


async def stage_judge(a):
    from aa_evals.gdpval import judge as J
    run_dir = RUNS / a.run_id
    cfg = json.loads((run_dir / 'config.json').read_text())
    arm, spec = judge_spec(a)
    extra = dict(arm.extra)
    budget = J.Budget()
    images = a.judge_images
    jkey = f'{a.judge_arm}' + ('' if images else '-textonly')
    out_path = run_dir / 'matches' / f'{jkey}.jsonl'
    meta = {'arm': a.judge_arm, 'spec': spec, 'extra': extra, 'prompt_version': J.JUDGE_PROMPT_VERSION,
            'budget': budget.describe(), 'images': images, 'orders': ['MA', 'MB'], 'opponent': elo.EXPERT}
    meta_path = run_dir / 'matches' / f'{jkey}.meta.json'
    if meta_path.exists():
        previous = json.loads(meta_path.read_text())
        if {k: v for k, v in previous.items() if k != 'history'} != meta:
            meta['history'] = previous.get('history', []) + [{k: v for k, v in previous.items() if k != 'history'}]
        elif previous.get('history'):
            meta['history'] = previous['history']
    meta_path.write_text(json.dumps(meta, indent=1))
    tasks = {t.task_id: t for t in T.load_tasks()}
    judge = J.Judge(spec, extra, budget=budget)
    sem = asyncio.Semaphore(a.concurrency)
    log = lambda *x: print(now(), *x, flush=True)  # noqa: E731
    fh = open(out_path, 'a')
    marker = run_dir / 'matches' / f'{JUDGE_MARKER}{os.getpid()}-{os.uname().nodename}'

    async def keep_marker():   # judge_active(): Pi-harness re-imports must not rewrite the match file under us
        while True:
            marker.write_text(json.dumps({'pid': os.getpid(), 'host': os.uname().nodename, 'arm': jkey, 'at': now()}))
            await asyncio.sleep(60)

    marker_task = asyncio.create_task(keep_marker())

    async def one(k, order):
        tid, rep = k.rsplit('__r', 1)
        t = tasks[tid]
        mine = run_dir / 'attempts' / k / 'render'
        exp = PREP / tid / 'expert'
        a_dir, b_dir = (mine, exp) if order == 'MA' else (exp, mine)
        stats = {}
        try:
            msgs = J.build_messages(t, PREP / tid / 'refs', a_dir, b_dir, budget, images, stats)
        except J.RenderIntegrityError as error:
            # A broken evaluator cache is unscored grading work. Return normally so guarded()
            # keeps this key in seen: --follow must not automatically retry it or regenerate an agent.
            res = {'winner': None, 'error': str(error), 'tries': 0,
                   'grading_pending': error.as_dict()}
        else:
            async with sem:
                res = await judge.ask(msgs, stats=stats)
        rec = {'key': k, 'task_id': tid, 'repeat': int(rep), 'order': order, 'opponent': elo.EXPERT,
               'winner': res.get('winner'), 'error': res.get('error'), 'tries': res.get('tries'),
               'usage': res.get('usage'), 'prompt_stats': stats, 'text_tail': (res.get('text') or '')[-1500:],
               'at': now()}
        if res.get('grading_pending'):
            rec['grading_pending'] = res['grading_pending']
        if rec['winner']:
            rec['model_score'] = J.score_for(order, rec['winner'])
        fh.write(json.dumps(rec) + '\n')
        fh.flush()
        log('judged', k, order, rec['winner'], rec.get('error') or '')

    seen = set()
    inflight: set[asyncio.Task] = set()

    async def guarded(k, o):
        # one slow verdict (a long generation, gate backpressure) no longer holds back the rest: each judgment is
        # its own task and the loop below keeps adding new ones every pass
        try:
            await one(k, o)
        except Exception as e:  # noqa: BLE001
            log('judge-infra', k, o, str(e)[:300])
            seen.discard((k, o))

    while True:
        done = {(r['key'], r['order']) for r in read_jsonl(out_path)
                if r.get('winner') or (r.get('error') and not a.retry_judge_errors)}
        keys = [key(t, r) for t in cfg['selection']['task_ids'] for r in range(cfg['repeats'])]
        ready = [k for k in keys if (run_dir / 'attempts' / k / 'render' / 'manifest.json').exists()]
        todo = [(k, o) for k in ready for o in ('MA', 'MB') if (k, o) not in done and (k, o) not in seen]
        missing_prep = {k.rsplit('__r', 1)[0] for k, _ in todo
                        if not (PREP / k.rsplit('__r', 1)[0] / 'expert' / 'manifest.json').exists()}
        todo = [(k, o) for k, o in todo if k.rsplit('__r', 1)[0] not in missing_prep]
        if missing_prep:
            log('waiting for prepare renders:', len(missing_prep))
        seen.update(todo)
        write_items(run_dir, item_rows(run_dir, cfg, keys, load_outcomes(run_dir, keys), jkey))
        for k, o in todo:
            t = asyncio.create_task(guarded(k, o))
            inflight.add(t)
            t.add_done_callback(inflight.discard)
        all_done = all(attempt_state(run_dir, k) == 'done' for k in keys)
        if not a.follow:
            if inflight:
                await asyncio.gather(*inflight)
            write_items(run_dir, item_rows(run_dir, cfg, keys, load_outcomes(run_dir, keys), jkey))
            break
        if all_done and not todo and not missing_prep and not inflight:
            write_items(run_dir, item_rows(run_dir, cfg, keys, load_outcomes(run_dir, keys), jkey))
            break
        await asyncio.sleep(60)
    fh.close()
    marker_task.cancel()
    marker.unlink(missing_ok=True)
    await judge.aclose()


JUDGE_MARKER = 'judge.active.'   # not *.jsonl: the match globs never see it
JUDGE_MARKER_FRESH_SEC = 300


def judge_active(run_dir) -> bool:
    """A judge stage is running on this run (its marker was refreshed within JUDGE_MARKER_FRESH_SEC)."""
    for p in (Path(run_dir) / 'matches').glob(JUDGE_MARKER + '*'):
        try:
            if time.time() - p.stat().st_mtime < JUDGE_MARKER_FRESH_SEC:
                return True
        except OSError:
            pass
    return False


# ---------------------------------------------------------------------------------------------------------- score
def aa_reference(slug):
    if not slug or not LEADERBOARD.exists():
        return None
    d = json.loads(LEADERBOARD.read_text())
    for m in d.get('models', d if isinstance(d, list) else []):
        if m.get('slug') == slug:
            v = (m.get('scores') or {}).get(AA_KEY)
            return {'slug': slug, 'name': m.get('name'), 'elo': v,
                    'implied_winrate_vs_expert': round(elo.winrate_from_elo(v), 4) if v else None}
    return None


def successful_judgments(run_dir, arm):
    """Latest successful record per attempt/orientation; an error row is never a completed judgment."""
    recs = {}
    if arm:
        for r in read_jsonl(run_dir / 'matches' / f'{arm}.jsonl'):
            if r.get('winner') and r.get('model_score') in (0, 0.5, 1) and r.get('order') in ('MA', 'MB'):
                recs[(r['key'], r['order'])] = r
    return recs


def judgment_completeness(outcomes, recs):
    eligible = {k for k, v in outcomes.items() if v.get('status') == 'ok'}
    expected = {(k, order) for k in eligible for order in ('MA', 'MB')}
    completed = expected.intersection(recs)
    paired = sum((k, 'MA') in completed and (k, 'MB') in completed for k in eligible)
    partial = sum(((k, 'MA') in completed) != ((k, 'MB') in completed) for k in eligible)
    return {'n_expected_judgments': len(expected), 'n_completed_judgments': len(completed),
            'n_pending_judgments': len(expected - completed), 'n_complete_attempts': paired,
            'n_partial_attempts': partial, 'n_unjudged_attempts': len(eligible) - paired - partial,
            'complete': completed == expected, 'partial': completed != expected}


def item_rows(run_dir, cfg, keys, outcomes, arm=None):
    """One row per generated attempt; a final score needs successful judgments in both A/B orders."""
    recs = successful_judgments(run_dir, arm)
    rows = []
    for k in keys:
        o = outcomes.get(k)
        if o is None:
            continue
        tid, rep = k.rsplit('__r', 1)
        row = {'key': f'{tid}/r{rep}', 'item_id': tid, 'repeat': int(rep), 'status': o.get('status'),
               'end': o.get('end'), 'error': o.get('error'), 'n_files': o.get('n_submitted'), 'judge': arm,
               'agent_network': o.get('agent_network', cfg.get('agent_network', {'mode': 'legacy-online', 'recorded': False})),
               'score_MA': (recs.get((k, 'MA')) or {}).get('model_score'),
               'score_MB': (recs.get((k, 'MB')) or {}).get('model_score'),
               'transcript': str(run_dir / 'attempts' / k / 'transcript.json.gz')}
        scores = [x for x in (row['score_MA'], row['score_MB']) if x is not None]
        eligible = o.get('status') == 'ok'
        row['score'] = sum(scores) / 2 if eligible and len(scores) == 2 else None
        row['judge_status'] = ('excluded_infra' if not eligible else 'complete' if len(scores) == 2 else
                               'partial' if scores else 'pending')
        rows.append(row)
    return rows


def write_items(run_dir, rows):
    tmp = run_dir / f'items.jsonl.partial.{os.getpid()}'
    with open(tmp, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    os.replace(tmp, run_dir / 'items.jsonl')


def load_outcomes(run_dir, keys):
    out = {}
    for k in keys:
        p = run_dir / 'attempts' / k / 'outcome.json'
        if p.exists() and attempt_state(run_dir, k) == 'done':
            try:
                out[k] = json.loads(p.read_text())
            except ValueError:
                pass
    return out


def _render_failed(d: Path) -> bool:
    try:
        return bool(json.loads((d / 'render' / 'manifest.json').read_text()).get('render_failed'))
    except (OSError, ValueError):
        return False


def judge_settings_key(run_dir, arm):
    """sha256 of the judge's recorded settings (arm, sampling, prompt version, budget, images, orders), not its URL."""
    try:
        meta = json.loads((run_dir / 'matches' / f'{arm}.meta.json').read_text())
    except (OSError, ValueError):
        return None
    meta = {k: v for k, v in meta.items() if k != 'spec'}
    if 'history' in meta:
        meta['history'] = [{k: v for k, v in old.items() if k != 'spec'} for old in meta['history']]
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:16]


def stage_score(a):
    run_dir = RUNS / a.run_id
    cfg = json.loads((run_dir / 'config.json').read_text())
    keys = [key(t, r) for t in cfg['selection']['task_ids'] for r in range(cfg['repeats'])]
    outcomes = load_outcomes(run_dir, keys)
    arms = [a.judge_arm] if a.judge_arm else sorted(p.stem for p in (run_dir / 'matches').glob('*.jsonl'))
    per_arm = {}
    items = item_rows(run_dir, cfg, keys, outcomes)
    # tasks whose gold set has no expert deliverable (35 of 220 in openai/gdpval@11e7900): every submission "wins"
    # against an empty side, so they can't anchor a vs-expert fit. Their attempts still run and are judged (kept for
    # model-vs-model fits); here they are left out of the Elo and the win rate (#183).
    no_expert = {t.task_id for t in T.load_tasks() if not t.deliverable_files}
    primary = a.judge_arm or (DEFAULT_JUDGE_ARM if DEFAULT_JUDGE_ARM in arms else (arms[0] if arms else None))
    for arm in arms:
        recs = {pair: r for pair, r in successful_judgments(run_dir, arm).items()
                if outcomes.get(pair[0], {}).get('status') == 'ok'}
        completeness = judgment_completeness(outcomes, recs)
        anchored_expected = {(k, order) for k, v in outcomes.items() if v.get('status') == 'ok' and
                             k.rsplit('__r', 1)[0] not in no_expert for order in ('MA', 'MB')}
        completeness.update(n_expected_anchored_judgments=len(anchored_expected),
                            n_pending_anchored_judgments=len(anchored_expected.difference(recs)))
        if arm == primary:
            items = item_rows(run_dir, cfg, keys, outcomes, arm)
        by_task: dict[str, list] = {}
        unanchored = []
        for (k, order), r in recs.items():
            if outcomes.get(k, {}).get('status') != 'ok':
                continue
            if r['task_id'] in no_expert:
                unanchored.append(r['model_score'])
                continue
            by_task.setdefault(r['task_id'], []).append((a.run_id, elo.EXPERT, r['model_score'], 0.5))
        obs = [o for v in by_task.values() for o in v]
        n_obs = len(obs)
        if not obs:
            per_arm[arm] = {'n_judgments': 0, **completeness,
                            'judge_errors': sum(bool(r.get('error')) and not r.get('winner') for r in
                                                read_jsonl(run_dir / 'matches' / f'{arm}.jsonl'))}
            continue
        ratings = elo.fit(obs)
        ci = elo.bootstrap(by_task, n=a.bootstrap)
        win = sum(s for _, _, s, _ in obs) / n_obs
        flips = []
        for k in {k for k, _ in recs}:
            if (k, 'MA') in recs and (k, 'MB') in recs:
                flips.append(recs[(k, 'MA')]['model_score'] != recs[(k, 'MB')]['model_score'])
        ma = [r['model_score'] for (k, o), r in recs.items() if o == 'MA']
        mb = [r['model_score'] for (k, o), r in recs.items() if o == 'MB']
        ties = sum(r['winner'] == 'tie' for r in recs.values())
        per_arm[arm] = {
            **completeness,
            'elo': round(ratings[a.run_id], 1), 'ci95': [round(x, 1) for x in ci.get(a.run_id, (None, None))],
            'winrate_vs_expert': round(win, 4), 'n_judgments': n_obs, 'n_tasks': len(by_task),
            'ties': ties, 'position': {'model_as_A': round(sum(ma) / len(ma), 4) if ma else None,
                                       'model_as_B': round(sum(mb) / len(mb), 4) if mb else None,
                                       'order_flip_rate': round(sum(flips) / len(flips), 4) if flips else None},
            'judge_errors': sum(1 for r in read_jsonl(run_dir / 'matches' / f'{arm}.jsonl')
                                if r.get('error') and not r.get('winner')),
            'excluded_no_expert': {'n_judgments': len(unanchored),
                                   'winrate': round(sum(unanchored) / len(unanchored), 4) if unanchored else None},
        }
    ends: dict[str, int] = {}
    for o in outcomes.values():
        ends[o.get('end')] = ends.get(o.get('end'), 0) + 1
    n_infra = sum(o.get('status') == 'infra_error' for o in outcomes.values())
    n_cutoff = sum(o.get('end') == 'time_limit' for o in outcomes.values())
    headline = per_arm.get(primary, {})
    primary_completeness = judgment_completeness(outcomes, successful_judgments(run_dir, primary))
    turns = [o.get('turns', 0) for o in outcomes.values() if o.get('status') == 'ok']
    summary = {
        'agent_network': cfg.get('agent_network', {'mode': 'legacy-online', 'recorded': False}),
        'benchmark': 'GDPval-AA', 'metric': 'Elo (BT vs the human expert deliverable, expert = 968 on AA v2.1 scale)',
        'score': headline.get('elo'), 'ci95': headline.get('ci95'), 'judge_arm': primary, 'per_judge': per_arm,
        'n_tasks_scored': len([t for t in cfg['selection']['task_ids'] if t not in no_expert]),
        'n_tasks_no_expert': len([t for t in cfg['selection']['task_ids'] if t in no_expert]),
        'n_items': len(cfg['selection']['task_ids']), 'n_repeats': cfg['repeats'], 'n_attempts': len(keys),
        'n_attempted': len(outcomes), 'n_infra_errors': n_infra, 'n_infra': n_infra, 'n_cutoff': n_cutoff,
        'n_bad_jsonl_lines': BAD_LINES.count(run_dir),
        'valid': n_infra <= 0.02 * max(1, len(keys)) and len(outcomes) == len(keys) and
                 primary_completeness['complete'] and headline.get('elo') is not None and
                 not BAD_LINES.count(run_dir),
        'ends': ends, 'mean_turns': round(sum(turns) / len(turns), 1) if turns else None,
        'n_context_summarizations': sum((o.get('logger') or {}).get('summarizations', 0) for o in outcomes.values()),
        'n_output_cap_hits': sum((o.get('client') or {}).get('output_cap_hits', 0) for o in outcomes.values()),
        'n_render_failed': sum(_render_failed(run_dir / 'attempts' / k) for k in outcomes),
        'aa_reference': aa_reference(cfg['model'].get('aa_slug')),
        **protocol.summary_fields(model=cfg['model']['spec'], served_name=cfg['model']['served_name'],
                                  body=cfg['model']['body'], context=cfg['context'], judge_arm=primary,
                                  judge_settings_key=judge_settings_key(run_dir, primary), code_sha=cfg['git_sha'],
                                  repeats=cfg['repeats'],
                                  image_inputs=cfg['model']['vision']),
    }
    write_items(run_dir, items)
    summary['generation_complete'] = len(outcomes) == len(keys)
    summary['complete'] = summary['generation_complete'] and primary_completeness['complete'] and primary is not None
    summary['partial'] = not summary['complete']
    summary['n_pending_generations'] = len(keys) - len(outcomes)
    summary['n_pending_judgments'] = primary_completeness['n_pending_judgments']
    summary['n_pending'] = summary['n_pending_generations'] + primary_completeness['n_partial_attempts'] + \
        primary_completeness['n_unjudged_attempts']
    try:
        from aa_evals.tracejudge import fold_summary
        summary = fold_summary(run_dir, summary)
    except ImportError:
        pass
    tmp = run_dir / 'summary.json.partial'
    tmp.write_text(json.dumps(summary, indent=1) + '\n')
    os.replace(tmp, run_dir / 'summary.json')
    print(json.dumps({k: summary[k] for k in ('score', 'ci95', 'n_attempted', 'n_infra', 'ends', 'aa_reference',
                                              'per_judge')}, indent=1))


def status(a):
    run_dir = RUNS / a.run_id
    cfg = json.loads((run_dir / 'config.json').read_text())
    keys = [key(t, r) for t in cfg['selection']['task_ids'] for r in range(cfg['repeats'])]
    st: dict[str, int] = {}
    ends: dict[str, int] = {}
    for k in keys:
        s = attempt_state(run_dir, k)
        st[s] = st.get(s, 0) + 1
        p = run_dir / 'attempts' / k / 'outcome.json'
        if p.exists():
            e = json.loads(p.read_text()).get('end')
            ends[e] = ends.get(e, 0) + 1
    claims = len(list((run_dir / 'claims').iterdir())) if (run_dir / 'claims').exists() else 0
    m = {p.stem: len(read_jsonl(p)) for p in (run_dir / 'matches').glob('*.jsonl')}
    print(json.dumps({'attempts': len(keys), 'state': st, 'ends': ends, 'active_claims': claims, 'judged': m},
                     indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-id')
    ap.add_argument('--stage', choices=['init', 'work', 'judge', 'score', 'prepare'], default=None)
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--model', help='serving:<name> | url:<base>@<served_name>')
    ap.add_argument('--reasoning-effort', default=None)
    ap.add_argument('--extra-body', default=None, help='JSON merged over the request body')
    ap.add_argument('--context-window', type=int, default=protocol.CONTEXT_WINDOW,
                    help='total tokens per request (default 131072, M3.0); 0 = the endpoint\'s own (Qwen3.8 calibration)')
    ap.add_argument('--max-output', type=int, default=None, help='output cap per reply (default: the rest of the window)')
    ap.add_argument('--vision', choices=['on', 'off'], default=None, help='override the launch-argv image check')
    ap.add_argument('--agent-network', choices=['offline'], default='offline',
                    help='new runs: agent code_exec has no egress; host model/setup/grading networking stays enabled')
    ap.add_argument('--aa-slug', default=None)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--task-ids', default=None)
    ap.add_argument('--task-set', choices=T.TASK_SETS, default=T.DEFAULT_TASK_SET,
                    help="text-only-v1 (default, #346): drops 13 tasks needing vision/audio; 'all': the full 220")
    ap.add_argument('--repeats', type=int, default=REPEATS)
    ap.add_argument('--wall-sec', type=int, default=6 * 3600, help='per-attempt wall-clock backstop (AA states none)')
    ap.add_argument('--concurrency', type=int, default=16)
    ap.add_argument('--judge-arm', default=None, help='aa_evals/judges/arms.py arm (default: qwen38-27b-medium-pin)')
    ap.add_argument('--judge-endpoint', default=None, help='serve the arm from this spec instead of its own server')
    ap.add_argument('--retry-judge-errors', action='store_true', help='judge: retry failed verdicts; keep successful orientations')
    ap.add_argument('--judge-images', type=int, default=1, help='0 = text-only judging')
    ap.add_argument('--follow', action='store_true', help='judge: keep polling until every attempt is done')
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--runtime-cache-manifest', default=None,
                    help='reviewed immutable generic-dependency manifest; required for the three explicit-stack tasks')
    ap.add_argument('--sif', default=None)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(argv)
    if a.status:
        return status(a)
    if a.stage == 'prepare':
        return stage_prepare(a)
    if not a.run_id:
        ap.error('--run-id is required')
    if a.stage == 'init':
        if not a.model:
            ap.error('--model is required for init')
        return stage_init(a)
    if a.stage == 'work':
        return asyncio.run(stage_work(a))
    if a.stage == 'judge':
        a.judge_arm = a.judge_arm or 'qwen38-27b-medium-pin'
        return asyncio.run(stage_judge(a))
    if a.stage == 'score':
        return stage_score(a)
    ap.error('pass --stage or --status')


if __name__ == '__main__':
    main()
