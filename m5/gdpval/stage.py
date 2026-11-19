"""Problem-side checks and Harbor bundles for generated GDPval problems (runs in a CPU condor job, not on gpu123).

  python -m m5.gdpval.stage check  --gen GEN.jsonl [GEN2.jsonl ...] --out DIR [--workers 8] [--ids-from N --ids-to M]
      runs each synthetic-input script in the GDPval SIF (offline: user+net namespace, --contain), opens every listed
      file inside the SIF (sif_validate.py), and writes DIR/check.jsonl (per problem: accepted or why not, file stats,
      text extracts) and DIR/inputs.tar (accepted problems' input files, <pid>/<name>)
  python -m m5.gdpval.stage bundle --check DIR [DIR2 ...] --out BUNDLE.tar [--docs-root ROOT]
      Harbor tasks for the accepted problems, in the format of aa_evals/gdpval/pi/tasks.py build_task: AA's prompts
      (pi_instruction), inputs at /home/user/<name>, an empty /home/user/outputs, the same task.toml/test.sh

Problem IDs: gdpm5-<spec index, 6 digits>.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

SIF = '/data/aai/aa_evals/sif/gdpval-v1.sif'
HERE = Path(__file__).resolve().parent
SCRIPT_TIMEOUT = 240
SCRIPT_MEM_KB = 4 * 1024 * 1024   # address-space cap per input script (a runaway script took a producer to 32 GB)
SOFT = ('input_not_named_in_prompt', 'real_doc_not_named_in_prompt')


def pid(idx: int) -> str:
    return f'gdpm5-{idx:06d}'


def sif_run(home: Path, tmp: Path, argv: list[str], timeout: int) -> subprocess.CompletedProcess:
    cmd = ['/usr/bin/unshare', '--user', '--map-root-user', '--net', '--', 'apptainer', '--silent', 'exec',
           '--contain', '--ipc', '--cleanenv', '--no-eval', '--no-nv', '-B', f'{home}:/home/user', '-B', f'{tmp}:/tmp',
           '-B', f'{HERE}:/opt/m5:ro', '--pwd', '/home/user', '--env', 'HOME=/home/user', '--env', 'MPLBACKEND=Agg',
           SIF, *argv]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def check_one(rec: dict, scratch: Path) -> tuple[dict, dict[str, bytes]]:
    p = pid(rec['idx'])
    res = {'pid': p, 'idx': rec['idx'], 'lane': rec['spec']['lane'], 'accepted': False}
    if 'problem' not in rec:
        res['why'] = 'parse_error' if rec.get('parse_error') else ('request_error' if rec.get('usage', {}).get('error')
                                                                   else 'no_problem')
        return res, {}
    hard = [b for b in rec.get('static_bad') or [] if not b.startswith(SOFT)]
    res['soft'] = [b for b in rec.get('static_bad') or [] if b.startswith(SOFT)]
    if hard:
        res['why'] = 'static:' + ';'.join(hard)
        return res, {}
    prob = rec['problem']
    names = [n for n, _ in prob['input_files']]
    if not names:
        res['accepted'] = True
        return res, {}
    d = Path(tempfile.mkdtemp(dir=scratch, prefix=p + '-'))
    home, tmp = d / 'home', d / 'tmp'
    home.mkdir()
    tmp.mkdir()
    try:
        (home / '_make_inputs.py').write_text(prob['script'])
        t0 = time.time()
        try:
            r = sif_run(home, tmp, ['bash', '-c', f'ulimit -v {SCRIPT_MEM_KB}; exec python3 /home/user/_make_inputs.py'],
                        SCRIPT_TIMEOUT)
        except subprocess.TimeoutExpired:
            res['why'] = 'script_timeout'
            return res, {}
        res['script_sec'] = round(time.time() - t0, 1)
        if r.returncode:
            res['why'] = 'script_failed'
            res['stderr'] = r.stderr[-1500:]
            return res, {}
        (home / '_make_inputs.py').unlink()
        made = sorted(str(x.relative_to(home)) for x in home.rglob('*') if x.is_file())
        res['extra_files'] = [m for m in made if m not in names]
        v = sif_run(home, tmp, ['bash', '-c', f'ulimit -v {SCRIPT_MEM_KB}; exec python3 /opt/m5/sif_validate.py /home/user '
                                + ' '.join(__import__('shlex').quote(n) for n in names)], 300)
        try:
            files = json.loads(v.stdout)
        except ValueError:
            res['why'] = 'validate_failed'
            res['stderr'] = (v.stderr or v.stdout)[-1500:]
            return res, {}
        res['files'] = files
        bad = [n for n, f in files.items() if not f['ok']]
        if bad:
            res['why'] = 'file_check:' + ';'.join(f'{n}:{files[n].get("error")}' for n in bad)
            return res, {}
        total = sum(f['bytes'] for f in files.values())
        if total > 50 << 20:
            res['why'] = f'inputs_too_large:{total}'
            return res, {}
        res['accepted'] = True
        return res, {n: (home / n).read_bytes() for n in names}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def read_gen(paths, lo=None, hi=None):
    best = {}
    for path in paths:
        for ln in Path(path).read_text().splitlines():
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if (lo is not None and r['idx'] < lo) or (hi is not None and r['idx'] >= hi):
                continue
            if r['idx'] not in best or 'problem' in r:
                best[r['idx']] = r
    return [best[k] for k in sorted(best)]


def check(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = read_gen(a.gen, a.ids_from, a.ids_to)
    scratch = Path(os.environ.get('_CONDOR_SCRATCH_DIR') or tempfile.gettempdir()) / 'm5check'
    scratch.mkdir(parents=True, exist_ok=True)
    local_tar = scratch / 'inputs.tar'
    n_acc = 0
    with open(out / 'check.jsonl.partial', 'w') as fo, tarfile.open(local_tar, 'w') as tf, \
            cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(check_one, r, scratch): r for r in recs}
        for f in cf.as_completed(futs):
            r = futs[f]
            try:
                res, blobs = f.result()
            except Exception as e:  # noqa: BLE001
                res, blobs = {'pid': pid(r['idx']), 'idx': r['idx'], 'accepted': False,
                              'why': f'check_error:{type(e).__name__}:{str(e)[:200]}'}, {}
            for n, b in blobs.items():
                ti = tarfile.TarInfo(f'{res["pid"]}/{n}')
                ti.size, ti.mtime = len(b), 0
                tf.addfile(ti, io.BytesIO(b))
            n_acc += res['accepted']
            fo.write(json.dumps(res) + '\n')
            fo.flush()
    shutil.copyfile(local_tar, out / 'inputs.tar.partial')
    os.replace(out / 'inputs.tar.partial', out / 'inputs.tar')
    os.replace(out / 'check.jsonl.partial', out / 'check.jsonl')
    print(f'checked {len(recs)}: accepted {n_acc}', flush=True)


# ------------------------------------------------------------------------------------------------------- bundle
def build_task(root: Path, rec: dict, files: dict[str, bytes]) -> dict:
    from aa_evals.common.pi_eval import task_inputs
    from aa_evals.gdpval import tasks as T
    from aa_evals.gdpval.office_fix import fix_file
    from aa_evals.gdpval.pi import tasks as PT

    p = pid(rec['idx'])
    d = root / p
    (d / 'environment').mkdir(parents=True)
    (d / 'tests').mkdir()
    inp = d / task_inputs.INPUTS
    (inp / PT.OUTPUT_DIR.lstrip('/')).mkdir(parents=True)
    targets = []
    for n, b in files.items():
        f = inp / 'home/user' / n
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b)
        fix_file(f)
        targets.append((f, f'{T.HOME}/{n}'))

    class SynthTask(T.Task):
        def ref_targets(self):
            return targets

    task = SynthTask(p, rec['spec']['sector'], rec['spec']['occupation'], rec['problem']['task_prompt'])
    instruction = PT.pi_instruction(task)
    (d / 'instruction.md').write_text(instruction)
    q = json.dumps
    (d / 'task.toml').write_text(PT.TASK_TOML.format(outputs=PT.OUTPUT_DIR, version='gdpval-m5-v1', task_id=p,
                                                     sector=q(task.sector), occupation=q(task.occupation),
                                                     base=PT.BASE_REF, cpus=PT.CPUS, memory=PT.MEMORY_MB))
    (d / 'environment' / 'Dockerfile').write_text(PT.DOCKERFILE)
    (d / 'tests' / 'test.sh').write_text(PT.TEST_SH)
    (d / 'tests' / 'test.sh').chmod(0o755)
    task_inputs.write_manifest(d, {'m5': {'idx': rec['idx'], 'lane': rec['spec']['lane']}})
    return {'pid': p, 'n_inputs': len(files), 'instruction_chars': len(instruction)}


def bundle(a):
    gen = {r['idx']: r for r in read_gen(a.gen)}
    picked = []
    for cdir in a.check:
        cdir = Path(cdir)
        inputs = {}
        with tarfile.open(cdir / 'inputs.tar') as tf:
            for m in tf.getmembers():
                if m.isfile():
                    p, n = m.name.split('/', 1)
                    inputs.setdefault(p, {})[n] = tf.extractfile(m).read()
        for ln in (cdir / 'check.jsonl').read_text().splitlines():
            c = json.loads(ln)
            if c['accepted']:
                rec = gen[c['idx']]
                files = dict(inputs.get(c['pid'], {}))
                if rec['spec']['lane'] == 'real':
                    doc = rec['doc']
                    from m5.gdpval.docs import pdf_bytes
                    files[doc['name']] = pdf_bytes(doc)
                picked.append((rec, files))
    if a.limit:
        picked = picked[:a.limit]
    out = Path(a.out)
    scratch = Path(os.environ.get('_CONDOR_SCRATCH_DIR') or tempfile.gettempdir())
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        root = Path(tmp) / 'tasks'
        root.mkdir()
        meta = [build_task(root, rec, files) for rec, files in picked]
        part = Path(tmp) / 'bundle.tar'

        def norm(ti):
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ''
            ti.mtime = 0
            return ti
        with tarfile.open(part, 'w') as tf:
            for p in sorted(Path(tmp).rglob('*')):
                if p == part:
                    continue
                tf.add(p, arcname=p.relative_to(tmp).as_posix(), recursive=False, filter=norm)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(part, out.with_suffix('.tar.partial'))
        os.replace(out.with_suffix('.tar.partial'), out)
    out.with_suffix('.json').write_text(json.dumps({'bundle_version': 'gdpval-m5-v1', 'n': len(meta),
                                                   'tasks': {m['pid']: m for m in meta}}, indent=1))
    print(f'{out}: {len(meta)} tasks')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['check', 'bundle'])
    ap.add_argument('--gen', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--check', nargs='+')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--ids-from', type=int, default=None)
    ap.add_argument('--ids-to', type=int, default=None)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    (check if a.cmd == 'check' else bundle)(a)


if __name__ == '__main__':
    main()
