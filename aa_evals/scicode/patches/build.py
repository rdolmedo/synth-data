"""Build and validate a versioned SciCode-Verified patch list (our fix pass, #184).

    python -m aa_evals.scicode.patches.build --version v1 [--envs ...] [--skip-validate]

Payload (eval data, kept out of git): /data/aai/aa_evals/data/scicode/patches/<version>/
  patches.json, patches_*.json   {"base": "verified-v2", "patches": [patch, ...]}; all are concatenated
                    (patches.json first, then the others sorted), so several authors can keep separate files
  tests/<step>.json {"test_cases": [str, ...], "targets": [[var, ...], ...] | null, "target_code": str | null}
  refs/<pid>.py     a reference implementation of the problem's steps (at least up to every patched step)
  mutants/<step>/<name>.py   code appended after the ref: a wrong implementation the new tests must reject
  alts/<step>/<name>.py      code appended after the ref: a valid alternative (another RNG, convention, algorithm
                             the spec allows) the new tests must accept

A patch is one of
  {"id", "kind": "tests",   "step", "tests": "tests/<step>.json", "mutants": [names], "alternatives": [names], ...}
        replaces the step's test cases and its targets (written to the overlay h5). Targets come from
        "targets" (JSON values; {"py": "<expr>"} is eval'd with numpy as np) or from "target_code", a snippet run
        after the ref that sets `targets = [(var1, ...), ...]`.
  {"id", "kind": "text",    "step" | "problem", "field", "old", "new", ...}
        exact replacement (old must occur exactly once) in a step field (step_description_prompt, function_header,
        return_line, step_background) or a problem field (problem_description_main, required_dependencies, ...).
  {"id", "kind": "exclude", "step", ...}   generated as usual (later steps build on it) but never scored.
A tests patch may carry "validate_envs": [labels] when the problem's own dependency block cannot import in the other
stacks (the OR rule grades it only where it imports); the rationale must say why (mention validate_envs).
Every patch also carries "summary", "rationale" and "evidence" (free text; file paths for the long form).

Validation (unless --skip-validate), per tests patch, in every grading env: the ref passes the new tests, and every
listed mutant fails them, and every listed alternative passes them. The build refuses to write a version whose validation fails. Outputs:
/data/aai/aa_evals/data/scicode/verified_aa_<version>/{problems_test.jsonl, targets_overlay.h5, manifest.json,
validation.json}.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import pickle
import subprocess
import tempfile
from pathlib import Path

from .. import data, grader

PAYLOAD_ROOT = data.ROOT / 'patches'
LOCAL_ENVS = {'np126': 'local:/data/aai/aa_evals/data/scicode/envs/np126/bin/python',
              'np22': 'local:/data/aai/aa_evals/data/scicode/envs/np22/bin/python'}
STEP_FIELDS = {'step_description_prompt', 'function_header', 'return_line', 'step_background'}


def sha256_tree(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:
            h.update(str(p.relative_to(root)).encode() + b'\0' + p.read_bytes() + b'\0')
    return h.hexdigest()


def _canon(v):
    """A target value in a stable, picklable form for the per-step digest (arrays as dtype, shape and bytes)."""
    import numpy as np
    if isinstance(v, (np.ndarray, np.generic)):
        a = np.ascontiguousarray(v)
        return ('nd', str(a.dtype), a.shape, a.tobytes())
    if isinstance(v, (list, tuple)):
        return (type(v).__name__, tuple(_canon(x) for x in v))
    return ('py', repr(v))


def target_python(envs: dict, validate_envs=None) -> str:
    """The python that runs target_code: the local copy of the patch's grading stack (np22 unless the patch is
    restricted by validate_envs, e.g. a problem whose dependency block only imports in np126). It needs h5py."""
    lab = (validate_envs or ['np22'])[0]
    env = envs.get(lab, LOCAL_ENVS[lab])
    return (env if env.startswith('local:') else LOCAL_ENVS[lab]).split(':', 1)[1]


def find_step(probs, sid):
    for p in probs:
        for k, s in enumerate(p['sub_steps']):
            if s['step_number'] == sid:
                return p, k, s
    raise SystemExit(f'step {sid} not in the base dataset')


def ref_code(root: Path, pid: str) -> str:
    p = root / 'refs' / f'{pid}.py'
    if not p.exists():
        raise SystemExit(f'{p} missing (a tests patch needs a reference implementation)')
    return p.read_text()


def compute_targets(spec: dict, prob: dict, ref: str, python: str) -> list[tuple]:
    if spec.get('targets') is not None:
        import numpy as np  # noqa: F401 - for {"py": ...} values
        out = []
        for t in spec['targets']:
            out.append(tuple(eval(v['py'], {'np': np}) if isinstance(v, dict) and 'py' in v else v for v in t))
        return out
    script = '\n'.join([prob['required_dependencies'] or '', ref, spec['target_code'],
                        'import pickle, base64, sys',
                        'sys.stdout.write("@@T@@" + base64.b64encode(pickle.dumps([tuple(t) if isinstance(t, (tuple, list)) else (t,) for t in targets])).decode())'])
    with tempfile.TemporaryDirectory() as d:
        Path(d, 'mk.py').write_text(script)
        r = subprocess.run([python, '-I', f'{d}/mk.py'], capture_output=True, text=True, timeout=3600, cwd=d)
    if r.returncode != 0 or '@@T@@' not in r.stdout:
        raise SystemExit(f'target_code failed: {r.stderr[-2000:]}')
    return pickle.loads(base64.b64decode(r.stdout.split('@@T@@')[1]))


def write_overlay(path: Path, steps: dict[str, list[tuple]]):
    import h5py
    import numpy as np
    tmp = path.with_suffix('.partial')
    with h5py.File(tmp, 'w') as f:
        for sid, targets in steps.items():
            for i, t in enumerate(targets):
                if len(t) > 9:   # process_hdf5_to_tuple reads var1..varN in h5py's (alphabetical) key order
                    raise SystemExit(f'{sid} test {i + 1}: {len(t)} target vars; at most 9 keep their order')
                g = f.create_group(f'{sid}/test{i + 1}')
                for j, v in enumerate(t):
                    if isinstance(v, (bool, np.bool_)):
                        g.create_dataset(f'var{j + 1}', data=np.bool_(v))
                    else:
                        g.create_dataset(f'var{j + 1}', data=np.asarray(v))
    tmp.replace(path)


def validate(root: Path, patches: list[dict], probs: list[dict], overlay: Path, envs: dict, jobs: int = 1) -> list[dict]:
    work = []
    for pt in patches:
        if pt['kind'] != 'tests':
            continue
        prob, k, s = find_step(probs, pt['step'])
        ref = ref_code(root, str(prob['problem_id']))
        cases = [('ref', '', True)] + [(m, (root / 'mutants' / pt['step'] / f'{m}.py').read_text(), False)
                                        for m in pt.get('mutants', [])] + \
                [(f'alt:{m}', (root / 'alts' / pt['step'] / f'{m}.py').read_text(), True)
                 for m in pt.get('alternatives', [])]
        penvs = envs
        if pt.get('validate_envs'):   # a problem whose own dependency block cannot import in some stack (e.g. a
            # name removed from SciPy) is gradable only in the others; the OR rule grades it there, and so do we
            penvs = {k: v for k, v in envs.items() if k in pt['validate_envs']}
            if not penvs or 'validate_envs' not in pt.get('rationale', '') + pt.get('evidence', ''):
                raise SystemExit(f"{pt['id']}: validate_envs needs a known env and a stated reason (rationale)")
        for name, extra, want_pass in cases:
            work.append((pt, prob, k, s, ref + '\n' + extra, name, want_pass, penvs))

    def one(item):
        pt, prob, k, s, code, name, want_pass, penvs = item
        codes = [code] + [''] * k   # the ref holds every step's code; nothing else is prepended
        fn = lambda vendor, h5: grader.build_script(prob, k, codes, len(s['test_cases']), s['test_cases'],
                                                    vendor=vendor, h5=h5)
        r = grader.grade_step(fn, penvs, overlay, 1800, scratch=grader.scratch_dir())
        per = {e: v['pass'] for e, v in r['per_env'].items()}
        ok = all(v == want_pass for v in per.values())
        row = {'patch': pt['id'], 'step': pt['step'], 'case': name, 'want': 'pass' if want_pass else 'fail',
               'per_env': per, 'ok': ok, 'secs': {e: v.get('secs') for e, v in r['per_env'].items()}}
        if not ok:
            row['tail'] = {e: v['tail'][-800:] for e, v in r['per_env'].items()}
        print(f"{pt['step']:>6} {name:<24} want {row['want']:<4} got {per} {'OK' if ok else 'BAD'}", flush=True)
        return row

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max(1, jobs)) as ex:
        return list(ex.map(one, work))   # results in patch/case order


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', required=True)
    ap.add_argument('--envs', default=None, help='label=env,... (default: local copies of the two grading stacks)')
    ap.add_argument('--skip-validate', action='store_true')
    ap.add_argument('--jobs', type=int, default=1, help='validation cases graded in parallel')
    ap.add_argument('--files', default=None, help='only these patch files (comma-separated, relative to the '
                                                   'version dir): a check build, written to --out-dir')
    ap.add_argument('--out-dir', default=None, help='where to write (default: data/scicode/verified_aa_<version>)')
    a = ap.parse_args(argv)
    envs = dict(x.split('=', 1) for x in a.envs.split(',')) if a.envs else LOCAL_ENVS
    root = PAYLOAD_ROOT / a.version
    if a.files:
        files = [root / f for f in a.files.split(',')]
        if not a.out_dir:
            raise SystemExit('--files needs --out-dir (a partial build never replaces the version)')
    else:
        files = ([root / 'patches.json'] if (root / 'patches.json').exists() else []) + \
            sorted(p for p in root.glob('patches_*.json'))
    specs = [json.loads(f.read_text()) for f in files]
    if not specs or len({sp['base'] for sp in specs}) != 1:
        raise SystemExit(f'need patch files with one common base: {files}')
    spec = {'base': specs[0]['base'], 'patches': [pt for sp in specs for pt in sp['patches']]}
    base = data.load(spec['base'])
    probs = copy.deepcopy(base.problems)
    overlay_targets, excluded, ids = {}, [], set()
    for pt in spec['patches']:
        if pt['id'] in ids:
            raise SystemExit(f'duplicate patch id {pt["id"]}')
        ids.add(pt['id'])
        for key in ('summary', 'rationale', 'evidence'):
            if not pt.get(key):
                raise SystemExit(f'{pt["id"]}: "{key}" is required')
        if pt['kind'] == 'text':
            if 'step' in pt:
                _, _, target = find_step(probs, pt['step'])
                if pt['field'] not in STEP_FIELDS:
                    raise SystemExit(f'{pt["id"]}: bad step field {pt["field"]}')
            else:
                target = next(p for p in probs if str(p['problem_id']) == str(pt['problem']))
            n = target[pt['field']].count(pt['old'])
            if n != 1:
                raise SystemExit(f'{pt["id"]}: "old" occurs {n} times in {pt["field"]} (must be exactly once)')
            target[pt['field']] = target[pt['field']].replace(pt['old'], pt['new'])
        elif pt['kind'] == 'tests':
            prob, k, s = find_step(probs, pt['step'])
            t = json.loads((root / pt['tests']).read_text())
            s['test_cases'] = list(t['test_cases'])
            tg = compute_targets(t, prob, ref_code(root, str(prob['problem_id'])) if t.get('target_code') else '',
                                 target_python(envs, pt.get('validate_envs')))
            if len(tg) != len(s['test_cases']):
                raise SystemExit(f'{pt["id"]}: {len(tg)} targets for {len(s["test_cases"])} test cases')
            overlay_targets[pt['step']] = tg
        elif pt['kind'] == 'exclude':
            find_step(probs, pt['step'])
            excluded.append(pt['step'])
        else:
            raise SystemExit(f'{pt["id"]}: unknown kind {pt["kind"]}')
    out = Path(a.out_dir) if a.out_dir else data.ROOT / f'verified_aa_{a.version}'
    out.mkdir(parents=True, exist_ok=True)
    overlay = out / 'targets_overlay.h5'
    if overlay_targets:
        write_overlay(overlay, overlay_targets)
    validation = [] if a.skip_validate else validate(root, spec['patches'], probs, overlay, envs, a.jobs)
    bad = [v for v in validation if not v['ok']]
    (out / 'validation.json').write_text(json.dumps({'envs': envs, 'results': validation}, indent=1))
    if bad:
        raise SystemExit(f'validation failed for {len(bad)} cases (see {out}/validation.json); manifest not written')
    pj = out / 'problems_test.jsonl'
    pj.write_text(''.join(json.dumps(p, ensure_ascii=False) + '\n' for p in probs))
    man = {'version': a.version, 'base': spec['base'], 'n_patches': len(spec['patches']),
           'patch_list_sha256': sha256_tree(root), 'patch_files': [str(f.relative_to(root)) for f in files], 'problems_md5': data.md5(pj),
           'overlay_steps': sorted(overlay_targets), 'overlay_h5_md5': data.md5(overlay) if overlay_targets else None,
           'overlay_step_sha256': {sid: hashlib.sha256(pickle.dumps([tuple(map(_canon, t)) for t in tg])).hexdigest()
                                   for sid, tg in sorted(overlay_targets.items())},
           'excluded_steps': sorted(excluded), 'validated': not a.skip_validate,
           'n_validation_cases': len(validation), 'patch_ids': [p['id'] for p in spec['patches']]}
    (out / 'manifest.json').write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))


if __name__ == '__main__':
    main()
