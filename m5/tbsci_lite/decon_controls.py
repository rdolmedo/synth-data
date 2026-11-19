"""Positive controls for the TB-Sci-lite decontamination (#353): a bundle tar of hand-written variants of real lite
tasks in the synthetic task layout, added to the pool as one more unit so retrieval, windows and the judge treat them
like any other candidate (blind: s900-* ids, no canary, no lite task name). decon_final.py reports their recall and
keeps them out of the list.

  python -m m5.tbsci_lite.decon_controls --out /data/aai/m5/tbsci_lite/decon-final/controls/controls.tar

decon_controls/controls.json describes each control; decon_controls/<name>/ holds its instruction, solution sketch and
authored environment files. Files copied from the lite task lose their canary lines (and the control's `replace`
pairs), so text copies are near-identical and binary or canary-free ones identical.
"""
import argparse
import io
import json
from pathlib import Path
import tarfile
import time

HERE = Path(__file__).resolve().parent / 'decon_controls'
LITE_TAR = '/data/aai/aa_evals/data/tbsci_lite/v2.1/tasks.tar'
CLOSING = ('\n\nYou have 7200 seconds to complete this task. Do not cheat by using online solutions or hints specific '
           'to this task.\n')
DOCKERFILE = ('FROM local/tasksynth-sci:1\n{pip}COPY files/ /\nCOPY tasksynth/ /tasksynth-meta/\n'
              'RUN bash /tasksynth-meta/setup.sh && rm -rf /tasksynth /tasksynth-meta/setup.sh\n'
              'RUN rm -rf /tasksynth-meta\nWORKDIR /app\n')
TEST_SH = ('#!/bin/bash\n# tasksynth tasks ship no verifier: record the final state for the trace judge; the reward is '
           'always 0.\nmkdir -p /logs/verifier\npython3 /tests/final_state.py /tests/paths.json > '
           '/logs/verifier/final_state.json 2> /logs/verifier/final_state.err\necho 0 > /logs/verifier/reward.txt\n')
MARKERS = ('canary', 'benchmark data should never')


def clean(data, replace):
    """Text files: drop canary lines and apply the replace pairs; binaries unchanged."""
    if b'\0' in data[:8192]:
        return data
    text = data.decode(errors='surrogateescape')
    lines = [l for l in text.splitlines(keepends=True) if not any(m in l.lower() for m in MARKERS)]
    text = ''.join(lines)
    for a, b in replace:
        text = text.replace(a, b)
    return text.encode(errors='surrogateescape')


def lite_files(wanted):
    """{(task, rel): bytes} for the (task, rel) pairs in wanted, from the lite tar."""
    out = {}
    with tarfile.open(LITE_TAR, 'r:') as t:
        for m in t:
            parts = m.name.split('/', 2)
            if m.isreg() and len(parts) == 3 and (parts[1], parts[2]) in wanted:
                out[(parts[1], parts[2])] = t.extractfile(m).read()
    missing = wanted - set(out)
    if missing:
        raise SystemExit(f'not in {LITE_TAR}: {sorted(missing)}')
    return out


def task_files(c, lite):
    name = c['name']
    d = HERE / name
    files = {'instruction.md': ((d / 'instruction.md').read_text().rstrip() + CLOSING).encode(),
             'solution/sketch.md': (d / 'sketch.md').read_bytes()}
    env = {}
    for src, dst in c.get('copy', []):
        env[dst] = clean(lite[(c['source'], src)], c.get('replace', []))
    for rel in c.get('files', []):
        env[rel] = (d / 'files' / rel).read_bytes()
    for rel, data in env.items():
        files[f'environment/files/{rel}'] = data
    pip = f'RUN pip install --no-cache-dir {" ".join(c["pip"])}\n' if c.get('pip') else ''
    files['environment/Dockerfile'] = (c.get('dockerfile') or DOCKERFILE.format(pip=pip)).encode()
    if not c.get('dockerfile'):
        setup = ['#!/bin/bash', 'set -euo pipefail', 'mkdir -p /app/data /app/output /tasksynth']
        if c.get('setup'):
            setup.append(c['setup'])
        setup += ['chmod -R a+rX /app/data', 'chmod -R a+rwX /app/output']
        files['environment/tasksynth/setup.sh'] = ('\n'.join(setup) + '\n').encode()
    inputs = sorted('/' + r for r in env if r.startswith('app/'))
    files['tests/test.sh'] = TEST_SH.encode()
    files['tests/paths.json'] = json.dumps({'inputs': inputs, 'outputs': []}, indent=1).encode()
    files['task.toml'] = (f'schema_version = "1.0"\n\n[metadata]\nauthor_name = "tasksynth"\n'
                          f'category = "{c["action"]}"\ntags = ["{c["domain"]}", "{c["action"]}", "substrate"]\n'
                          f'title = "{c["title"]}"\nexpert_time_estimate_min = 120\ntasksynth_spec = "{name}"\n'
                          f'tasksynth_generator = "deepseek-v41-flash-1017"\n\n[verifier]\ntimeout_sec = 60.0\n\n'
                          f'[agent]\ntimeout_sec = 21600.0\n\n[environment]\nnetwork_mode = "no-network"\n'
                          f'build_timeout_sec = 1800\ncpus = 2\nmemory_mb = 8192\nstorage_mb = 10240\ngpus = 0\n').encode()
    spec = {'spec_id': name, 'bench': 'tbsci_lite', 'domain': c['domain'], 'subtopic': {'name': c['subtopic']},
            'agent_action': c['action'], 'source': 'simulated', 'minutes': 120, 'closing_seconds': 7200}
    files['tasksynth.json'] = json.dumps({'spec': spec, 'meta': {'title': c['title'], 'inputs': inputs},
                                          'provenance': {'round': 'decon-positive-control'}}, indent=1).encode()
    return files


def build(out):
    controls = json.loads((HERE / 'controls.json').read_text())['controls']
    lite = lite_files({(c['source'], src) for c in controls for src, _ in c.get('copy', [])})
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + '.partial')
    now = time.time()
    with tarfile.open(tmp, 'w') as t:
        for c in controls:
            for rel, data in sorted(task_files(c, lite).items()):
                info = tarfile.TarInfo(f'tasks/{c["name"]}/{rel}')
                info.size, info.mtime, info.mode = len(data), now, 0o644
                t.addfile(info, io.BytesIO(data))
    tmp.replace(out)
    manifest = [{k: c[k] for k in ('name', 'source', 'kind')} for c in controls]
    out.with_suffix('.json').write_text(json.dumps(manifest, indent=1))
    print(f'{len(controls)} controls -> {out}')
    return manifest


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    build(ap.parse_args().out)
