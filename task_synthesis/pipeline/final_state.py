"""Final state of a tasksynth sandbox after the agent, for the trace judge (tasks ship no verifier).

Copied into each task as /tests/final_state.py and run by /tests/test.sh, with /tests/paths.json holding the task's
declared inputs and outputs. Prints JSON: every declared output (size, sha256, head of text files or a listing of
directories) and a bounded tree of the directories those paths live in. Standard library only, Python 3.8+.
"""
import hashlib
import json
import os
import sys

HEAD, TOTAL, LISTING, TREE = 4000, 60000, 50, 400
SKIP = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', '.cache', '.mypy_cache', '.pytest_cache', 'target'}


def sha256(path, limit=200 << 20):
    if os.path.getsize(path) > limit:
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def head(path, n):
    with open(path, 'rb') as f:
        data = f.read(n + 1)
    try:
        text = data[:n].decode()
    except UnicodeDecodeError:
        return {'binary_hex': data[:256].hex()}
    return {'text': text, 'truncated': len(data) > n}


def describe(path, budget):
    if not os.path.lexists(path):
        return {'exists': False}, 0
    if os.path.isdir(path):
        names = sorted(os.listdir(path))
        return {'exists': True, 'type': 'dir', 'entries': len(names), 'listing': names[:LISTING]}, 0
    if not os.path.isfile(path):
        return {'exists': True, 'type': 'other'}, 0
    out = {'exists': True, 'type': 'file', 'size': os.path.getsize(path), 'sha256': sha256(path),
           'executable': os.access(path, os.X_OK)}
    if budget > 0:
        out.update(head(path, min(HEAD, budget)))
    return out, len(out.get('text', ''))


def tree(roots):
    lines = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP)
            for name in sorted(filenames):
                p = os.path.join(dirpath, name)
                try:
                    lines.append(f'{os.path.getsize(p):>12} {p}')
                except OSError:
                    continue
                if len(lines) >= TREE:
                    return lines + [f'[... stopped at {TREE} files]']
    return lines


def main(paths_file):
    with open(paths_file) as f:
        paths = json.load(f)
    budget, outputs = TOTAL, {}
    for p in paths.get('outputs', []):
        outputs[p], used = describe(p, budget)
        budget -= used
    declared = paths.get('inputs', []) + paths.get('outputs', [])
    roots = sorted({'/' + p.strip('/').split('/')[0] for p in declared if p.strip('/')} - {'/etc', '/usr', '/var', '/tmp'})
    json.dump({'outputs': outputs, 'tree': tree([r for r in roots if os.path.isdir(r)])}, sys.stdout, indent=1)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '/tests/paths.json')
