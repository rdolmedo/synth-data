"""Brainstorm leaves below each of AA's 86 subtopics with Opus (`claude -p`, modest volume: ~1 call per 40 leaves).

Input: subtopics.json (fit.py). Output: leaves/<domain>__<subtopic>.json (one per subtopic), merged by build.py.
Only labels and class-level statistics enter the prompt; no test question does.
"""
import concurrent.futures as cf, json, math, os, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).parent
SOURCES = {
    'Law': (['case_law', 'wikipedia'], 'US court opinions (Supreme Court, federal appeals, state supreme and appellate courts) and Wikipedia'),
    'Software Engineering': (['software_docs', 'changelog', 'wikipedia'],
                             'official language/library documentation, release notes, changelogs and Wikipedia'),
    'Health': (['pubmed', 'web_edu', 'wikipedia'], 'PubMed abstracts, educational web pages and Wikipedia'),
    'Finance': (['web_edu', 'sec_filings', 'wikipedia'], 'educational/professional web pages, SEC 10-K filings and Wikipedia'),
    'Humanities and Social Sciences': (['wikipedia', 'web_edu'], 'Wikipedia and educational web pages'),
    'Science Engineering and Mathematics': (['wikipedia', 'web_edu'], 'Wikipedia and educational/technical web pages'),
}
TOTAL_CALLS = 92000      # ~66k questions x 1.4 oversupply
PER_LEAF = 20            # target generator calls per leaf
BATCH = 40


def n_leaves(w):
    return max(30, min(160, round(TOTAL_CALLS * w / PER_LEAF)))


def ask(prompt, tries=3):
    for _ in range(tries):
        try:
            env = {k: v for k, v in os.environ.items() if k not in ('CLAUDE_CODE_SESSION_ID', 'CLAUDE_CODE_MESSAGING_SOCKET', 'CLAUDE_CODE_MESSAGING_TOKEN', 'CLAUDE_CODE_CHILD_SESSION', 'CLAUDECODE')}
            out = subprocess.run(['claude', '-p', '--model', 'opus', prompt], capture_output=True, text=True, env=env,
                                 timeout=1800).stdout
            m = re.search(r'\[.*\]', out, re.S)
            arr = json.loads(m.group(0))
            if isinstance(arr, list) and arr and all('leaf' in a and 'queries' in a for a in arr):
                return arr
        except Exception as e:
            print('retry', e, file=sys.stderr)
    raise RuntimeError('brainstorm failed')


def run_subtopic(domain, sub, w, outdir):
    out = outdir / f"{re.sub(r'[^A-Za-z0-9]+', '_', domain)}__{re.sub(r'[^A-Za-z0-9]+', '_', sub)}.json"
    if out.exists():
        return out
    tmpl = (HERE / 'leaf_prompt.md').read_text()
    types, desc = SOURCES[domain]
    want = n_leaves(w)
    leaves = []
    while len(leaves) < want:
        n = min(BATCH, want - len(leaves))
        avoid = ''
        if leaves:
            avoid = ('\nThese leaves already exist; propose different ones that fill remaining gaps:\n'
                     + '\n'.join(f'- {l["leaf"]}' for l in leaves))
        arr = ask(tmpl.format(domain=domain, subtopic=sub, n=n, avoid=avoid, sources_desc=desc,
                              source_types=', '.join(types)))
        seen = {l['leaf'].lower() for l in leaves}
        leaves += [a for a in arr if a['leaf'].lower() not in seen][:n]
    out.write_text(json.dumps({'domain': domain, 'subtopic': sub, 'weight': w, 'leaves': leaves}, indent=1,
                              ensure_ascii=False))
    print(f'{domain} / {sub}: {len(leaves)} leaves', flush=True)
    return out


def main():
    tax = json.loads((HERE / 'subtopics.json').read_text())
    outdir = HERE / 'leaves'
    outdir.mkdir(exist_ok=True)
    jobs = [(d, s, dv['weight'] * sv['weight']) for d, dv in tax['domains'].items() for s, sv in dv['subtopics'].items()]
    print(sum(n_leaves(w) for *_, w in jobs), 'leaves planned', flush=True)
    with cf.ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as ex:
        for f in cf.as_completed([ex.submit(run_subtopic, d, s, w, outdir) for d, s, w in jobs]):
            f.result()


if __name__ == '__main__':
    main()
