"""Build-and-check self-test on hand-written packages (no LLM). Run inside a CPU job:
  python -m task_synthesis.pipeline.selftest <out_dir>"""
import asyncio
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('HARBOR_NS_PERSIST_IMAGES', '0')

from .check import check, make_store  # noqa: E402
from .package import materialize, parse  # noqa: E402

WORDS = ' '.join(['context'] * 60)

LEDGER = f'''<task_json>
{{"slug": "ledger-check", "title": "Ledger", "kind": "substrate", "expert_minutes": 30, "apt": [], "pip": [],
 "inputs": ["/app/data/ledger.csv", "/app/README.md"], "outputs": ["/app/report.json"],
 "defect_check": "", "services": false, "summary": "s"}}
</task_json>
<instruction>
{WORDS} Read /app/data/ledger.csv and write /app/report.json; scratch space is /tmp/work.
</instruction>
<file path="/app/README.md">
# Ledger
</file>
<file path="/tasksynth/gen.py">
import csv, random
random.seed(0)
with open('/app/data/ledger.csv', 'w') as f:
    w = csv.writer(f)
    w.writerow(['id', 'amount'])
    for i in range(100):
        w.writerow([i, round(random.gauss(100, 20), 2)])
</file>
<setup>
#!/bin/bash
set -euo pipefail
mkdir -p /app/data
python3 /tasksynth/gen.py
</setup>
<solution_sketch>
sum
</solution_sketch>'''

SERVICE = f'''<task_json>
{{"slug": "svc-check", "title": "Service", "kind": "puzzle", "expert_minutes": 30, "apt": [], "pip": [],
 "inputs": ["/srv/site/index.html"], "outputs": [],
 "defect_check": "curl -s http://127.0.0.1:8080/index.html | grep -q BROKEN", "services": true, "summary": "s"}}
</task_json>
<instruction>
{WORDS} The page at http://127.0.0.1:8080/ served from /srv/site is broken; fix /srv/site/index.html.
</instruction>
<file path="/srv/site/index.html">
<html>BROKEN</html>
</file>
<setup>
#!/bin/bash
set -euo pipefail
true
</setup>
<services>
#!/bin/bash
cd /srv/site && nohup python3 -m http.server 8080 >/var/log/site.log 2>&1 &
for i in $(seq 1 30); do curl -s http://127.0.0.1:8080/ >/dev/null && exit 0; sleep 1; done
exit 1
</services>
<solution_sketch>
edit
</solution_sketch>'''


BROKEN = LEDGER.replace('random.seed(0)', 'raise SystemExit(3)')


async def main(out):
    store = make_store()
    results = {}
    for name, text in (('ledger', LEDGER), ('service', SERVICE), ('broken', BROKEN)):
        pkg = parse(text)
        d = materialize(pkg, {'spec_id': name, 'minutes': 30}, {'model': 'selftest'}, Path(out) / name)
        results[name] = await check(store, d)
        print(name, json.dumps({k: v for k, v in results[name].items() if k != 'snapshot'}), flush=True)
        print(name, 'tree', results[name].get('snapshot', {}).get('tree', [])[:8], flush=True)
    # every task image is gone afterwards, the failed build's too (r6 filled job scratch with failed builds)
    left = sorted(p.name for p in (store.local_root / 'images').iterdir()) if (store.local_root / 'images').exists() else []
    print('ok' if results['ledger']['ok'] and results['service']['ok'] and not results['broken']['ok'] else 'FAIL',
          'images left:', left, flush=True)
    (Path(out) / 'selftest.json').write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    asyncio.run(main(sys.argv[1]))
