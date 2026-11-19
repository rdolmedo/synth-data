"""pi_eval's frozen code snapshot (harbor_mswea.run.snapshot_code) carries everything its control-side modules import:
the run is created from the snapshot and anything run from it must not reach into the live checkout (#449, #451)."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

PROBE = r'''
import json, sys
from pathlib import Path
from aa_evals.common.pi_eval import adaptive, control
import tbsched.work
assert Path(adaptive.__file__).resolve().is_relative_to(Path.cwd().resolve()), adaptive.__file__
assert Path(tbsched.work.__file__).resolve().is_relative_to(Path.cwd().resolve()), tbsched.work.__file__
run = Path(sys.argv[1])
print(json.dumps({'state': adaptive.state(str(run)), 'control': control.update(run, {'max_in_flight': 3})}))
'''


@pytest.mark.skipif(shutil.which('rsync') is None, reason='snapshot_code copies with rsync')
def test_snapshot_imports_control_side_modules(tmp_path):
    pytest.importorskip('harbor')
    from aa_evals.common.harbor_mswea import run as hm
    assert 'tbsched' in hm.CODE_DIRS
    snap = Path(hm.snapshot_code(root=tmp_path / 'code'))
    assert (snap / '.complete').exists() and (snap / 'tbsched' / 'work.py').exists()
    run = tmp_path / 'run'
    (run / 'queue' / 'claims').mkdir(parents=True)
    (run / 'queue' / 'items.txt').write_text('t1 0\nt2 0\n')
    (run / 'control.json').write_text('{"max_starts": 5}')
    # as a job runs it: the snapshot alone on the path, no user site
    r = subprocess.run([sys.executable, '-s', '-c', PROBE, str(run)], cwd=snap, capture_output=True, text=True,
                       env={'PATH': '/usr/bin:/bin', 'PYTHONPATH': str(snap), 'PYTHONNOUSERSITE': '1'})
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout)
    assert out == {'state': [0, 2], 'control': {'max_starts': 5, 'max_in_flight': 3}}
