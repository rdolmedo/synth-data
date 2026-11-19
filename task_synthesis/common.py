"""Shared paths and helpers for the task-synthesis pipeline."""
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# TASKSYNTH_WORKTREES: where frozen checkouts go (e.g. on /data, off the home quota); default unchanged
WORKTREES = Path(os.environ.get('TASKSYNTH_WORKTREES') or '/opt/aai/worktrees')
ROOT = Path('/data/aai/tbench/tasksynth')


def frozen_checkout():
    """A detached worktree of HEAD for jobs to run from, so edits here never reach a running job."""
    sha = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                         check=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(REPO), 'status', '--porcelain', '--untracked-files=no'],
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise SystemExit('Commit first: jobs run from a frozen worktree of HEAD')
    path = WORKTREES / f'tasksynth-run-{sha[:10]}'
    if not path.exists():
        subprocess.run(['git', '-C', str(REPO), 'worktree', 'add', '--detach', str(path), sha], check=True,
                       capture_output=True)
    return path


# ------------------------------------------------------------------------------------------------- prune-checkouts
# Each deploy freezes a checkout (frozen_checkout) and nothing removed them: 129 had piled up by 2026-11-05 (~166 MB
# each, ~21 GB of /lustre home; #245, R10). prune_checkouts applies the checks the coordinator used by hand that day.
CHECKOUT_PREFIX = 'tasksynth-run-'
# run metadata that names the checkout a job runs from (job.json, run.json, campaign.json, *.sub, *.toml): the
# production, serving and synth roots, plus the rounds' and binds' job files. Globs stay shallow (a round's work/
# holds millions of files).
METADATA_GLOBS = (
    '/data/aai/tbench/tasksynth/production/*/*.json', '/data/aai/tbench/tasksynth/production/*/*.sub',
    '/data/aai/tbench/tasksynth/production/*/*.toml',
    '/data/aai/tbench/tasksynth/serving/*/*.json', '/data/aai/tbench/tasksynth/serving/*/*.sub',
    '/data/aai/tbench/tasksynth/rounds/*/job.json', '/data/aai/tbench/tasksynth/rounds/*/*.sub',
    '/data/aai/tbench/tasksynth/assets/binds/*/job.json', '/data/aai/tbench/tasksynth/assets/binds/*/*.sub',
    '/data/aai/tbench/synth/*/*.toml', '/data/aai/tbench/synth/*/ops/campaign.json',
    '/data/aai/tbench/synth/*/ops/*.sub', '/data/aai/tbench/synth/*/ops/jobs/*.sub',
    '/data/aai/tbench/synth/*/ops/jobs/*.json',
)


def git(path, *args):
    return subprocess.run(['git', '-C', str(path), *args], capture_output=True, text=True)


def condor_ads():
    """Text of every queued or running job's ad (condor_q -l, all users), or None if the schedd did not answer."""
    r = subprocess.run(['condor_q', '-l'], capture_output=True, text=True, timeout=300)
    return r.stdout if r.returncode == 0 else None


def local_references():
    """Command lines and working directories of this node's processes."""
    import os
    out = []
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            out.append(open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='replace'))
            out.append(os.readlink(f'/proc/{pid}/cwd'))
        except OSError:   # gone, or another user's
            continue
    return '\n'.join(out)


def pin_planned_run(name):
    """A round or bind submitted by hand for a production run (`round submit` / `bind submit` --round/--name <run>):
    this process onto the run's registry root (run.json serving_env; tbshared.serving.pin), as the run's driver is, so
    the job gets the run's root and not the shell's. No run.json of that name: nothing (the process's own root)."""
    from .pipeline import production
    info = production.read_json(production.RUNS / name / 'run.json') if name else None
    if info:
        production.pin_run_root(info)


def recent_metadata(days, globs=None):
    """{path: text} of the run metadata files modified in the last `days` days. Default globs: METADATA_GLOBS, plus the
    serving registry $TASKSYNTH_SERVING_ROOT names when it is another one (a canary's servers name checkouts too)."""
    import glob
    import time
    if globs is None:
        from tbshared import serving
        reg = serving.registry_root()
        globs = METADATA_GLOBS + ((f'{reg}/*/*.json', f'{reg}/*/*.sub') if reg != serving.ROOT else ())
    cutoff, out = time.time() - days * 86400, {}
    for pattern in globs:
        for p in glob.glob(pattern):
            try:
                if Path(p).stat().st_mtime >= cutoff:
                    out[p] = Path(p).read_text(errors='replace')
            except OSError:
                continue
    return out


def checkout_verdict(path, min_age_days, ads, local, metadata, now=None):
    """Why the frozen checkout must stay ([] = removable)."""
    import time
    path, why = Path(path), []
    now = now or time.time()
    try:
        created = (path / '.git').stat().st_mtime   # written when the worktree was added
    except OSError:
        return ['not a git worktree']
    if now - created < min_age_days * 86400:
        why.append(f'created {(now - created) / 86400:.1f} days ago (< {min_age_days})')
    refs = (str(path), path.name)
    if ads is None:
        why.append('condor_q failed: cannot tell whether a job uses it')
    elif any(r in ads for r in refs):
        why.append('named by a queued or running condor job')
    if any(r in local for r in refs):
        why.append('named by a local process (args or cwd)')
    named = [p for p, text in metadata.items() if any(r in text for r in refs)]
    if named:
        why.append(f'named by recent run metadata: {named[:3]}')
    status = git(path, 'status', '--porcelain', '--ignored', '--untracked-files=all')
    if status.returncode != 0:
        why.append(f'git status failed: {status.stderr.strip()[:200]}')
    else:
        dirty = [l for l in status.stdout.splitlines()
                 if '__pycache__/' not in l and not l.rstrip().endswith('.pyc')]
        if dirty:
            why.append(f'not clean: {dirty[:3]}')
    head = git(path, 'rev-parse', 'HEAD').stdout.strip()
    remote = git(path, 'branch', '-r', '--contains', head) if head else None
    if not head or remote.returncode != 0 or not remote.stdout.strip():
        why.append(f'HEAD {head[:10] or "?"} is not on any origin branch')
    return why


def prune_checkouts(root=WORKTREES, min_age_days=2, metadata_days=4, remove=False, log=print,
                    ads=None, local=None, metadata=None):
    """Frozen run checkouts (root/tasksynth-run-*) that nothing uses any more: created at least `min_age_days` ago;
    not named by a queued or running condor job (condor_q -l), a local process (args, cwd) or run metadata modified
    in the last `metadata_days` days; clean except __pycache__; HEAD on an origin branch. A dry run unless `remove`,
    which runs `git worktree remove --force` (the only untracked files are __pycache__) from the main repository.
    Returns {checkout: [reasons to keep]}."""
    import time
    checkouts = sorted(p for p in Path(root).glob(f'{CHECKOUT_PREFIX}*') if p.is_dir())
    ads = condor_ads() if ads is None else ads
    local = local_references() if local is None else local
    metadata = recent_metadata(metadata_days) if metadata is None else metadata
    now, verdicts = time.time(), {}
    for path in checkouts:
        verdicts[path] = checkout_verdict(path, min_age_days, ads, local, metadata, now)
    keep = {p: w for p, w in verdicts.items() if w}
    drop = [p for p, w in verdicts.items() if not w]
    for p, w in keep.items():
        log(f'keep   {p.name}: {"; ".join(w)}')
    for p in drop:
        log(f'{"remove" if remove else "would remove"} {p.name} {git(p, "rev-parse", "HEAD").stdout.strip()}')
    if remove:
        for p in drop:
            common = Path(git(p, 'rev-parse', '--path-format=absolute', '--git-common-dir').stdout.strip())
            r = subprocess.run(['git', '-C', str(common.parent), 'worktree', 'remove', '--force', str(p)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                log(f'FAILED to remove {p.name}: {r.stderr.strip()[:300]}')
                verdicts[p] = [f'git worktree remove failed: {r.stderr.strip()[:200]}']
    log(f'{len(checkouts)} checkouts: {len(drop)} {"removed" if remove else "removable"}, {len(keep)} kept')
    return verdicts


def main():
    import argparse
    ap = argparse.ArgumentParser(description='task-synthesis maintenance')
    sub = ap.add_subparsers(dest='op', required=True)
    p = sub.add_parser('prune-checkouts', help='remove frozen run checkouts nothing uses (dry run by default)')
    p.add_argument('--root', default=str(WORKTREES))
    p.add_argument('--min-age-days', type=float, default=2)
    p.add_argument('--metadata-days', type=float, default=4)
    p.add_argument('--remove', action='store_true', help='remove them (default: list only)')
    a = ap.parse_args()
    prune_checkouts(a.root, a.min_age_days, a.metadata_days, a.remove, log=lambda m: print(m, flush=True))


if __name__ == '__main__':
    main()
