"""Held jobs, for long-running server jobs of every pipeline (GPU model servers above all): the policy decided on
2026-11-14 (#378, coordinator): a held server must never block its run.

1. Memory holds (HoldReasonCode 34: the cgroup at its limit with nothing left to reclaim) are released by condor
   itself with more memory, bounded: request_memory grows by MEMORY_STEP of the base per memory hold so far, and
   periodic_release acts for at most MEMORY_RELEASES of them. A too-tight shape then costs a restart (new node,
   ~2-15 min start-up), not a dead model. This is optim-servers' aa-serving safety net (serve_teacher, validated live
   2026-11-14: held at 1024 MB -> auto-released at 1280 MB -> finished), moved here unchanged.
2. Any other hold, or a memory hold past that bound: the run removes the held job and submits a replacement that
   excludes the node it was held on (held_action -> 'replace'; remove_held returns that node and the hold code).
   After memory holds past the bound the replacement starts bigger (replacement_gib: the last released size, 1.5x),
   so a too-small base does not cycle; a run replaces one server at most MAX_REPLACEMENTS times, then alerts.
3. A user's own hold (HoldReasonCode 1, condor_hold) is deliberate: left alone.

  submit file:  request_memory = {memory_request(gib)}
                periodic_release = {MEMORY_RELEASE}
  the run's loop, for each of its server jobs in JobStatus 5:
                action, node, code = remove_held(job)
                if action == 'replace' and replaced < MAX_REPLACEMENTS:
                    resubmit excluding node, with replacement_gib(gib, code) GiB

Stdlib only.
"""
from __future__ import annotations

import subprocess

MEMORY_HOLDS = 'ifThenElse(isUndefined(NumHoldsByReason.JobOutOfResources), 0, NumHoldsByReason.JobOutOfResources)'
MEMORY_STEP = 0.25
MEMORY_RELEASES = 2
MEMORY_RELEASE = f'(HoldReasonCode =?= 34) && ({MEMORY_HOLDS} <= {MEMORY_RELEASES})'
MAX_REPLACEMENTS = 3          # per server and run: a hold that fails the same way every time must not cycle forever
USER_HOLD = 1                 # condor_hold by the user
# condor_q -af fields that held_action reads, in order: HoldReasonCode, memory holds so far, the node it ran on
HOLD_FIELDS = ('HoldReasonCode', MEMORY_HOLDS, 'LastRemoteHost')


def memory_request(gib: float) -> str:
    """request_memory (MB) as an expression: the base, plus MEMORY_STEP of it per memory hold so far."""
    base = int(gib * 1024)
    return f'{base} + {int(base * MEMORY_STEP)} * {MEMORY_HOLDS}'


def submit_lines(gib: float) -> str:
    """The two submit-file lines of the memory net."""
    return f'request_memory = {memory_request(gib)}\nperiodic_release = {MEMORY_RELEASE}\n'


def held_action(hold_code, memory_holds, has_net: bool = True) -> str:
    """'release' when the job's own periodic_release will release it (a memory hold within the bound; only with
    has_net), 'wait' for a user's own hold or unknown values (condor_q failed: decide next pass), else 'replace'."""
    try:
        code, holds = int(hold_code), int(memory_holds)
    except (TypeError, ValueError):
        return 'wait'
    if code == USER_HOLD:
        return 'wait'   # deliberate: whoever held it releases or removes it
    # has_net=False: a job submitted without the memory net (no periodic_release): nothing will release it
    return 'release' if has_net and code == 34 and holds <= MEMORY_RELEASES else 'replace'


def node_of(remote_host) -> str | None:
    """'slot1_2@cpu302.internal.cluster.internal' -> 'cpu302'."""
    if not remote_host or remote_host == 'undefined':
        return None
    return remote_host.split('@', 1)[-1].split('.', 1)[0] or None


def hold_info(job_id: str, runner=subprocess.run) -> tuple:
    """(HoldReasonCode, memory holds, node) of a job; (None, None, None) when condor_q fails."""
    try:
        r = runner(['condor_q', job_id, '-af', *HOLD_FIELDS], capture_output=True, text=True, timeout=60)
        p = (r.stdout or '').split()
        if r.returncode or len(p) < 3:
            return None, None, None
        return p[0], p[1], node_of(p[2])
    except (OSError, subprocess.SubprocessError):
        return None, None, None


def replacement_gib(gib: float, hold_code) -> float:
    """Memory of a replacement: after memory holds past the bound, the largest size the net already tried (base x
    (1 + MEMORY_STEP x MEMORY_RELEASES)), so a too-small base does not cycle; else the base."""
    try:
        memory = int(hold_code) == 34
    except (TypeError, ValueError):
        memory = False
    return gib * (1 + MEMORY_STEP * MEMORY_RELEASES) if memory else gib


def remove_held(job_id: str, runner=subprocess.run, has_net: bool = True) -> tuple[str, str | None, str | None]:
    """Decide for one held job (CLUSTER.PROC) and act: ('release', None, code) leaves it to condor; ('replace', node,
    code) removed it (resubmit a replacement excluding `node`, sized by replacement_gib); ('wait', None, code) when
    its state could not be read, it is a user's hold, or it was no longer held (the removal matches JobStatus == 5
    only, so a job released meanwhile is never removed). has_net=False for jobs submitted without submit_lines: their
    memory holds are replaced too (nothing would release them)."""
    cluster, _, proc = str(job_id).partition('.')
    if not (cluster.isdigit() and proc.isdigit()):
        raise ValueError(f'remove_held needs CLUSTER.PROC, got {job_id!r}')
    code, holds, node = hold_info(job_id, runner)
    action = held_action(code, holds, has_net)
    if action != 'replace':
        return action, None, code
    const = f'ClusterId == {cluster} && ProcId == {proc} && JobStatus == 5'
    try:
        r = runner(['condor_rm', '-constraint', const], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return 'wait', None, code
    return ('replace', node, code) if r.returncode == 0 else ('wait', None, code)
