# harbor_ns multi-uid mode (opt-in)

By default a harbor_ns sandbox maps one id: uid/gid 0 is the calling user. Every other user is emulated by `nsfake.so` (see `nsfake.c`).

Some verifiers drop privileges before grading:

- `setpriv --reuid nobody` before running the submitted solver;
- `su nobody -c 'kill -9 -1'` to reap leftover processes;
- checks that a file is owned by the current user, or that a sandbox user cannot read `/tests`.

Under emulation these drops fail with `EPERM` or are silently no-ops, so the reference solution fails. In TB-Science v0.1.0 this hits about 10 of 70 tasks (`aa_evals/tbsci/README.md`).

## What it does

Enable it with `--ek ns_multiuid=true`, or with `HARBOR_NS_MULTIUID=1` in the environment Harbor runs in. Then, per trial:

| Step | Behavior |
|---|---|
| ID mapping | The pod's user namespace maps uid 0 → the caller and 1..N → the caller's `/etc/subuid` range, and the same for gids with `/etc/subgid`. `newuidmap`/`newgidmap` write the mapping; `setgroups` stays allowed. `setuid`/`setpriv`/`su` are then real. With the cluster's 65,536 subordinate ids the mapped ids are the bands 1..11000, 20000..65537 and 100000..108997 (`MULTIUID_BANDS`, chosen from a scan of the run uids of TB 2.1, TB-Science and lite: `60000 + index`, `65531–65537` of `tess-transit-vetting`, nobody 65534, a fresh `100000 + n` per study in `weighted-vaccine-inference`, ...). An id outside the bands is unmapped: `setuid` into it fails. |
| Emulation | Each sandbox of the pod runs without id emulation: the original `gosu` is restored and `nsfake.so` (the `/etc/ld.so.preload` entry) is replaced by `nsdev.so`, in that sandbox's overlay only; the cached image is not modified. `nsdev.so` (`nsdev.c`) keeps the id-independent parts of nsfake: `sched_setaffinity` onto CPUs outside the box's cpuset is remapped (`taskset -c 0-3`), `RLIMIT_CORE` is clamped, mknod of device nodes is emulated by bind mounts, and with `HARBOR_NS_CPU_VIEW=1` `sysconf(_SC_NPROCESSORS_*)`/`get_nprocs` report the box's CPU count (boxes are pinned to their task's `cpus`, 4 when unspecified, `HARBOR_NS_CPU_DEFAULT`). The CPU view is multi-uid only. |
| Log dirs | The trial's `/logs/*` bind mounts are made world-writable for the trial, so task verifier/agent users such as 65532 can write them (Docker's mounts were). Teardown chowns them back. |
| Non-root exec | Harbor's `exec(user=...)` enters as the real uid/gid (`nsenter -S/-G`). |
| Entrypoints | Each service's entrypoint runs as its image `USER`, as under Docker. `HOME` comes from that user's passwd entry unless the image or compose sets it. In single-uid mode, entrypoints run as root with `HOME=/root`. |
| Disk cap | `ns_diskcap` / `HARBOR_NS_DISKCAP` (`diskcap.py`): `RLIMIT_FSIZE` survives `setuid`/`setpriv`/`su`, so a box user's write past the cap gets EFBIG / SIGXFSZ. The host user cannot read directories that subordinate ids made private (0700) or their processes' `/proc/<pid>/task/*/io` (0400, owned by the id), so the watchdog's walks and write samples run as namespace root (`nsenter -U` into the pod); finding and killing those processes works from the host. Known gap, as single-uid: nsdev clamps only `RLIMIT_CORE`, so `ulimit -f unlimited` in a box fails with EPERM. Tests: `test_diskcap_multiuid.py`. |
| Teardown | Files owned by subordinate ids belong to no host user we control, so cleanup happens from inside the namespace before it goes away: sandbox dirs and pod volumes are removed (`rm -rf` as namespace root), and the trial's bind-mounted log dirs are chowned back to the caller. With `delete=False`, sandbox dirs are chowned instead of removed. |

Image builds are unchanged: they stay single-uid, so the shared image cache is the same in both modes. With the flag off, every code path is the default one.

## Requirements and limits

- **Subordinate ranges:** the user needs ranges in `/etc/subuid` and `/etc/subgid`, plus the setuid `newuidmap`/`newgidmap`. Cluster compute nodes have both (checked 2026-11-02 on gpu106, gpu123 and gpu166). Without them the pod fails to start with a clear error.
- **Ownership:** image files keep the single-uid build's ownership, so everything in the image is owned by root. Files that a Dockerfile `chown`s to a service user are root-owned here, while nsfake would have reported the recorded owner.
- **Where foreign-uid files live:** only on node-local scratch (sandbox overlays and the trial dir) while the trial runs. `aa_evals/tbsci/probe_worker.sh` lists any foreign-uid file left on a job's scratch after a run, and must find none.
