#!/bin/bash
# Runs as PID 1 inside fresh namespaces created by
#   unshare --user --map-root-user --mount --pid --fork --net --ipc --uts --propagation private
# Sets up the sandbox rootfs and execs nsinit, which pivots into it.
#
# Inputs (environment):
#   NS_LOWER   colon-separated read-only image layers (top first)
#   NS_UPPER, NS_WORK, NS_MERGED   overlay dirs on node-local disk
#   NS_DIRECT  (build mode) rootfs to bind-mount writable at NS_MERGED instead
#              of an overlay
#   NS_PRIVATE_NET  1 if a private network namespace was created (bring up lo)
#   NS_HOSTNAME                    hostname inside the sandbox
#   NS_BINDS   newline-separated "host_src<TAB>container_dst" bind mounts
#   NS_NSINIT  path to the static nsinit binary
#   NS_ETC_DIR dir holding resolv.conf / hosts to install into /etc
#   NS_SHM_SIZE  optional size of /dev/shm (tmpfs size= value)
set -euo pipefail

fail() { echo "NSLAUNCH_FAIL: $*" >&2; exit 97; }

M="$NS_MERGED"
if [ -n "${NS_DIRECT:-}" ]; then
  mount --rbind "$NS_DIRECT" "$M" || fail "rootfs bind"
else
  mount -t overlay overlay -o "lowerdir=$NS_LOWER,upperdir=$NS_UPPER,workdir=$NS_WORK" "$M" \
    || fail "overlay mount"
fi

mkdir -p "$M/proc" "$M/dev" "$M/sys" "$M/tmp" "$M/etc"
# A private procfs needs a fully visible host /proc (fails under LXCfs unless the
# condor job sets +BypassLXCfs); fall back to the host's /proc.
if mount -t proc -o nosuid,nodev,noexec proc "$M/proc" 2>/dev/null; then
  echo "NSLAUNCH_PROC=private" >&2
else
  mount --rbind /proc "$M/proc" || fail "proc"
  echo "NSLAUNCH_PROC=host" >&2
fi

mount -t tmpfs -o nosuid,mode=755,size=65536k tmpfs "$M/dev" || fail "dev tmpfs"
# Device binds need a character-device mount point. Over a regular file, readdir reports
# /dev/zero, /dev/urandom, ... as regular files (d_type), so `find / -type f | xargs grep`,
# grep -r and os.scandir read endless streams: such commands ran for hours in the 2026-10-17/18
# campaigns. A whiteout node (c 0 0) is the one device a user namespace may create (Linux >= 5.8).
devnode() { mknod "$1" c 0 0 2>/dev/null || touch "$1"; }
for d in null zero full random urandom tty; do
  devnode "$M/dev/$d"
  mount --bind "/dev/$d" "$M/dev/$d" || fail "dev $d"
done
if [ -c /dev/net/tun ]; then  # TUN works inside a userns-owned netns (openvpn etc.; Docker privileged)
  mkdir -p "$M/dev/net"; devnode "$M/dev/net/tun"; mount --bind /dev/net/tun "$M/dev/net/tun" || true
fi
mkdir -p "$M/dev/pts" "$M/dev/shm" "$M/dev/mqueue"
mount -t devpts -o newinstance,ptmxmode=0666,mode=0620 devpts "$M/dev/pts" || fail "devpts"
ln -s pts/ptmx "$M/dev/ptmx"
# NS_SHM_SIZE (e.g. 4096m): cap the sandbox's /dev/shm. Unset: the kernel default (half the host's RAM), and tmpfs
# pages are invisible to the PSS guard (m1-hero #244: one sandbox's /dev/shm took a 48-slot worker from 37 to 89 GB).
mount -t tmpfs -o "nosuid,nodev,mode=1777${NS_SHM_SIZE:+,size=$NS_SHM_SIZE}" tmpfs "$M/dev/shm" || fail "shm"
mount -t mqueue mqueue "$M/dev/mqueue" 2>/dev/null || true
ln -s /proc/self/fd "$M/dev/fd"
ln -s /proc/self/fd/0 "$M/dev/stdin"
ln -s /proc/self/fd/1 "$M/dev/stdout"
ln -s /proc/self/fd/2 "$M/dev/stderr"
ln -s /proc/kcore "$M/dev/core" 2>/dev/null || true

if ! mount -t sysfs -o nosuid,nodev,noexec,ro sysfs "$M/sys" 2>/dev/null; then
  mount --rbind /sys "$M/sys" || fail "sys"
fi

# /etc/resolv.conf and /etc/hosts: overwrite in the overlay (copy, not bind,
# so the sandbox may edit them like in Docker).
if [ -n "${NS_ETC_DIR:-}" ]; then
  for f in resolv.conf hosts hostname; do
    if [ -f "$NS_ETC_DIR/$f" ]; then
      rm -f "$M/etc/$f" 2>/dev/null || true
      cp "$NS_ETC_DIR/$f" "$M/etc/$f"
    fi
  done
fi

if [ -n "${NS_BINDS:-}" ]; then
  while IFS=$'\t' read -r src dst; do
    [ -z "$src" ] && continue
    if [ -d "$src" ]; then mkdir -p "$M$dst"; else mkdir -p "$(dirname "$M$dst")"; touch "$M$dst"; fi
    mount --bind "$src" "$M$dst" || fail "bind $src -> $dst"
  done <<< "$NS_BINDS"
fi

hostname "${NS_HOSTNAME:-sandbox}" || true
if [ "${NS_PRIVATE_NET:-0}" = 1 ]; then ip link set lo up || fail "lo up"; fi

# nsinit must be reachable after pivot_root: copy it into the private /dev tmpfs.
cp "$NS_NSINIT" "$M/dev/.nsinit"
chmod 755 "$M/dev/.nsinit"
exec "$M/dev/.nsinit" "$M" "$@"
