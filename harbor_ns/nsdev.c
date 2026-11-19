/*
 * nsdev.so: the preload of subordinate-id pods (HARBOR_NS_SUBUID, see subids.py).
 *
 * Those pods have real uids, so nsfake's id emulation is off; one thing a rootless
 * sandbox still cannot do is mknod(2) a device node: that needs CAP_MKNOD in the
 * initial user namespace.  Verifiers that build a chroot jail for agent code create
 * /dev/null, /dev/urandom, ... in it.  When mknod of a character/block device fails
 * with EPERM, this shim creates an empty file and bind-mounts the sandbox's own
 * device node with the same st_rdev onto it (the sandbox's /dev holds bind mounts of
 * the host's null, zero, full, random, urandom, tty).  unlink/unlinkat of such a file
 * fails with EBUSY while it is a mount point, so they detach the mount and retry.
 *
 * Also carried over from nsfake (id-independent): sched_setaffinity onto CPUs outside
 * the box's cpuset is remapped instead of failing with EINVAL (`taskset -c 0`), and
 * raising RLIMIT_CORE above the host hard limit is clamped.  chmod of an emulated device
 * node to the permissions it already has succeeds (see the chmod section).
 *
 * Raw syscalls only (no dlsym), like nsfake.
 * Build: gcc -shared -fPIC -O2 -o nsdev.so nsdev.c
 */
#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <sched.h>
#include <sys/mount.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

static int find_dev(dev_t rdev, int blk, char *out, size_t n) {
    DIR *d = opendir("/dev");
    if (!d) return 0;
    struct dirent *e;
    int found = 0;
    while (!found && (e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.') continue;
        char p[PATH_MAX];
        snprintf(p, sizeof p, "/dev/%s", e->d_name);
        struct stat st;
        /* lstat: /dev/stdin etc. are symlinks into /proc/self/fd and cannot be bind sources */
        if (syscall(SYS_newfstatat, AT_FDCWD, p, &st, AT_SYMLINK_NOFOLLOW) != 0) continue;
        if ((blk ? S_ISBLK(st.st_mode) : S_ISCHR(st.st_mode)) && st.st_rdev == rdev) {
            snprintf(out, n, "%s", p);
            found = 1;
        }
    }
    closedir(d);
    return found;
}

static const char *at_path(int dirfd, const char *path, char *buf, size_t n) {
    if (!path || path[0] == '/' || dirfd == AT_FDCWD) return path;
    snprintf(buf, n, "/proc/self/fd/%d/%s", dirfd, path);
    return buf;
}

int mknodat(int dirfd, const char *path, mode_t mode, dev_t dev) {
    long r = syscall(SYS_mknodat, dirfd, path, mode, dev);
    if (r == 0 || errno != EPERM || !(S_ISCHR(mode) || S_ISBLK(mode))) return (int)r;
    char src[PATH_MAX], buf[PATH_MAX];
    if (!find_dev(dev, S_ISBLK(mode), src, sizeof src)) {
        dprintf(2, "nsdev: mknod %s: no device with rdev %u:%u in /dev\n", path, major(dev), minor(dev));
        errno = EPERM;
        return -1;
    }
    int fd = (int)syscall(SYS_openat, dirfd, path, O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, mode & 07777);
    if (fd < 0) return -1;
    syscall(SYS_close, fd);
    const char *tgt = at_path(dirfd, path, buf, sizeof buf);
    if (syscall(SYS_mount, src, tgt, NULL, MS_BIND, NULL) != 0) {
        int e = errno;
        dprintf(2, "nsdev: mknod %s: bind mount of %s failed: %s\n", path, src, strerror(e));
        syscall(SYS_unlinkat, dirfd, path, 0);
        errno = e == EPERM ? EPERM : e;
        return -1;
    }
    return 0;
}

int mknod(const char *path, mode_t mode, dev_t dev) { return mknodat(AT_FDCWD, path, mode, dev); }
int __xmknod(int ver, const char *path, mode_t mode, dev_t *dev) {
    (void)ver;
    return mknodat(AT_FDCWD, path, mode, *dev);
}
int __xmknodat(int ver, int dirfd, const char *path, mode_t mode, dev_t *dev) {
    (void)ver;
    return mknodat(dirfd, path, mode, *dev);
}

int unlinkat(int dirfd, const char *path, int flags) {
    long r = syscall(SYS_unlinkat, dirfd, path, flags);
    if (r == 0 || errno != EBUSY || (flags & AT_REMOVEDIR)) return (int)r;
    char buf[PATH_MAX];
    if (syscall(SYS_umount2, at_path(dirfd, path, buf, sizeof buf), MNT_DETACH) != 0) {
        errno = EBUSY;
        return -1;
    }
    return (int)syscall(SYS_unlinkat, dirfd, path, flags);
}

int unlink(const char *path) { return unlinkat(AT_FDCWD, path, 0); }

/* ---------------------------------------------------------------- CPU affinity (as nsfake.c) */
static unsigned int cpu_offset(void) {
    char buf[64];
    long n = syscall(SYS_readlinkat, AT_FDCWD, "/proc/self/ns/pid", buf, sizeof buf - 1);
    if (n <= 0) return 0;
    buf[n] = 0;
    unsigned long ino = 0;
    for (char *p = buf; *p; p++)
        if (*p >= '0' && *p <= '9') ino = ino * 10 + (unsigned long)(*p - '0');
    return (unsigned int)(ino * 2654435761UL >> 7);
}

int sched_setaffinity(pid_t pid, size_t sz, const cpu_set_t *set) {
    long r = syscall(SYS_sched_setaffinity, pid, sz, set);
    if (r == 0 || errno != EINVAL || !set) return (int)r;
    cpu_set_t allowed, want;
    CPU_ZERO(&allowed);
    if (syscall(SYS_sched_getaffinity, 0, sizeof allowed, &allowed) < 0) { errno = EINVAL; return -1; }
    int map[CPU_SETSIZE], n = 0;
    for (int c = 0; c < CPU_SETSIZE; c++) if (CPU_ISSET(c, &allowed)) map[n++] = c;
    if (!n) { errno = EINVAL; return -1; }
    unsigned int off = cpu_offset();
    CPU_ZERO(&want);
    int any = 0;
    for (size_t c = 0; c < sz * 8 && c < CPU_SETSIZE; c++)
        if (CPU_ISSET_S(c, sz, set)) { CPU_SET(map[(c + off) % (unsigned int)n], &want); any = 1; }
    if (!any) { errno = EINVAL; return -1; }
    return (int)syscall(SYS_sched_setaffinity, pid, sizeof want, &want);
}

/* ---------------------------------------------------------------- RLIMIT_CORE (as nsfake.c) */
static int prlimit_core_clamp(pid_t pid, int res, const struct rlimit *nl, struct rlimit *ol) {
    long r = syscall(SYS_prlimit64, pid, res, nl, ol);
    if (r == 0 || errno != EPERM || res != RLIMIT_CORE || !nl) return (int)r;
    int saved = errno;
    struct rlimit cur;
    if (syscall(SYS_prlimit64, pid, res, NULL, &cur) != 0) { errno = saved; return -1; }
    struct rlimit c = *nl;
    if (c.rlim_max > cur.rlim_max) c.rlim_max = cur.rlim_max;
    if (c.rlim_cur > c.rlim_max) c.rlim_cur = c.rlim_max;
    if (syscall(SYS_prlimit64, pid, res, &c, ol) != 0) { errno = saved; return -1; }
    return 0;
}
int setrlimit(__rlimit_resource_t res, const struct rlimit *rl) { return prlimit_core_clamp(0, res, rl, NULL); }
int setrlimit64(__rlimit_resource_t res, const struct rlimit64 *rl) {
    return prlimit_core_clamp(0, res, (const struct rlimit *)rl, NULL);
}
int prlimit(pid_t pid, __rlimit_resource_t res, const struct rlimit *nl, struct rlimit *ol) {
    return prlimit_core_clamp(pid, res, nl, ol);
}
int prlimit64(pid_t pid, __rlimit_resource_t res, const struct rlimit64 *nl, struct rlimit64 *ol) {
    return prlimit_core_clamp(pid, res, (const struct rlimit *)nl, (struct rlimit *)ol);
}

/* ---------------------------------------------------------------- chmod of emulated device nodes */
/* An emulated node is a bind mount of the host's device, whose owner is not mapped in the
 * namespace, so chmod fails with EPERM.  Docker's mknod made a fresh root-owned node that
 * the jail builder may chmod (typically to the 0666 the host node already has): succeed
 * when the requested permission bits are already in place, fail as before otherwise. */
static int dev_mode_matches(int r_errno, const struct stat *st, mode_t mode) {
    return r_errno == EPERM && (S_ISCHR(st->st_mode) || S_ISBLK(st->st_mode)) &&
           (st->st_mode & 07777) == (mode & 07777);
}

int fchmodat(int dirfd, const char *path, mode_t mode, int flags) {
    long r = syscall(SYS_fchmodat, dirfd, path, mode);
    if (r == 0 || errno != EPERM) return (int)r;
    int e = errno;
    struct stat st;
    if (syscall(SYS_newfstatat, dirfd, path, &st, flags & AT_SYMLINK_NOFOLLOW) == 0 && dev_mode_matches(e, &st, mode))
        return 0;
    errno = e;
    return -1;
}

int chmod(const char *path, mode_t mode) { return fchmodat(AT_FDCWD, path, mode, 0); }

int fchmod(int fd, mode_t mode) {
    long r = syscall(SYS_fchmod, fd, mode);
    if (r == 0 || errno != EPERM) return (int)r;
    int e = errno;
    struct stat st;
    if (syscall(SYS_fstat, fd, &st) == 0 && dev_mode_matches(e, &st, mode)) return 0;
    errno = e;
    return -1;
}

/* ---------------------------------------------------------------- CPU count (HARBOR_NS_CPU_VIEW) */
/* A box pinned to N CPUs (taskset) still sees every node CPU in sysfs, so glibc's
 * get_nprocs / sysconf(_SC_NPROCESSORS_*) (Python os.cpu_count, libstdc++
 * hardware_concurrency) report the whole node and thread pools oversubscribe the box.
 * When the sandbox marks the CPU view on (/usr/lib/harbor-ns/cpu_view), report the
 * affinity size instead; sysfs itself stays real (hwloc/Open MPI read it). */
extern long __sysconf(int);
static int cpu_view_on(void) {
    static int on = -1;
    if (on < 0) {
        struct stat st;
        on = syscall(SYS_newfstatat, AT_FDCWD, "/usr/lib/harbor-ns/cpu_view", &st, 0) == 0;
    }
    return on;
}
static int affinity_cpus(void) {
    cpu_set_t s;
    CPU_ZERO(&s);
    if (syscall(SYS_sched_getaffinity, 0, sizeof s, &s) < 0) return 0;
    return CPU_COUNT(&s);
}
long sysconf(int name) {
    if ((name == _SC_NPROCESSORS_ONLN || name == _SC_NPROCESSORS_CONF) && cpu_view_on()) {
        int n = affinity_cpus();
        if (n > 0) return n;
    }
    return __sysconf(name);
}
int get_nprocs(void) { return (int)sysconf(_SC_NPROCESSORS_ONLN); }
int get_nprocs_conf(void) { return (int)sysconf(_SC_NPROCESSORS_CONF); }

