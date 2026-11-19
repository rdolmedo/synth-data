/*
 * nsfake.so: emulate multiple users inside a harbor_ns sandbox.
 *
 * The sandbox is an unprivileged user namespace that maps only uid/gid 0 (the
 * cluster gives us no subuid range), so any operation that needs another id
 * fails with EINVAL/EPERM: chown to a service user, setgroups, setuid(www-data)
 * in nginx/su/apt, PostgreSQL refusing to run as root, ... .  Under Docker these
 * work, and task environments/tests rely on them.  This preload (installed via
 * /etc/ld.so.preload in glibc images only) emulates them, fakeroot-style:
 *
 *  - set*uid/set*gid/setgroups follow the kernel's rules on emulated ids: an
 *    emulated root may switch to any id, an emulated unprivileged process only
 *    within its real/effective/saved set.  The ids are kept per process and
 *    inherited across exec through NSFAKE_ID in the environment (re-inserted
 *    by the exec hooks when a program builds its own environment);
 *    get*uid/get*gid/getgroups report them.  Everything still really runs as
 *    namespace root.
 *  - exec of a set-user/group-ID file by an emulated user switches the
 *    effective+saved id to the file's owner, so sudo/su/passwd work as under
 *    Docker for a non-root image USER.
 *  - chown/lchown/fchown/fchownat "succeed" and record the owner in the
 *    user.harborns.owner xattr; stat/lstat/fstat/fstatat/statx report it.  A
 *    process running fully under an emulated uid (ruid == euid) sees otherwise
 *    root-owned files as its own (so e.g. postgres' data-directory ownership
 *    check passes); a root process that only raised its euid (sudo reads
 *    sudoers with euid 1) still sees them as root's.
 *  - getsockopt(SO_PEERCRED) reports the peer's emulated ids (read from the
 *    peer's /proc/<pid>/environ), so unix-socket peer auth works.
 *  - sched_setaffinity onto CPUs outside the condor slot's cpuset (`taskset -c 0`)
 *    is mapped onto the slot's CPUs instead of failing with EINVAL.
 *  - raising RLIMIT_CORE above the (host-imposed) hard limit is clamped.
 *  - AddressSanitizer binaries are exec'd with verify_asan_link_order=0 and
 *    without ASLR (see the exec section).
 *
 * The "real" operations are done with raw syscalls (no dlsym/libdl), so the
 * shim loads in any x86_64 glibc image (dlsym moved from libdl into libc in
 * 2.34; a versioned dlsym reference fails on one side or the other).
 * On x86_64 the kernel's struct stat is glibc's struct stat/stat64.
 *
 * Build: gcc -shared -fPIC -O2 -fno-delete-null-pointer-checks -U_FORTIFY_SOURCE \
 *            -fno-stack-protector -o nsfake.so nsfake.c
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <grp.h>
#include <sched.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/personality.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/xattr.h>
#include <unistd.h>

#define XATTR "user.harborns.owner"
extern char **environ;

/* ruid euid suid rgid egid sgid; 0 = real root */
static unsigned int ids[6];
/* NSFAKE_ID this process image started with ("" if none): a root process that
 * exec's with a *different* NSFAKE_ID (the gosu shim, `env NSFAKE_ID=...`) asks
 * for that identity; the one it inherited is stale once it switched back to root. */
static char initial_id[96];

static int read_owner(int how, int fd, const char *path, unsigned int *u, unsigned int *g);
static void save_ids(void);

static int ids_nonroot(void) {
    int any = 0;
    for (int i = 0; i < 6; i++) any |= ids[i] != 0;
    return any;
}

static void __attribute__((constructor)) nsfake_init(void) {
    const char *e = getenv("NSFAKE_ID");
    if (e) {
        snprintf(initial_id, sizeof initial_id, "%s", e);
        sscanf(e, "%u:%u:%u:%u:%u:%u", &ids[0], &ids[1], &ids[2], &ids[3], &ids[4], &ids[5]);
    }
    if (!ids_nonroot()) return;
    int saved = errno;
    /* execve copies the effective ids to the saved ones ... */
    ids[2] = ids[1];
    ids[5] = ids[4];
    /* ... after a set-user/group-ID program (sudo, su, passwd, ...) switched the
     * effective ids to the file's owner/group. */
    struct stat st;
    if (syscall(SYS_newfstatat, AT_FDCWD, "/proc/self/exe", &st, 0) == 0 &&
        (st.st_mode & (S_ISUID | S_ISGID))) {
        unsigned int u = st.st_uid, g = st.st_gid;
        read_owner(0, -1, "/proc/self/exe", &u, &g);
        if (st.st_mode & S_ISUID) ids[1] = ids[2] = u;
        if ((st.st_mode & (S_ISGID | S_IXGRP)) == (S_ISGID | S_IXGRP)) ids[4] = ids[5] = g;
    }
    save_ids();
    errno = saved;
}

static void save_ids(void) {
    if (!ids_nonroot()) {
        unsetenv("NSFAKE_ID");
        return;
    }
    char buf[96];
    snprintf(buf, sizeof buf, "%u:%u:%u:%u:%u:%u", ids[0], ids[1], ids[2], ids[3], ids[4], ids[5]);
    setenv("NSFAKE_ID", buf, 1);
}

/* clearenv() (su -l, sudo) would drop the emulated identity before exec. */
int clearenv(void) {
    environ = NULL;
    save_ids();
    return 0;
}

/* ---------------------------------------------------------------- identity */
uid_t getuid(void) { return ids[0]; }
uid_t geteuid(void) { return ids[1]; }
gid_t getgid(void) { return ids[3]; }
gid_t getegid(void) { return ids[4]; }
int getresuid(uid_t *r, uid_t *e, uid_t *s) { *r = ids[0]; *e = ids[1]; *s = ids[2]; return 0; }
int getresgid(gid_t *r, gid_t *e, gid_t *s) { *r = ids[3]; *e = ids[4]; *s = ids[5]; return 0; }

int getgroups(int size, gid_t list[]) {
    if (ids[1] == 0 && ids[4] == 0) return (int)syscall(SYS_getgroups, size, list);
    if (size == 0) return 1;
    if (size < 1) { errno = EINVAL; return -1; }
    list[0] = ids[4];
    return 1;
}

#define KEEP(x) ((x) == (unsigned int)-1)
/* Kernel permission rules for an emulated unprivileged process (emulated euid != 0,
 * i.e. no CAP_SETUID/CAP_SETGID): ids may only move within the current
 * real/effective/saved set.  OpenSSH's permanently_set_uid() verifies that a dropped
 * process can NOT regain its old ids ("was able to restore old [e]gid" -> every
 * connection reset), and other privilege-dropping code checks the same. */
#define PRIV() (ids[1] == 0)
#define IN_U(x) ((x) == ids[0] || (x) == ids[1] || (x) == ids[2])
#define IN_G(x) ((x) == ids[3] || (x) == ids[4] || (x) == ids[5])
#define DENY() do { errno = EPERM; return -1; } while (0)
int setuid(uid_t u) {
    if (PRIV()) ids[0] = ids[1] = ids[2] = u;
    else if (u == ids[0] || u == ids[2]) ids[1] = u;
    else DENY();
    save_ids();
    return 0;
}
int setgid(gid_t g) {
    if (PRIV()) ids[3] = ids[4] = ids[5] = g;
    else if (g == ids[3] || g == ids[5]) ids[4] = g;
    else DENY();
    save_ids();
    return 0;
}
int seteuid(uid_t u) {
    if (!KEEP(u) && !PRIV() && !IN_U(u)) DENY();
    if (!KEEP(u)) ids[1] = u;
    save_ids();
    return 0;
}
int setegid(gid_t g) {
    if (!KEEP(g) && !PRIV() && !IN_G(g)) DENY();
    if (!KEEP(g)) ids[4] = g;
    save_ids();
    return 0;
}
int setreuid(uid_t r, uid_t e) {
    if (!PRIV() && ((!KEEP(r) && r != ids[0] && r != ids[1]) || (!KEEP(e) && !IN_U(e)))) DENY();
    unsigned int old_r = ids[0];
    if (!KEEP(r)) ids[0] = r;
    if (!KEEP(e)) ids[1] = e;
    if (!KEEP(r) || (!KEEP(e) && e != old_r)) ids[2] = ids[1];
    save_ids();
    return 0;
}
int setregid(gid_t r, gid_t e) {
    if (!PRIV() && ((!KEEP(r) && r != ids[3] && r != ids[4]) || (!KEEP(e) && !IN_G(e)))) DENY();
    unsigned int old_r = ids[3];
    if (!KEEP(r)) ids[3] = r;
    if (!KEEP(e)) ids[4] = e;
    if (!KEEP(r) || (!KEEP(e) && e != old_r)) ids[5] = ids[4];
    save_ids();
    return 0;
}
int setresuid(uid_t r, uid_t e, uid_t s) {
    if (!PRIV() && ((!KEEP(r) && !IN_U(r)) || (!KEEP(e) && !IN_U(e)) || (!KEEP(s) && !IN_U(s)))) DENY();
    if (!KEEP(r)) ids[0] = r;
    if (!KEEP(e)) ids[1] = e;
    if (!KEEP(s)) ids[2] = s;
    save_ids();
    return 0;
}
int setresgid(gid_t r, gid_t e, gid_t s) {
    if (!PRIV() && ((!KEEP(r) && !IN_G(r)) || (!KEEP(e) && !IN_G(e)) || (!KEEP(s) && !IN_G(s)))) DENY();
    if (!KEEP(r)) ids[3] = r;
    if (!KEEP(e)) ids[4] = e;
    if (!KEEP(s)) ids[5] = s;
    save_ids();
    return 0;
}
int setgroups(size_t n, const gid_t *l) { (void)n; (void)l; if (!PRIV()) DENY(); return 0; }
int initgroups(const char *u, gid_t g) { (void)u; (void)g; if (!PRIV()) DENY(); return 0; }

/* ---------------------------------------------------------------- ownership */
static int read_owner(int how, int fd, const char *path, unsigned int *u, unsigned int *g) {
    char buf[64];
    ssize_t n;
    if (how == 0) n = getxattr(path, XATTR, buf, sizeof buf - 1);
    else if (how == 1) n = lgetxattr(path, XATTR, buf, sizeof buf - 1);
    else n = fgetxattr(fd, XATTR, buf, sizeof buf - 1);
    if (n <= 0) return 0;
    buf[n] = 0;
    return sscanf(buf, "%u:%u", u, g) == 2;
}

static void write_owner(int how, int fd, const char *path, uid_t u, gid_t g) {
    unsigned int cu = 0, cg = 0;
    int saved = errno;
    read_owner(how, fd, path, &cu, &cg);
    if (!KEEP(u)) cu = u;
    if (!KEEP(g)) cg = g;
    if (cu == 0 && cg == 0) {
        if (how == 0) removexattr(path, XATTR);
        else if (how == 1) lremovexattr(path, XATTR);
        else fremovexattr(fd, XATTR);
    } else {
        char buf[64];
        int len = snprintf(buf, sizeof buf, "%u:%u", cu, cg);
        if (how == 0) setxattr(path, XATTR, buf, len, 0);
        else if (how == 1) lsetxattr(path, XATTR, buf, len, 0);
        else fsetxattr(fd, XATTR, buf, len, 0);
    }
    errno = saved;
}

/* Path usable with the *xattr calls for (dirfd, path). */
static const char *at_path(int dirfd, const char *path, char *buf, size_t sz) {
    if (!path || path[0] == '/' || dirfd == AT_FDCWD) return path;
    snprintf(buf, sz, "/proc/self/fd/%d/%s", dirfd, path);
    return buf;
}

static int chown_common(int how, int fd, const char *path, uid_t u, gid_t g) {
    struct stat st;
    int r = how == 2 ? fstat(fd, &st) : (how == 1 ? lstat(path, &st) : stat(path, &st));
    if (r != 0) return -1;
    if (how == 1 && S_ISLNK(st.st_mode)) return 0; /* no user xattrs on symlinks */
    write_owner(how, fd, path, u, g);
    return 0;
}

int chown(const char *p, uid_t u, gid_t g) { return chown_common(0, -1, p, u, g); }
int lchown(const char *p, uid_t u, gid_t g) { return chown_common(1, -1, p, u, g); }
int fchown(int fd, uid_t u, gid_t g) { return chown_common(2, fd, NULL, u, g); }
int fchownat(int d, const char *p, uid_t u, gid_t g, int f) {
    char buf[4096];
    if ((f & AT_EMPTY_PATH) && (!p || !*p)) return chown_common(2, d, NULL, u, g);
    return chown_common((f & AT_SYMLINK_NOFOLLOW) ? 1 : 0, -1, at_path(d, p, buf, sizeof buf), u, g);
}

/* Files created while only the *effective* ids are emulated (ruid != euid: postfix's
 * local(8) delivering as the recipient via seteuid, set-uid helpers) belong to the
 * effective user, as the kernel would make them.  Record that owner, because such a
 * process sees unrecorded (really root-owned) files as root's (see fix_ids).  A fully
 * switched identity keeps the older behaviour: it sees root-owned files as its own. */
static int raised_only(void) {
    return (ids[1] != 0 && ids[0] != ids[1]) || (ids[4] != 0 && ids[3] != ids[4]);
}

static void record_created(int fd) {
    if (fd >= 0) write_owner(2, fd, NULL, ids[1], ids[4]);
}

static int open_common(int dirfd, const char *path, int flags, mode_t mode) {
    int creat = (flags & O_CREAT) && (flags & O_TMPFILE) != O_TMPFILE;
    if (!creat || !raised_only())
        return (int)syscall(SYS_openat, dirfd, path, flags, mode);
    int fd;
    if (!(flags & O_EXCL)) {
        fd = (int)syscall(SYS_openat, dirfd, path, flags & ~O_CREAT, mode);
        if (fd >= 0 || errno != ENOENT) return fd;
    }
    fd = (int)syscall(SYS_openat, dirfd, path, flags | O_EXCL, mode);
    if (fd >= 0) {
        int saved = errno;
        record_created(fd);
        errno = saved;
    } else if (errno == EEXIST && !(flags & O_EXCL)) {
        fd = (int)syscall(SYS_openat, dirfd, path, flags, mode);  /* raced, or a dangling symlink */
    }
    return fd;
}

#define OPEN_MODE(flags, last) mode_t mode = 0; \
    if (((flags) & O_CREAT) || ((flags) & O_TMPFILE) == O_TMPFILE) { \
        va_list ap; va_start(ap, last); mode = (mode_t)va_arg(ap, int); va_end(ap); }
int open(const char *p, int flags, ...) { OPEN_MODE(flags, flags) return open_common(AT_FDCWD, p, flags, mode); }
int open64(const char *p, int flags, ...) { OPEN_MODE(flags, flags) return open_common(AT_FDCWD, p, flags, mode); }
int openat(int d, const char *p, int flags, ...) { OPEN_MODE(flags, flags) return open_common(d, p, flags, mode); }
int openat64(int d, const char *p, int flags, ...) { OPEN_MODE(flags, flags) return open_common(d, p, flags, mode); }
int creat(const char *p, mode_t mode) { return open_common(AT_FDCWD, p, O_CREAT | O_WRONLY | O_TRUNC, mode); }
int creat64(const char *p, mode_t mode) { return open_common(AT_FDCWD, p, O_CREAT | O_WRONLY | O_TRUNC, mode); }

int mkdirat(int d, const char *p, mode_t mode) {
    int r = (int)syscall(SYS_mkdirat, d, p, mode);
    if (r == 0 && raised_only()) {
        int saved = errno;
        int fd = (int)syscall(SYS_openat, d, p, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC, 0);
        if (fd >= 0) { record_created(fd); syscall(SYS_close, fd); }
        errno = saved;
    }
    return r;
}
int mkdir(const char *p, mode_t mode) { return mkdirat(AT_FDCWD, p, mode); }

static void fix_ids(int how, int fd, const char *path, unsigned int *su, unsigned int *sg) {
    unsigned int u, g;
    int saved = errno;
    if (read_owner(how, fd, path, &u, &g)) {
        *su = u;
        *sg = g;
    } else {
        /* Under an emulated uid, root-owned files look like the caller's own.  Only for a
         * fully switched identity (ruid == euid): a root process that temporarily raises a
         * different euid (sudo reads sudoers with euid 1) must still see root:root. */
        if (*su == 0 && ids[1] != 0 && ids[0] == ids[1]) *su = ids[1];
        if (*sg == 0 && ids[4] != 0 && ids[3] == ids[4]) *sg = ids[4];
    }
    errno = saved;
}

#define FIX_ST(how, fd, path, st) do { unsigned int _u = (st)->st_uid, _g = (st)->st_gid; \
    fix_ids(how, fd, path, &_u, &_g); (st)->st_uid = _u; (st)->st_gid = _g; } while (0)

#define K_STAT(p, st) syscall(SYS_newfstatat, AT_FDCWD, (p), (st), 0)
#define K_LSTAT(p, st) syscall(SYS_newfstatat, AT_FDCWD, (p), (st), AT_SYMLINK_NOFOLLOW)
#define K_FSTAT(fd, st) syscall(SYS_fstat, (fd), (st))
#define K_FSTATAT(d, p, st, f) syscall(SYS_newfstatat, (d), (p), (st), (f))

static int do_fstatat(int d, const char *p, struct stat *st, int f) {
    int r = (int)K_FSTATAT(d, p, st, f);
    if (!r) {
        char buf[4096];
        if ((f & AT_EMPTY_PATH) && (!p || !*p)) FIX_ST(2, d, NULL, st);
        else FIX_ST((f & AT_SYMLINK_NOFOLLOW) ? 1 : 0, -1, at_path(d, p, buf, sizeof buf), st);
    }
    return r;
}

/* glibc >= 2.33 exports these; older glibc routes through __xstat & co (below). */
int stat(const char *p, struct stat *st) { int r = (int)K_STAT(p, st); if (!r) FIX_ST(0, -1, p, st); return r; }
int lstat(const char *p, struct stat *st) { int r = (int)K_LSTAT(p, st); if (!r) FIX_ST(1, -1, p, st); return r; }
int fstat(int fd, struct stat *st) { int r = (int)K_FSTAT(fd, st); if (!r) FIX_ST(2, fd, NULL, st); return r; }
int fstatat(int d, const char *p, struct stat *st, int f) { return do_fstatat(d, p, st, f); }
int stat64(const char *p, struct stat64 *st) { return stat(p, (struct stat *)st); }
int lstat64(const char *p, struct stat64 *st) { return lstat(p, (struct stat *)st); }
int fstat64(int fd, struct stat64 *st) { return fstat(fd, (struct stat *)st); }
int fstatat64(int d, const char *p, struct stat64 *st, int f) { return do_fstatat(d, p, (struct stat *)st, f); }

int statx(int d, const char *p, int f, unsigned int mask, struct statx *sx) {
    int r = (int)syscall(SYS_statx, d, p, f, mask, sx);
    if (!r) {
        char buf[4096];
        unsigned int u = sx->stx_uid, g = sx->stx_gid;
        if ((f & AT_EMPTY_PATH) && (!p || !*p)) fix_ids(2, d, NULL, &u, &g);
        else fix_ids((f & AT_SYMLINK_NOFOLLOW) ? 1 : 0, -1, at_path(d, p, buf, sizeof buf), &u, &g);
        sx->stx_uid = u;
        sx->stx_gid = g;
    }
    return r;
}

/* pre-2.33 glibc entry points (x86_64: _STAT_VER is 1, kernel layout) */
int __xstat(int v, const char *p, struct stat *st) { (void)v; return stat(p, st); }
int __lxstat(int v, const char *p, struct stat *st) { (void)v; return lstat(p, st); }
int __fxstat(int v, int fd, struct stat *st) { (void)v; return fstat(fd, st); }
int __fxstatat(int v, int d, const char *p, struct stat *st, int f) { (void)v; return do_fstatat(d, p, st, f); }
int __xstat64(int v, const char *p, struct stat64 *st) { (void)v; return stat(p, (struct stat *)st); }
int __lxstat64(int v, const char *p, struct stat64 *st) { (void)v; return lstat(p, (struct stat *)st); }
int __fxstat64(int v, int fd, struct stat64 *st) { (void)v; return fstat(fd, (struct stat *)st); }
int __fxstatat64(int v, int d, const char *p, struct stat64 *st, int f) { (void)v; return do_fstatat(d, p, (struct stat *)st, f); }

/* ---------------------------------------------------------------- peer credentials */
static int peer_ids(pid_t pid, unsigned int *u, unsigned int *g) {
    char path[64], buf[65536];
    snprintf(path, sizeof path, "/proc/%d/environ", (int)pid);
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return 0;
    ssize_t n = read(fd, buf, sizeof buf - 1);
    close(fd);
    if (n <= 0) return 0;
    buf[n] = 0;
    for (char *p = buf; p < buf + n; p += strlen(p) + 1) {
        unsigned int v[6];
        if (strncmp(p, "NSFAKE_ID=", 10) == 0 &&
            sscanf(p + 10, "%u:%u:%u:%u:%u:%u", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) == 6) {
            *u = v[1];
            *g = v[4];
            return 1;
        }
    }
    return 0;
}

int getsockopt(int fd, int level, int name, void *val, socklen_t *len) {
    int r = (int)syscall(SYS_getsockopt, fd, level, name, val, len);
    if (r == 0 && level == SOL_SOCKET && name == SO_PEERCRED && val && len && *len >= sizeof(struct ucred)) {
        struct ucred *c = (struct ucred *)val;
        unsigned int u, g;
        int saved = errno;
        if (c->uid == 0 && c->pid > 0 && peer_ids(c->pid, &u, &g)) {
            c->uid = u;
            c->gid = g;
        }
        errno = saved;
    }
    return r;
}

/* ---------------------------------------------------------------- CPU affinity */
/* Docker containers normally see every host CPU, so tasks pin with `taskset -c 0`
 * (benchmarks) and expect it to work.  A condor slot only owns a cpuset slice
 * that may not contain CPU 0 -> EINVAL.  If the request fails with EINVAL, number
 * the slot's CPUs 0..n-1 and map requested CPU k onto slot CPU (k + off) mod n.
 * `off` is derived from the sandbox's pid namespace, so concurrent sandboxes that
 * all ask for "CPU 0" spread over the slot instead of piling onto one core. */
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

/* ---------------------------------------------------------------- resource limits */
/* Docker containers usually get RLIMIT_CORE=unlimited (dockerd's LimitCORE=infinity);
 * here the host/condor hard limit is often 0 and only init-userns CAP_SYS_RESOURCE could
 * raise it, so `ulimit -c unlimited` fails with EPERM (and kills `set -e` scripts).
 * Raising RLIMIT_CORE beyond the hard limit is clamped to the hard limit and reported
 * as success (no core files are written, as with a 0-size limit). */
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

/* ---------------------------------------------------------------- exec */
/* The emulated identity travels in NSFAKE_ID, but programs may exec with an
 * environment they built themselves: sshd's session child, `env -i`, Perl's
 * `%ENV = (...)` (perl <= 5.36 rewrites environ behind libc; Debian's
 * pg_ctlcluster does that before exec'ing pg_ctl, which then refused to "run as
 * root"), subprocess env=... .  Real credentials survive exec, so:
 *  - a process under an emulated non-root identity always passes its current
 *    ids (replacing any copy in envp);
 *  - a root process passes an NSFAKE_ID through only if it differs from the one
 *    it started with (the gosu shim / `env NSFAKE_ID=...` switching identity);
 *    an inherited copy is stale (e.g. `sudo -E` after switching to root).
 *
 * AddressSanitizer binaries.  (1) The /etc/ld.so.preload shim loads before libasan,
 * which then aborts ("ASan runtime does not come first in initial library list")
 * unless ASAN_OPTIONS has verify_asan_link_order=0.  (2) With the cluster kernels'
 * vm.mmap_rnd_bits=32, gcc<=13 libasan randomly aborts with "Shadow memory range
 * interleaves with an existing memory mapping" (also under Docker on these hosts);
 * newer compiler-rt re-execs itself without ASLR in that case, so do that up front.
 *
 * glibc's execv*()/execl*() call its internal __execve, which a preload cannot
 * interpose, so they are reimplemented on top of execve() (PATH search and the
 * ENOEXEC -> /bin/sh fallback as in glibc).  All of this may run in a vfork()ed
 * child sharing memory with a multi-threaded parent (CPython's subprocess,
 * posix_spawn users): stack buffers only, no malloc, no writes to globals. */
static int is_asan_binary(int dirfd, const char *path, int flags) {
    char buf[8192 + 16];
    int fd, own = 1;
    if ((flags & AT_EMPTY_PATH) && (!path || !*path)) { fd = dirfd; own = 0; }
    else if (!path) return 0;
    else fd = (int)syscall(SYS_openat, dirfd, path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return 0;
    int found = 0;
    size_t keep = 0;
    long off = 0;
    while (off < 131072) {
        long r = syscall(SYS_pread64, fd, buf + keep, 8192, off);
        if (r <= 0) break;
        if (off == 0 && (r < 64 || memcmp(buf, "\x7f" "ELF", 4) != 0)) break;
        size_t len = keep + (size_t)r;
        if (memmem(buf, len, "libasan.so", 10) || memmem(buf, len, "__asan_init", 11)) { found = 1; break; }
        keep = len < 15 ? len : 15;
        memmove(buf, buf + len - keep, keep);
        off += r;
    }
    if (own) syscall(SYS_close, fd);
    return found;
}

static long exec_common(int dirfd, const char *path, char *const argv[], char *const envp[], int flags, int at) {
    int any = ids_nonroot();
    size_t n = 0;
    const char *envid = NULL;
    if (envp)
        for (; envp[n]; n++)
            if (strncmp(envp[n], "NSFAKE_ID=", 10) == 0) envid = envp[n] + 10;
    /* root process asking for a new identity: pass envp as is */
    int pass_id = !any && envid && strcmp(envid, initial_id) != 0;
    int asan = is_asan_binary(at ? dirfd : AT_FDCWD, path, at ? flags : 0);
    if (!asan && !any && (!envid || pass_id))
        return at ? syscall(SYS_execveat, dirfd, path, argv, envp, flags) : syscall(SYS_execve, path, argv, envp);
    char *ne[n + 3];
    size_t k = 0;
    char *asan_opts = NULL;
    for (size_t i = 0; i < n; i++) {
        if (!pass_id && strncmp(envp[i], "NSFAKE_ID=", 10) == 0) continue;
        if (asan && strncmp(envp[i], "ASAN_OPTIONS=", 13) == 0) { asan_opts = envp[i]; continue; }
        ne[k++] = envp[i];
    }
    char idbuf[112];
    if (any) {
        snprintf(idbuf, sizeof idbuf, "NSFAKE_ID=%u:%u:%u:%u:%u:%u", ids[0], ids[1], ids[2], ids[3], ids[4], ids[5]);
        ne[k++] = idbuf;
    }
    char aobuf[4096];
    long pers = -1;
    if (asan) {
        const char *cur = asan_opts ? asan_opts + 13 : "";
        if (strstr(cur, "verify_asan_link_order") || strlen(cur) > sizeof aobuf - 64) {
            if (asan_opts) ne[k++] = asan_opts;
        } else {
            snprintf(aobuf, sizeof aobuf, "ASAN_OPTIONS=%s%sverify_asan_link_order=0", cur, *cur ? ":" : "");
            ne[k++] = aobuf;
        }
        pers = syscall(SYS_personality, 0xffffffffUL);
        if (pers != -1 && !(pers & ADDR_NO_RANDOMIZE))
            syscall(SYS_personality, (unsigned long)(pers | ADDR_NO_RANDOMIZE));
        else
            pers = -1;
    }
    ne[k] = NULL;
    long r = at ? syscall(SYS_execveat, dirfd, path, argv, ne, flags) : syscall(SYS_execve, path, argv, ne);
    int saved = errno;
    if (pers != -1) syscall(SYS_personality, (unsigned long)pers);  /* exec failed: restore */
    errno = saved;
    return r;
}

int execve(const char *path, char *const argv[], char *const envp[]) {
    return (int)exec_common(AT_FDCWD, path, argv, envp, 0, 0);
}
int execveat(int dirfd, const char *path, char *const argv[], char *const envp[], int flags) {
    return (int)exec_common(dirfd, path, argv, envp, flags, 1);
}
int fexecve(int fd, char *const argv[], char *const envp[]) {
    return (int)exec_common(fd, "", argv, envp, AT_EMPTY_PATH, 1);
}

int execv(const char *path, char *const argv[]) { return execve(path, argv, environ); }

/* ENOEXEC: run the file as a shell script, like glibc's execvp. */
static void exec_sh(const char *file, char *const argv[], char *const envp[]) {
    size_t n = 0;
    while (argv[n]) n++;
    char *a[n + 3];
    a[0] = (char *)"/bin/sh";
    a[1] = (char *)file;
    a[2] = NULL;
    for (size_t i = 1; i <= n; i++) a[i + 1] = argv[i];
    execve("/bin/sh", a, envp);
}

int execvpe(const char *file, char *const argv[], char *const envp[]) {
    if (!file || !*file) { errno = ENOENT; return -1; }
    if (strchr(file, '/')) {
        execve(file, argv, envp);
        if (errno == ENOEXEC) { exec_sh(file, argv, envp); errno = ENOEXEC; }
        return -1;
    }
    const char *path = getenv("PATH");
    if (!path) path = "/bin:/usr/bin";
    size_t flen = strlen(file);
    char buf[4096 + 256];
    int got_eacces = 0;
    for (const char *p = path;; ) {
        const char *c = strchr(p, ':');
        size_t dlen = c ? (size_t)(c - p) : strlen(p);
        if (dlen + flen + 2 <= sizeof buf) {
            if (dlen) { memcpy(buf, p, dlen); buf[dlen] = '/'; memcpy(buf + dlen + 1, file, flen + 1); }
            else memcpy(buf, file, flen + 1);  /* empty element = cwd */
            execve(buf, argv, envp);
            switch (errno) {
            case ENOEXEC: exec_sh(buf, argv, envp); errno = ENOEXEC; return -1;
            case EACCES: got_eacces = 1; /* fallthrough */
            case ENOENT: case ENOTDIR: case ESTALE: case ENODEV: case ETIMEDOUT: case ELOOP: case ENAMETOOLONG:
                break;
            default: return -1;
            }
        }
        if (!c) break;
        p = c + 1;
    }
    errno = got_eacces ? EACCES : ENOENT;
    return -1;
}

int execvp(const char *file, char *const argv[]) { return execvpe(file, argv, environ); }

#define COLLECT_ARGS(first, ap, argv_name)                                  \
    size_t _n = 1;                                                          \
    { va_list _c; va_copy(_c, ap); while (va_arg(_c, char *)) _n++; va_end(_c); } \
    char *argv_name[_n + 1];                                                \
    argv_name[0] = (char *)(first);                                         \
    for (size_t _i = 1; _i <= _n; _i++) argv_name[_i] = va_arg(ap, char *);

int execl(const char *path, const char *arg, ...) {
    va_list ap; va_start(ap, arg);
    COLLECT_ARGS(arg, ap, av);
    va_end(ap);
    return execve(path, av, environ);
}

int execlp(const char *file, const char *arg, ...) {
    va_list ap; va_start(ap, arg);
    COLLECT_ARGS(arg, ap, av);
    va_end(ap);
    return execvpe(file, av, environ);
}

int execle(const char *path, const char *arg, ...) {
    va_list ap; va_start(ap, arg);
    COLLECT_ARGS(arg, ap, av);
    char *const *envp = va_arg(ap, char *const *);
    va_end(ap);
    return execve(path, av, envp);
}
