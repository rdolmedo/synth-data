/*
 * nsinit: PID 1 of a harbor_ns sandbox.
 *
 * Invoked by launch.sh (inside fresh user/mount/pid/net/ipc/uts namespaces)
 * after the rootfs overlay and its /proc, /dev, /sys mounts are in place:
 *
 *   nsinit <rootfs>                 (service mode)
 *   nsinit <rootfs> -- cmd [args]   (command mode, used for image builds)
 *
 * It pivots into <rootfs> (pivot_root(".", ".") + lazy unmount of the old
 * root, so no host path stays reachable from inside the sandbox).
 * Service mode: prints "NSINIT_READY" on stdout, closes stdio, and reaps
 * orphans forever; the host stops the sandbox by SIGKILLing this process,
 * which tears down the whole PID namespace.
 * Command mode: runs cmd with the inherited stdio/environment, reaps until
 * cmd exits, and exits with its status (killing any leftover processes).
 *
 * Static build: gcc -static -O2 -o nsinit nsinit.c
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mount.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

static void die(const char *what) {
    fprintf(stderr, "nsinit: %s: %s\n", what, strerror(errno));
    fflush(stderr);
    _exit(111);
}

int main(int argc, char **argv) {
    int cmd_mode = argc >= 4 && strcmp(argv[2], "--") == 0;
    if (argc != 2 && !cmd_mode) {
        fprintf(stderr, "usage: nsinit <rootfs> [-- cmd args...]\n");
        return 2;
    }
    if (chdir(argv[1]) != 0) die("chdir rootfs");
    if (syscall(SYS_pivot_root, ".", ".") != 0) die("pivot_root");
    if (umount2(".", MNT_DETACH) != 0) die("umount old root");
    if (chdir("/") != 0) die("chdir /");

    if (cmd_mode) {
        pid_t child = fork();
        if (child < 0) die("fork");
        if (child == 0) {
            execvp(argv[3], &argv[3]);
            fprintf(stderr, "nsinit: exec %s: %s\n", argv[3], strerror(errno));
            _exit(127);
        }
        int st = 0;
        for (;;) {
            int s2;
            pid_t p = waitpid(-1, &s2, 0);
            if (p == child) { st = s2; break; }
            if (p < 0 && errno != EINTR) die("waitpid");
        }
        if (WIFEXITED(st)) return WEXITSTATUS(st);
        return 128 + (WIFSIGNALED(st) ? WTERMSIG(st) : 0);
    }

    /* Look like Harbor's Docker PID 1 (`sleep infinity`) instead of exposing
     * host paths in /proc/1/cmdline. The argv strings are contiguous. */
    {
        char *start = argv[0];
        char *end = argv[argc - 1] + strlen(argv[argc - 1]);
        static const char disguise[] = "sleep\0infinity";
        if ((size_t)(end - start) >= sizeof(disguise)) {
            memset(start, 0, end - start);
            memcpy(start, disguise, sizeof(disguise));
        }
        prctl(PR_SET_NAME, "sleep", 0, 0, 0);
    }
    puts("NSINIT_READY");
    fflush(stdout);
    int devnull = open("/dev/null", O_RDWR);
    if (devnull >= 0) {
        dup2(devnull, 0);
        dup2(devnull, 1);
        dup2(devnull, 2);
        if (devnull > 2) close(devnull);
    }

    /* PID 1 ignores default-action signals from inside the namespace; make
     * SIGTERM/SIGINT/SIGHUP explicit no-ops so agents can't stop the box. */
    signal(SIGTERM, SIG_IGN);
    signal(SIGINT, SIG_IGN);
    signal(SIGHUP, SIG_IGN);
    for (;;) {
        pid_t p = waitpid(-1, NULL, 0);
        if (p < 0) {
            if (errno == ECHILD) sleep(1);
            else if (errno != EINTR) sleep(1);
        }
    }
}
