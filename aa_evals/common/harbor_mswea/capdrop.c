/* Agent descendants may not join the evaluator's network namespace or trace its privileged processes. */
#define _GNU_SOURCE
#include <errno.h>
#include <linux/capability.h>
#include <stdio.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: capdrop COMMAND [ARG...]\n"); return 2; }
    int denied[] = {CAP_SYS_ADMIN, CAP_SYS_PTRACE};
    if (prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_CLEAR_ALL, 0, 0, 0) < 0 ||
        prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) < 0) { perror("capdrop no-new-privileges"); return 126; }
    struct __user_cap_header_struct header = {_LINUX_CAPABILITY_VERSION_3, 0};
    struct __user_cap_data_struct data[2];
    if (syscall(SYS_capget, &header, data) < 0) { perror("capget"); return 126; }
    for (unsigned int i = 0; i < sizeof(denied) / sizeof(denied[0]); ++i) {
        int cap = denied[i];
        /* A task's non-root user may already lack both capabilities and CAP_SETPCAP. no_new_privs
           also prevents that user gaining them through setuid executables or file capabilities. */
        if (prctl(PR_CAPBSET_DROP, cap, 0, 0, 0) < 0 &&
            !(errno == EPERM && !(data[cap / 32].permitted & (1U << (cap % 32))))) {
            perror("cap bounding drop"); return 126;
        }
        unsigned int mask = ~(1U << (cap % 32));
        data[cap / 32].effective &= mask;
        data[cap / 32].permitted &= mask;
        data[cap / 32].inheritable &= mask;
    }
    if (syscall(SYS_capset, &header, data) < 0) { perror("capset"); return 126; }
    execvp(argv[1], argv + 1);
    perror("execvp");
    return errno == ENOENT ? 127 : 126;
}
