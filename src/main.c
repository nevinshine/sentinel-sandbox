#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <string.h>

#include <sys/types.h>
#include <sys/wait.h>
#include <sys/ptrace.h>
#include <sys/user.h>
#include <sys/syscall.h>

/*
 * ARGUMENT CLASSIFICATION
 * 0 = other / unknown
 * 1 = system files (/etc, /bin, /lib)
 * 2 = proc/sys files (/proc, /sys)
 * 3 = user files (/home, /tmp)
 */
int classify_path(const char *path) {
    if (!path) return 0;

    if (strncmp(path, "/etc", 4) == 0 ||
        strncmp(path, "/bin", 4) == 0 ||
        strncmp(path, "/lib", 4) == 0)
        return 1;

    if (strncmp(path, "/proc", 5) == 0 ||
        strncmp(path, "/sys", 4) == 0)
        return 2;

    if (strncmp(path, "/home", 5) == 0 ||
        strncmp(path, "/tmp", 4) == 0)
        return 3;

    return 0;
}

/*
 * SAFELY READ STRING FROM CHILD MEMORY
 */
void read_child_string(pid_t pid, unsigned long addr, char *buf, size_t maxlen) {
    size_t i = 0;
    long word;

    while (i < maxlen - sizeof(long)) {
        word = ptrace(PTRACE_PEEKDATA, pid, addr + i, NULL);
        if (word == -1) break;

        memcpy(buf + i, &word, sizeof(long));

        if (memchr(&word, 0, sizeof(long)) != NULL)
            break;

        i += sizeof(long);
    }

    buf[maxlen - 1] = '\0';
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <program_to_trace>\n", argv[0]);
        return 1;
    }

    FILE *log_file = fopen("sentinel_log.csv", "w");
    if (!log_file) {
        perror("fopen");
        return 1;
    }

    fprintf(log_file, "pid,syscall_nr,arg_class\n");
    fflush(log_file);

    pid_t child = fork();

    if (child == 0) {
        /* CHILD */
        ptrace(PTRACE_TRACEME, 0, NULL, NULL);
        raise(SIGSTOP);
        execvp(argv[1], &argv[1]);
        perror("execvp");
        exit(1);
    }

    /* PARENT */
    int status;
    int in_syscall = 0;
    struct user_regs_struct regs;

    waitpid(child, &status, 0);
    ptrace(PTRACE_SETOPTIONS, child, 0, PTRACE_O_TRACESYSGOOD);

    while (1) {
        ptrace(PTRACE_SYSCALL, child, 0, 0);
        waitpid(child, &status, 0);

        if (WIFEXITED(status))
            break;

        if (WIFSTOPPED(status) &&
            WSTOPSIG(status) == (SIGTRAP | 0x80)) {

            if (!in_syscall) {
                /* SYSCALL ENTRY */
                ptrace(PTRACE_GETREGS, child, 0, &regs);

                long syscall_nr = regs.orig_rax;
                int arg_class = 0;

                if (syscall_nr == SYS_open) {
    char path[256] = {0};
    unsigned long path_addr = regs.rdi;

    read_child_string(child, path_addr, path, sizeof(path));
    arg_class = classify_path(path);

} else if (syscall_nr == SYS_openat) {
    char path[256] = {0};
    unsigned long path_addr = regs.rsi;  // <-- FIX HERE

    read_child_string(child, path_addr, path, sizeof(path));
    arg_class = classify_path(path);
}


                fprintf(log_file, "%d,%ld,%d\n",
                        child, syscall_nr, arg_class);

                in_syscall = 1;
            } else {
                /* SYSCALL EXIT — ignore */
                in_syscall = 0;
            }
        }
    }

    printf("[Sentinel] Trace complete. PID %d exited.\n", child);
    fclose(log_file);
    return 0;
}
