#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/ptrace.h>
#include <sys/user.h>
#include <sys/syscall.h>

int main(int argc, char *argv[]) {
    pid_t child_pid;
    FILE *log_file;
    int in_syscall = 0; // <--- The Fix: Toggle flag to track entry vs exit

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <program_to_trace>\n", argv[0]);
        return 1;
    }

    log_file = fopen("sentinel_log.csv", "w");
    if (!log_file) {
        perror("fopen");
        return 1;
    }
    fprintf(log_file, "pid,syscall_nr\n");
    fflush(log_file);

    child_pid = fork();

    if (child_pid == 0) {
        /* CHILD */
        ptrace(PTRACE_TRACEME, 0, NULL, NULL);
        raise(SIGSTOP); 
        execvp(argv[1], &argv[1]);
        perror("execvp");
        exit(1);
    } else {
        /* PARENT */
        int status;
        struct user_regs_struct regs;
        
        waitpid(child_pid, &status, 0);
        ptrace(PTRACE_SETOPTIONS, child_pid, 0, PTRACE_O_TRACESYSGOOD);

        while(1) {
            ptrace(PTRACE_SYSCALL, child_pid, 0, 0);
            waitpid(child_pid, &status, 0);

            if (WIFEXITED(status)) break;

            if (WIFSTOPPED(status) && WSTOPSIG(status) == (SIGTRAP | 0x80)) {
                if (!in_syscall) {
                    // SYSCALL ENTRY
                    ptrace(PTRACE_GETREGS, child_pid, 0, &regs);
                    long syscall_nr = (long)regs.orig_rax;
                    
                    if (syscall_nr != -1) {
                        fprintf(log_file, "%d,%ld\n", child_pid, syscall_nr);
                    }
                    in_syscall = 1; // Mark that we are inside
                } else {
                    // SYSCALL EXIT (Ignore this stop)
                    in_syscall = 0; // Mark that we are out
                }
            }
        }
        printf("[Sentinel] Trace complete. PID %d exited.\n", child_pid);
        fclose(log_file);
    }
    return 0;
}