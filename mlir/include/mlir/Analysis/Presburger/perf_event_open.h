#include <asm/unistd.h>
#include <linux/perf_event.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "assert.h"

#include <inttypes.h>

static long
perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                int cpu, int group_fd, unsigned long flags)
{
    int ret;

    ret = syscall(__NR_perf_event_open, hw_event, pid, cpu,
                    group_fd, flags);
    return ret;
}

int get_fd(__u32 type, __u64 config) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(struct perf_event_attr));
    pe.type = type;
    pe.size = sizeof(struct perf_event_attr);
    pe.config = config;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    // Don't count hypervisor events.
    pe.exclude_hv = 1;
    return perf_event_open(&pe, 0, -1, -1, 0);
}

void reset_and_enable(int fd) {
  ioctl(fd, PERF_EVENT_IOC_RESET, 0);
  ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
}

long long disable_and_get_count(int fd) {
  long long count;
  ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
  read(fd, &count, sizeof(long long));
  return count;
}

#define FD_ARRAY_LENGTH  5
int fd[FD_ARRAY_LENGTH];
unsigned fd_count = 0;

void init_perf_fds() {
  fd[fd_count++] = get_fd(PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16));
  fd[fd_count++] = get_fd(PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));

  for (unsigned i = 0; i < fd_count; ++i) {
    if (fd[i] == -1) {
      perror("error: perf_event_open");
      exit(1);
    }
  }

  assert(fd_count <= FD_ARRAY_LENGTH);
}

void reset_and_enable_all() {
  assert(fd_count != 0 && "fds not initialized!");
  for (unsigned i = 0; i < fd_count; ++i)
    reset_and_enable(fd[i]);
}

void disable_all_and_print_counts(FILE *file) {
  for (unsigned i = 0; i < fd_count; ++i)
    fprintf(file, "%lld\n", disable_and_get_count(fd[i]));
  assert(fd_count != 0 && "fds not initialized!");
}
