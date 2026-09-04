#include "numa.h"
#include "ggml-impl.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#elif defined(__gnu_linux__)
#    include <sched.h>
#    include <sys/mman.h>
#    include <sys/syscall.h>
#    include <unistd.h>
#endif

#define GGML_CPU_NUMA_MAX_NODES 64

#if defined(_WIN32)

static std::vector<ggml_cpu_numa_node> ggml_cpu_numa_enumerate(void) {
    std::vector<ggml_cpu_numa_node> nodes;

    ULONG highest = 0;
    if (!GetNumaHighestNodeNumber(&highest)) {
        return nodes;
    }

    for (ULONG n = 0; n <= highest; n++) {
        GROUP_AFFINITY ga;
        memset(&ga, 0, sizeof(ga));
        if (!GetNumaNodeProcessorMaskEx((USHORT) n, &ga) || ga.Mask == 0) {
            continue; // node without processors
        }
        if (ga.Group != 0) {
            GGML_LOG_WARN("%s: NUMA node %lu is in processor group %u, only group 0 is supported\n", __func__, (unsigned long) n, (unsigned) ga.Group);
            continue;
        }

        ggml_cpu_numa_node node;
        memset(&node, 0, sizeof(node));
        node.id = (int) n;
        for (int i = 0; i < 64; i++) {
            if (ga.Mask & ((KAFFINITY) 1 << i)) {
                node.cpumask[i] = true;
                node.n_cpus++;
            }
        }
        nodes.push_back(node);
    }

    return nodes;
}

static int ggml_cpu_numa_process_cpus(bool * cpumask) {
    DWORD_PTR process_mask = 0;
    DWORD_PTR system_mask  = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask)) {
        return 0;
    }

    int n_cpus = 0;
    for (int i = 0; i < 64; i++) {
        if (process_mask & ((DWORD_PTR) 1 << i)) {
            cpumask[i] = true;
            n_cpus++;
        }
    }
    return n_cpus;
}

void ggml_cpu_numa_memory(int node, size_t * free, size_t * total) {
    ULONGLONG avail = 0;
    if (!GetNumaAvailableMemoryNodeEx((USHORT) node, &avail)) {
        avail = 0;
    }
    *free = (size_t) avail;

    // Windows does not report the size of a node, assume equal nodes
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    ULONG highest = 0;
    GetNumaHighestNodeNumber(&highest);
    *total = (size_t) (status.ullTotalPhys / (highest + 1));
}

void * ggml_cpu_numa_alloc(size_t size, int node) {
    // pages are placed on the node at first touch, other nodes are used when it is full
    return VirtualAllocExNuma(GetCurrentProcess(), NULL, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE, (DWORD) node);
}

void ggml_cpu_numa_free(void * ptr, size_t size) {
    GGML_UNUSED(size);
    if (ptr != NULL) {
        VirtualFree(ptr, 0, MEM_RELEASE);
    }
}

#elif defined(__gnu_linux__)

// "0-3,8-11" -> cpumask
static int ggml_cpu_numa_parse_cpulist(const char * list, bool * cpumask) {
    int n_cpus = 0;
    const char * p = list;
    while (*p != '\0') {
        char * end = NULL;
        const long first = strtol(p, &end, 10);
        if (end == p) {
            break;
        }
        long last = first;
        p = end;
        if (*p == '-') {
            last = strtol(p + 1, &end, 10);
            p = end;
        }
        for (long c = first; c <= last && c < GGML_MAX_N_THREADS; c++) {
            if (c >= 0 && !cpumask[c]) {
                cpumask[c] = true;
                n_cpus++;
            }
        }
        while (*p == ',' || *p == ' ' || *p == '\n') {
            p++;
        }
    }
    return n_cpus;
}

static std::vector<ggml_cpu_numa_node> ggml_cpu_numa_enumerate(void) {
    std::vector<ggml_cpu_numa_node> nodes;

    for (int n = 0; n < GGML_CPU_NUMA_MAX_NODES; n++) {
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", n);
        FILE * f = fopen(path, "r");
        if (f == NULL) {
            continue;
        }
        char buf[4096];
        if (fgets(buf, sizeof(buf), f) == NULL) {
            buf[0] = '\0';
        }
        fclose(f);

        ggml_cpu_numa_node node;
        memset(&node, 0, sizeof(node));
        node.id     = n;
        node.n_cpus = ggml_cpu_numa_parse_cpulist(buf, node.cpumask);
        if (node.n_cpus > 0) {
            nodes.push_back(node);
        }
    }

    return nodes;
}

static int ggml_cpu_numa_process_cpus(bool * cpumask) {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) {
        return 0;
    }

    int n_cpus = 0;
    for (int i = 0; i < CPU_SETSIZE && i < GGML_MAX_N_THREADS; i++) {
        if (CPU_ISSET(i, &set)) {
            cpumask[i] = true;
            n_cpus++;
        }
    }
    return n_cpus;
}

// value of a "Node N MemTotal: 123 kB" line
static size_t ggml_cpu_numa_meminfo(int node, const char * key) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/meminfo", node);
    FILE * f = fopen(path, "r");
    if (f == NULL) {
        return 0;
    }

    size_t result = 0;
    char line[256];
    while (fgets(line, sizeof(line), f) != NULL) {
        const char * p = strstr(line, key);
        if (p == NULL) {
            continue;
        }
        result = (size_t) strtoull(p + strlen(key), NULL, 10) * 1024;
        break;
    }
    fclose(f);
    return result;
}

void ggml_cpu_numa_memory(int node, size_t * free, size_t * total) {
    *total = ggml_cpu_numa_meminfo(node, "MemTotal:");
    *free  = ggml_cpu_numa_meminfo(node, "MemFree:");
}

void * ggml_cpu_numa_alloc(size_t size, int node) {
    void * ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        return NULL;
    }

    // MPOL_BIND, same policy as numa_alloc_onnode(); numaif.h is not always installed
    const int mpol_bind = 2;
    unsigned long nodemask = 1UL << node;
    if (syscall(SYS_mbind, ptr, size, mpol_bind, &nodemask, 8*sizeof(nodemask), 0) != 0) {
        GGML_LOG_WARN("%s: mbind to NUMA node %d failed: %s\n", __func__, node, strerror(errno));
    }
    return ptr;
}

void ggml_cpu_numa_free(void * ptr, size_t size) {
    if (ptr != NULL) {
        munmap(ptr, size);
    }
}

#else

static std::vector<ggml_cpu_numa_node> ggml_cpu_numa_enumerate(void) {
    return {};
}

static int ggml_cpu_numa_process_cpus(bool * cpumask) {
    GGML_UNUSED(cpumask);
    return 0;
}

void ggml_cpu_numa_memory(int node, size_t * free, size_t * total) {
    GGML_UNUSED(node);
    *free  = 0;
    *total = 0;
}

void * ggml_cpu_numa_alloc(size_t size, int node) {
    GGML_UNUSED(size);
    GGML_UNUSED(node);
    return NULL;
}

void ggml_cpu_numa_free(void * ptr, size_t size) {
    GGML_UNUSED(ptr);
    GGML_UNUSED(size);
}

#endif

// split the processors of the process into n virtual nodes with all memory on node 0, for testing on a machine without NUMA
static std::vector<ggml_cpu_numa_node> ggml_cpu_numa_fake_nodes(int n_nodes) {
    std::vector<ggml_cpu_numa_node> nodes;

    bool cpumask[GGML_MAX_N_THREADS];
    memset(cpumask, 0, sizeof(cpumask));
    const int n_cpus = ggml_cpu_numa_process_cpus(cpumask);
    if (n_cpus < n_nodes) {
        return nodes;
    }

    int cpu = 0;
    for (int n = 0; n < n_nodes; n++) {
        ggml_cpu_numa_node node;
        memset(&node, 0, sizeof(node));
        node.id = 0;
        const int n_take = n_cpus / n_nodes + (n < n_cpus % n_nodes ? 1 : 0);
        for (; node.n_cpus < n_take && cpu < GGML_MAX_N_THREADS; cpu++) {
            if (cpumask[cpu]) {
                node.cpumask[cpu] = true;
                node.n_cpus++;
            }
        }
        nodes.push_back(node);
    }

    return nodes;
}

const std::vector<ggml_cpu_numa_node> & ggml_cpu_numa_nodes(void) {
    static std::vector<ggml_cpu_numa_node> nodes = []() {
        // GGML_CPU_NUMA_DEVICES=0 disables the NUMA devices, n>=2 makes n virtual nodes (testing only)
        const char * env = getenv("GGML_CPU_NUMA_DEVICES");
        if (env != NULL) {
            const int n = atoi(env);
            return n >= 2 ? ggml_cpu_numa_fake_nodes(n) : std::vector<ggml_cpu_numa_node>();
        }
        return ggml_cpu_numa_enumerate();
    }();

    return nodes;
}
