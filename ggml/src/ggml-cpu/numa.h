#pragma once

#include "ggml.h"

#include <stddef.h>
#include <vector>

// NUMA node with at least one logical processor
struct ggml_cpu_numa_node {
    int  id;                          // node number of the OS
    int  n_cpus;
    bool cpumask[GGML_MAX_N_THREADS]; // logical processors of the node
};

// nodes of the system, empty if NUMA is not supported by the OS or disabled with GGML_CPU_NUMA_DEVICES=0
const std::vector<ggml_cpu_numa_node> & ggml_cpu_numa_nodes(void);

void ggml_cpu_numa_memory(int node, size_t * free, size_t * total);

// zeroed, page-aligned memory placed on a node, NULL on failure
void * ggml_cpu_numa_alloc(size_t size, int node);
void   ggml_cpu_numa_free (void * ptr, size_t size);
