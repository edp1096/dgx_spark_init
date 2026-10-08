package orchestrator

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// Account for resident host allocations outside CUDA device memory. Include
// anonymous pages, shared memory and unreclaimable kernel pages, but never
// ordinary checkpoint file cache. shmem is a subset of file, not anon.
func containerHostResidentMemoryGiB(ctx context.Context, container string) float64 {
	memory, _ := containerHostResidentMemory(ctx, container)
	return memory
}

// Keep failed measurements distinct from a successfully measured zero.
func containerHostResidentMemory(ctx context.Context, container string) (float64, bool) {
	for _, pid := range containerPIDs(ctx, container) {
		data, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/cgroup")
		if err != nil {
			continue
		}
		for _, line := range strings.Split(string(data), "\n") {
			if !strings.HasPrefix(line, "0::") {
				continue
			}
			relative := strings.TrimPrefix(strings.TrimPrefix(line, "0::"), "/")
			if relative == "" || strings.Contains(relative, "..") {
				continue
			}
			stat, err := os.ReadFile(filepath.Join("/sys/fs/cgroup", relative, "memory.stat"))
			if err != nil {
				continue
			}
			return hostResidentMemoryGiB(stat), len(stat) > 0
		}
	}
	return 0, false
}

func hostResidentMemoryGiB(stat []byte) float64 {
	values := map[string]uint64{}
	for _, line := range strings.Split(string(stat), "\n") {
		fields := strings.Fields(line)
		if len(fields) == 2 {
			n, err := strconv.ParseUint(fields[1], 10, 64)
			if err == nil {
				values[fields[0]] = n
			}
		}
	}
	kernel := values["kernel"]
	reclaimable := min(kernel, values["slab_reclaimable"])
	return bytesToGiB(values["anon"] + values["shmem"] + kernel - reclaimable)
}
