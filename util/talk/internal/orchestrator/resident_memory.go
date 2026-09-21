package orchestrator

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// Account for anonymous host allocations already deducted from MemAvailable.
// Do not credit reclaimable file cache (including mmap checkpoint pages).
func containerAnonymousMemoryGiB(ctx context.Context, container string) float64 {
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
			return anonymousMemoryGiB(stat)
		}
	}
	return 0
}

func anonymousMemoryGiB(stat []byte) float64 {
	for _, line := range strings.Split(string(stat), "\n") {
		fields := strings.Fields(line)
		if len(fields) == 2 && fields[0] == "anon" {
			n, err := strconv.ParseUint(fields[1], 10, 64)
			if err == nil {
				return bytesToGiB(n)
			}
		}
	}
	return 0
}
