package server

import (
	"bufio"
	"context"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

type systemUsage struct {
	CPUPercent int       `json:"cpu_percent"`
	GPUPercent *int      `json:"gpu_percent,omitempty"`
	MemPercent int       `json:"mem_percent"`
	MemUsedGB  float64   `json:"mem_used_gb"`
	MemTotalGB float64   `json:"mem_total_gb"`
	SampledAt  time.Time `json:"sampled_at"`
}

func (s *Server) systemUsage(w http.ResponseWriter, _ *http.Request) {
	s.systemMu.Lock()
	defer s.systemMu.Unlock()

	if time.Since(s.systemStatsAt) < 5*time.Second {
		writeJSON(w, http.StatusOK, s.systemStats)
		return
	}

	total, idle, ok := readCPUTime("/proc/stat")
	cpu := s.systemStats.CPUPercent
	if ok && s.cpuPrevTotal > 0 && total > s.cpuPrevTotal {
		deltaTotal := total - s.cpuPrevTotal
		deltaIdle := idle - s.cpuPrevIdle
		cpu = clampPercent(int((deltaTotal - deltaIdle) * 100 / deltaTotal))
	}
	if ok {
		s.cpuPrevTotal = total
		s.cpuPrevIdle = idle
	}

	memPercent, memUsedGB, memTotalGB := readMemoryUtilization()
	stats := systemUsage{
		CPUPercent: cpu, GPUPercent: readGPUUtilization(), MemPercent: memPercent,
		MemUsedGB: memUsedGB, MemTotalGB: memTotalGB, SampledAt: time.Now(),
	}
	s.systemStats = stats
	s.systemStatsAt = stats.SampledAt
	writeJSON(w, http.StatusOK, stats)
}

func readCPUTime(path string) (total uint64, idle uint64, ok bool) {
	file, err := os.Open(path)
	if err != nil {
		return 0, 0, false
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	if !scanner.Scan() {
		return 0, 0, false
	}
	fields := strings.Fields(scanner.Text())
	if len(fields) < 5 || fields[0] != "cpu" {
		return 0, 0, false
	}
	values := make([]uint64, 0, len(fields)-1)
	for _, field := range fields[1:] {
		value, err := strconv.ParseUint(field, 10, 64)
		if err != nil {
			return 0, 0, false
		}
		values = append(values, value)
		total += value
	}
	idle = values[3]
	if len(values) > 4 {
		idle += values[4]
	}
	return total, idle, true
}

func readGPUUtilization() *int {
	ctx, cancel := context.WithTimeout(context.Background(), 1500*time.Millisecond)
	defer cancel()
	output, err := exec.CommandContext(ctx, "nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return nil
	}
	values := make([]int, 0, 2)
	for _, line := range strings.Split(strings.TrimSpace(string(output)), "\n") {
		value, err := strconv.Atoi(strings.TrimSpace(line))
		if err == nil {
			values = append(values, value)
		}
	}
	if len(values) == 0 {
		return nil
	}
	total := 0
	for _, value := range values {
		total += value
	}
	average := clampPercent(total / len(values))
	return &average
}

func readMemoryUtilization() (percent int, usedGB float64, totalGB float64) {
	file, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0, 0, 0
	}
	defer file.Close()
	var total, available uint64
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) < 2 {
			continue
		}
		value, parseErr := strconv.ParseUint(fields[1], 10, 64)
		if parseErr != nil {
			continue
		}
		switch strings.TrimSuffix(fields[0], ":") {
		case "MemTotal":
			total = value
		case "MemAvailable":
			available = value
		}
	}
	if total == 0 || available > total {
		return 0, 0, 0
	}
	used := total - available
	const kibPerGiB = 1024 * 1024
	return clampPercent(int(used * 100 / total)), float64(used) / kibPerGiB, float64(total) / kibPerGiB
}

func clampPercent(value int) int {
	if value < 0 {
		return 0
	}
	if value > 100 {
		return 100
	}
	return value
}
