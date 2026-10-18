package orchestrator

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

var b12xShardProgress = regexp.MustCompile(`(?m)^b12x routing checkpoint shards: (\d+)/(\d+) shards routed[^\n]*`)
var b12xKernelProgress = regexp.MustCompile(`(?m)^b12x [^\n]*?: (\d+)/(\d+) ready[^\n]*`)

func inferB12XProgress(logs string) progressInfo {
	if hasAny(logs, "Application startup complete", "Uvicorn running on") {
		return inferVLLMProgress(logs)
	}
	if strings.Contains(logs, "JIT kernel warmup") {
		return progressInfo{Key: "b12x-warmup", Phase: "추론 경로 예열", Detail: "가중치와 KV 캐시 준비를 마치고 마지막 커널을 예열합니다.", Progress: .98}
	}
	kernels := b12xKernelProgress.FindAllStringSubmatchIndex(logs, -1)
	if len(kernels) > 0 {
		last := kernels[len(kernels)-1]
		current, _ := strconv.Atoi(logs[last[2]:last[3]])
		total, _ := strconv.Atoi(logs[last[4]:last[5]])
		if total > 0 && current <= total {
			if last[0] > strings.LastIndex(logs, "GPU KV cache size") && strings.Contains(logs, "GPU KV cache size") {
				return progressInfo{Key: "b12x-state", Phase: "긴 문맥 커널 준비", Detail: fmt.Sprintf("%d/%d 커널 준비 완료", current, total), Progress: .90 + .07*float64(current)/float64(total)}
			}
			if !hasAny(logs[last[1]:], "Available KV cache memory", "GPU KV cache size", "Graph capturing finished", "Capturing CUDA graphs") {
				return progressInfo{Key: "b12x-weights", Phase: "가중치 변환·커널 준비", Detail: fmt.Sprintf("%d/%d 커널 준비 완료 · 첫 실행에서는 컴파일에 시간이 걸립니다.", current, total), Progress: .62 + .17*float64(current)/float64(total)}
			}
		}
	}
	if hasAny(logs, "Loading weights took", "Model loading took") {
		return inferVLLMProgress(logs)
	}
	shards := b12xShardProgress.FindAllStringSubmatch(logs, -1)
	if len(shards) > 0 {
		last := shards[len(shards)-1]
		current, _ := strconv.Atoi(last[1])
		total, _ := strconv.Atoi(last[2])
		if total > 0 && current <= total {
			return progressInfo{Key: "b12x-shards", Phase: "FP4-Local 가중치 적재", Detail: fmt.Sprintf("%d/%d 샤드 · SSD에서 통합메모리로 읽는 중", current, total), Progress: .10 + .50*float64(current)/float64(total)}
		}
	}
	return inferVLLMProgress(logs)
}
