package orchestrator

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

type progressInfo struct {
	Key      string
	Phase    string
	Detail   string
	Progress float64
	ETA      string
}

var completedProgressPattern = regexp.MustCompile(`(?m)(\d+)% Completed \|\s*(\d+)/(\d+)`)
var barProgressPattern = regexp.MustCompile(`(?m)(\d+)%\|[^\n]*?\|\s*(\d+)/(\d+)`)
var progressETAPattern = regexp.MustCompile(`\[[0-9:]+<([0-9:]+)(?:,|\])`)
var structuredWeightProgressPattern = regexp.MustCompile(`(?m)^SGLANG_WEIGHT_PROGRESS current=(\d+) total=(\d+) elapsed_seconds=([0-9.]+) eta_seconds=([0-9.]+)$`)

func inferProgress(component Component, logs string) progressInfo {
	switch component.ProgressKind {
	case "vllm":
		return inferVLLMProgress(logs)
	case "sglang":
		return inferSGLangProgress(component, logs)
	case "exl3":
		return inferEXL3Progress(logs)
	case "comfy":
		return inferImageProgress(logs)
	default:
		return inferServiceProgress(logs)
	}
}

func inferEXL3Progress(logs string) progressInfo {
	if strings.Contains(logs, "== model ready; accepting requests") {
		return progressInfo{Key: "api", Phase: "API 온라인", Detail: "EXL3 API가 요청을 받을 준비를 마쳤습니다.", Progress: 1}
	}
	if strings.Contains(logs, "-- Loading tokenizer") {
		return progressInfo{Key: "tokenizer", Phase: "토크나이저 준비", Detail: "가중치 적재를 마치고 토크나이저와 API를 준비합니다.", Progress: .88}
	}
	if strings.Contains(logs, "-- Loading /models/target") {
		return progressInfo{Key: "weights", Phase: "EXL3 가중치 적재", Detail: "4bpw 모델과 MTP head를 통합메모리에 올리고 있습니다.", Progress: .45}
	}
	if strings.Contains(logs, "== loading /models/target") {
		return progressInfo{Key: "engine", Phase: "EXL3 엔진 구성", Detail: "262K NVFP4 캐시와 MTP 구성을 적용합니다.", Progress: .10}
	}
	return progressInfo{Key: "container", Phase: "EXL3 컨테이너 시작", Detail: "EXL3 프로세스의 첫 로그를 기다리고 있습니다.", Progress: .03}
}

func inferVLLMProgress(logs string) progressInfo {
	if hasAny(logs, "Application startup complete", "Uvicorn running on") {
		return progressInfo{Key: "api", Phase: "API 온라인", Detail: "vLLM API가 요청을 받을 준비를 마쳤습니다.", Progress: 1}
	}
	if hasAny(logs, "Graph capturing finished", "Capturing CUDA graphs", "Capturing prefill CUDA graphs") {
		progress, detail := graphProgress(logs)
		return progressInfo{Key: "cuda-graph", Phase: "CUDA Graph 캡처", Detail: detail, Progress: .94 + progress*.04, ETA: etaForCurrentProgressLine(logs, barProgressPattern)}
	}
	if strings.Contains(logs, "Capturing model for speculator") {
		return progressInfo{Key: "mtp", Phase: "MTP 추측 디코더 준비", Detail: "Flash Next의 보조 예측 경로를 GPU에 준비하고 있습니다.", Progress: .92}
	}
	if strings.Contains(logs, "GPU KV cache size") {
		return progressInfo{Key: "kv-cache", Phase: "KV 캐시 할당", Detail: "64K 컨텍스트용 GPU KV 캐시를 구성했습니다.", Progress: .89}
	}
	if strings.Contains(logs, "Loading weights took") {
		return progressInfo{Key: "weights-loaded", Phase: "모델 가중치 적재 완료", Detail: "체크포인트 적재를 마치고 추론 메모리를 구성합니다.", Progress: .76}
	}
	if match, percent, ok := latestCompletedProgress(logs); ok {
		return progressInfo{
			Key: "main-weights", Phase: "Flash Next 체크포인트 적재", Detail: fmt.Sprintf("%s/%s 샤드 · SSD에서 통합메모리로 읽는 중", match[2], match[3]),
			Progress: .16 + percent*.58, ETA: etaForCurrentProgressLine(logs, completedProgressPattern),
		}
	}
	if strings.Contains(logs, "PLE mmap") {
		return progressInfo{Key: "ple-mmap", Phase: "SSD PLE·ngram 연결", Detail: "대형 예측 테이블을 RAM에 복사하지 않고 SSD에서 mmap으로 연결합니다.", Progress: .14}
	}
	if hasAny(logs, "Detected ModelOpt NVFP4 checkpoint", "Resolved architecture") {
		return progressInfo{Key: "architecture", Phase: "모델 구조·NVFP4 확인", Detail: "Flash Next 본체와 MTP 구성을 확인했습니다.", Progress: .11}
	}
	if hasAny(logs, "Initializing a V1 LLM engine", "non-default args") {
		return progressInfo{Key: "engine", Phase: "vLLM 엔진 구성", Detail: "실행 옵션과 메모리 계획을 적용하고 있습니다.", Progress: .07}
	}
	return progressInfo{Key: "container", Phase: "Flash Next 컨테이너 시작", Detail: "vLLM 프로세스의 첫 로그를 기다리고 있습니다.", Progress: .03}
}

func inferSGLangProgress(component Component, logs string) progressInfo {
	draft := "DFlash"
	switch component.ID {
	case "qwen27":
		draft = "DFlash2"
	case "flash-next":
		draft = "MTP"
	}
	ready := strings.Contains(logs, "The server is fired up and ready to roll")
	if component.ID != "flash-next" {
		ready = ready || hasAny(logs, "Application startup complete", "Uvicorn running on")
	}
	if ready {
		return progressInfo{Key: "api", Phase: "API 온라인", Detail: "SGLang API가 요청을 받을 준비를 마쳤습니다.", Progress: 1}
	}
	// SGLang's DFlash startup profiles many one-token prefill paths after the
	// graph and cache setup. Those lines can fill the entire Docker log tail.
	if strings.Contains(logs, "Prefill batch, #new-seq: 1") {
		return progressInfo{Key: "draft-warmup", Phase: draft + " 보정·워밍업", Detail: "초기 추론 경로를 측정하고 최적 커널을 고르고 있습니다.", Progress: .97}
	}
	if hasAny(logs, "Capture target verify CUDA graph", "Capture draft verify CUDA graph", "Capturing batches") {
		graphLogs, graphKey, graphName := currentSGLangGraph(logs, draft)
		progress, detail := graphProgress(graphLogs)
		if progress == 0 && hasAny(graphLogs, "torch._dynamo", "torch/_dynamo", "Dynamo detected") {
			return progressInfo{
				Key: "torch-compile-" + graphKey, Phase: graphName + " 커널 컴파일",
				Detail: graphName + "에 사용할 최적 커널을 처음 한 번 컴파일하고 있습니다.", Progress: .84,
			}
		}
		return progressInfo{Key: "cuda-graph-" + graphKey, Phase: graphName + " 캡처", Detail: detail, Progress: .84 + progress*.10, ETA: etaForCurrentProgressLine(graphLogs, barProgressPattern)}
	}
	if hasAny(logs, "Full KV Cache is allocated", "SWA KV Cache is allocated", "KV Cache is allocated", "Memory pool end") {
		phase := "FP8 KV 캐시 할당"
		if component.ID == "flash-next" {
			phase = "KV·Mamba 캐시 할당"
		}
		return progressInfo{Key: "kv-cache", Phase: phase, Detail: "컨텍스트용 KV·상태 메모리 풀을 구성하고 있습니다.", Progress: .80}
	}
	if hasAny(logs, "Initialized DFLASH draft runner", "DFLASH draft runner ready") {
		return progressInfo{Key: "draft-runtime", Phase: draft + " 추측 디코더 구성", Detail: "보조 예측 모델과 fused KV 경로를 연결하고 있습니다.", Progress: .74}
	}
	if match, percent, eta, elapsed, index, ok := latestStructuredWeightProgress(logs); ok {
		isDraft := isDraftWeightProgressAt(logs, index)
		eta = adjustedStructuredWeightETA(component, match, elapsed, eta, isDraft)
		if isDraft {
			return progressInfo{
				Key: "draft-weights", Phase: draft + " 가중치 적재", Detail: fmt.Sprintf("%s/%s 샤드 · 보조 예측 모델", match[1], match[2]),
				Progress: .52 + percent*.18, ETA: eta,
			}
		}
		return progressInfo{
			Key: "main-weights", Phase: modelDisplayName(component) + " 체크포인트 적재", Detail: fmt.Sprintf("%s/%s 샤드 · NVFP4 본체 모델", match[1], match[2]),
			Progress: .12 + percent*.36, ETA: eta,
		}
	}
	if match, percent, ok := latestCompletedProgress(logs); ok {
		isDraft := isDraftWeightProgress(logs)
		if isDraft {
			return progressInfo{
				Key: "draft-weights", Phase: draft + " 가중치 적재", Detail: fmt.Sprintf("%s/%s 샤드 · 보조 예측 모델", match[2], match[3]),
				Progress: .52 + percent*.18, ETA: etaForCurrentProgressLine(logs, completedProgressPattern),
			}
		}
		return progressInfo{
			Key: "main-weights", Phase: modelDisplayName(component) + " 체크포인트 적재", Detail: fmt.Sprintf("%s/%s 샤드 · NVFP4 본체 모델", match[2], match[3]),
			Progress: .12 + percent*.36, ETA: etaForCurrentProgressLine(logs, completedProgressPattern),
		}
	}
	if hasAny(logs, "type=DFlashDraftModel", "type=Qwen4ExpForCausalLMMTP") {
		return progressInfo{Key: "draft-loaded", Phase: draft + " 가중치 적재 완료", Detail: "보조 예측 모델을 통합메모리에 올렸습니다.", Progress: .72}
	}
	if strings.Contains(logs, "Load weight end") {
		return progressInfo{Key: "main-loaded", Phase: modelDisplayName(component) + " 가중치 적재 완료", Detail: "본체 모델을 통합메모리에 올렸습니다.", Progress: .50}
	}
	if component.ID == "flash-next" && hasAny(logs, "using attn output gate", "PLE table: resident set capped") {
		return progressInfo{
			Key: "main-weights", Phase: modelDisplayName(component) + " 체크포인트 적재",
			Detail: "206개 NVFP4 샤드를 열고 첫 샤드 완료를 기다리고 있습니다.", Progress: .16,
		}
	}
	if component.ID == "flash-next" && hasAny(logs, "PLE table", "ple_offload_backend", "file-backed") {
		return progressInfo{Key: "ple-file", Phase: "SSD PLE·ngram 연결", Detail: "대형 ngram 표를 통합메모리에 고정하지 않고 NVMe 파일에 연결하고 있습니다.", Progress: .11}
	}
	if strings.Contains(logs, "Load weight begin") {
		return progressInfo{Key: "weights-open", Phase: modelDisplayName(component) + " 체크포인트 열기", Detail: "로컬 Hugging Face 캐시에서 가중치를 열고 있습니다.", Progress: .10}
	}
	if hasAny(logs, "server_args=", "Launch server") {
		return progressInfo{Key: "engine", Phase: "SGLang 엔진 구성", Detail: modelDisplayName(component) + " 실행 옵션과 추측 디코딩 구성을 적용합니다.", Progress: .06}
	}
	return progressInfo{Key: "container", Phase: modelDisplayName(component) + " 컨테이너 시작", Detail: "SGLang 프로세스의 첫 로그를 기다리고 있습니다.", Progress: .03}
}

func inferImageProgress(logs string) progressInfo {
	if hasAny(logs, "Found quantization metadata", "Loading text encoder", "Loading transformer") {
		return progressInfo{Key: "image-weights", Phase: "FLUX 모델 적재", Detail: "NVFP4 이미지 모델과 텍스트 인코더를 준비하고 있습니다.", Progress: .72}
	}
	return progressInfo{Key: "container", Phase: "FLUX 컨테이너 시작", Detail: "이미지 API 프로세스를 시작하고 있습니다.", Progress: .12}
}

func inferServiceProgress(logs string) progressInfo {
	if hasAny(logs, "Application startup complete", "Uvicorn running on", "listening on", "server started") {
		return progressInfo{Key: "api", Phase: "API 온라인", Detail: "보조 서비스가 요청을 받을 준비를 마쳤습니다.", Progress: 1}
	}
	if hasAny(logs, "Loading model", "load model", "model loaded") {
		return progressInfo{Key: "service-model", Phase: "경량 모델 적재", Detail: "음성 모델을 메모리에 올리고 있습니다.", Progress: .65}
	}
	return progressInfo{Key: "container", Phase: "보조 서비스 시작", Detail: "API 프로세스를 시작하고 있습니다.", Progress: .15}
}

func latestCompletedProgress(logs string) ([]string, float64, bool) {
	matches := completedProgressPattern.FindAllStringSubmatch(logs, -1)
	if len(matches) == 0 {
		return nil, 0, false
	}
	match := matches[len(matches)-1]
	percent, _ := strconv.ParseFloat(match[1], 64)
	return match, percent / 100, true
}

func latestStructuredWeightProgress(logs string) ([]string, float64, string, float64, int, bool) {
	indexes := structuredWeightProgressPattern.FindAllStringSubmatchIndex(logs, -1)
	if len(indexes) == 0 {
		return nil, 0, "", 0, 0, false
	}
	index := indexes[len(indexes)-1]
	match := structuredWeightProgressPattern.FindStringSubmatch(logs[index[0]:index[1]])
	current, currentErr := strconv.Atoi(match[1])
	total, totalErr := strconv.Atoi(match[2])
	elapsedSeconds, elapsedErr := strconv.ParseFloat(match[3], 64)
	etaSeconds, etaErr := strconv.ParseFloat(match[4], 64)
	if currentErr != nil || totalErr != nil || elapsedErr != nil || etaErr != nil || total <= 0 || current < 0 || current > total {
		return nil, 0, "", 0, 0, false
	}
	return match, float64(current) / float64(total), formatETASeconds(etaSeconds), elapsedSeconds, index[0], true
}

func adjustedStructuredWeightETA(component Component, match []string, elapsed float64, fallback string, isDraft bool) string {
	if component.ID != "flash-next" {
		return fallback
	}
	current, currentErr := strconv.Atoi(match[1])
	total, totalErr := strconv.Atoi(match[2])
	if currentErr != nil || totalErr != nil || total != 206 || current <= 0 {
		return fallback
	}

	if isDraft {
		// MTP reads the same index but only one late shard triggers its online
		// expert conversion. Keep the observed ~40 second conversion reserve
		// until that shard has been consumed.
		if current <= 192 {
			fastRate := elapsed / float64(current)
			fastRate = min(.4, max(.15, fastRate))
			return formatETASeconds(float64(192-current)*fastRate + 40)
		}
		return fallback
	}

	// The first 192 shards mostly stream at a steady rate. The final 14 also
	// trigger online NVFP4 materialization and take roughly 26.5 seconds each
	// on GB10. A linear tqdm ETA therefore collapses to seconds several minutes
	// too early. Preserve that measured tail instead of pretending all shards
	// have equal cost.
	const (
		fastShards       = 192
		slowShardSeconds = 26.5
	)
	var remaining float64
	if current < fastShards {
		fastRate := elapsed / float64(current)
		fastRate = min(2.2, max(1.6, fastRate))
		remaining = float64(fastShards-current)*fastRate + float64(total-fastShards)*slowShardSeconds
	} else {
		remaining = float64(total-current) * slowShardSeconds
	}
	return formatETASeconds(remaining)
}

func formatETASeconds(seconds float64) string {
	if seconds < 0 {
		seconds = 0
	}
	totalSeconds := int(seconds + .5)
	if totalSeconds >= 3600 {
		return fmt.Sprintf("%d:%02d:%02d", totalSeconds/3600, (totalSeconds%3600)/60, totalSeconds%60)
	}
	return fmt.Sprintf("%02d:%02d", totalSeconds/60, totalSeconds%60)
}

func graphProgress(logs string) (float64, string) {
	matches := barProgressPattern.FindAllStringSubmatch(logs, -1)
	if len(matches) == 0 {
		if hasAny(logs, "Graph capturing finished", "Capture draft verify CUDA graph end") {
			return 1, "CUDA Graph 캡처를 마쳤습니다."
		}
		return 0, "반복 추론 경로를 CUDA Graph로 캡처하고 있습니다."
	}
	match := matches[len(matches)-1]
	percent, _ := strconv.ParseFloat(match[1], 64)
	return percent / 100, fmt.Sprintf("%s/%s 그래프 배치 캡처", match[2], match[3])
}

func currentSGLangGraph(logs, draft string) (string, string, string) {
	targetIndex := strings.LastIndex(logs, "Capture target verify CUDA graph begin")
	draftIndex := -1
	for _, marker := range []string{
		"Capture draft verify CUDA graph begin",
		"Capture draft decode CUDA graph begin",
		"Capture draft extend CUDA graph begin",
	} {
		if index := strings.LastIndex(logs, marker); index > draftIndex {
			draftIndex = index
		}
	}
	if draftIndex > targetIndex {
		return logs[draftIndex:], "draft", draft + " CUDA Graph"
	}
	if targetIndex >= 0 {
		return logs[targetIndex:], "target", "Target CUDA Graph"
	}
	return logs, "graph", "Target·" + draft + " CUDA Graph"
}

func isDraftWeightProgress(logs string) bool {
	progressIndexes := completedProgressPattern.FindAllStringIndex(logs, -1)
	if len(progressIndexes) == 0 {
		return false
	}
	return isDraftWeightProgressAt(logs, progressIndexes[len(progressIndexes)-1][0])
}

func isDraftWeightProgressAt(logs string, progressIndex int) bool {
	return strings.LastIndex(logs[:progressIndex], "Load weight end") >= 0 ||
		strings.Contains(logs[:progressIndex], "DFlashDraftModel")
}

func etaFromLogs(logs string) string {
	matches := progressETAPattern.FindAllStringSubmatch(logs, -1)
	if len(matches) == 0 {
		return ""
	}
	return strings.TrimSpace(matches[len(matches)-1][1])
}

func etaForCurrentProgressLine(logs string, pattern *regexp.Regexp) string {
	indexes := pattern.FindAllStringIndex(logs, -1)
	if len(indexes) == 0 {
		return ""
	}
	start := indexes[len(indexes)-1][0]
	end := len(logs)
	if offset := strings.IndexByte(logs[start:], '\n'); offset >= 0 {
		end = start + offset
	}
	return etaFromLogs(logs[start:end])
}

func modelDisplayName(component Component) string {
	switch component.ID {
	case "gemma31":
		return "Gemma 31B"
	case "qwen27":
		return "Qwen 27B"
	default:
		return component.Name
	}
}

func hasAny(value string, needles ...string) bool {
	for _, needle := range needles {
		if strings.Contains(value, needle) {
			return true
		}
	}
	return false
}
