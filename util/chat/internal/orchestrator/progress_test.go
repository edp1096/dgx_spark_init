package orchestrator

import (
	"math"
	"testing"
	"time"
)

func TestInferProgress(t *testing.T) {
	component := Component{ID: "flash-next", Name: "Qwen3.8 Flash-Next", ProgressKind: "sglang"}
	info := inferProgress(component, "Load weight begin.\nMulti-thread loading shards:  10% Completed | 20/206 [00:03<01:10,  2.50it/s]\nMulti-thread loading shards:  57% Completed | 118/206 [00:24<00:20,  4.33it/s]")
	if info.Phase != "Qwen3.8 Flash-Next 체크포인트 적재" || info.Detail != "118/206 샤드 · NVFP4 본체 모델" || math.Abs(info.Progress-.3252) > .00001 || info.ETA != "00:20" {
		t.Fatalf("unexpected progress: %#v", info)
	}
	info = inferProgress(component, "PLE table opened with ple_offload_backend=file\nMamba Cache is allocated\nKV Cache is allocated")
	if info.Phase != "KV·Mamba 캐시 할당" || info.Progress != .80 {
		t.Fatalf("unexpected warmup: %#v", info)
	}
}

func TestInferEXL3Progress(t *testing.T) {
	component := Component{ID: "qwen27-exl3", ProgressKind: "exl3"}
	for _, test := range []struct {
		logs string
		key  string
	}{
		{"== loading /models/target + MTP head", "engine"},
		{"== loading /models/target + MTP head\n-- Loading /models/target", "weights"},
		{"-- Loading /models/target\n-- Loading tokenizer...", "tokenizer"},
		{"== model ready; accepting requests", "api"},
	} {
		info := inferProgress(component, test.logs)
		if info.Key != test.key {
			t.Fatalf("logs %q produced %#v, want %q", test.logs, info, test.key)
		}
	}
}

func TestInferFlashNextPLEAndMTPProgress(t *testing.T) {
	component := Component{ID: "flash-next", Name: "Qwen3.8 Flash-Next", ProgressKind: "sglang"}
	info := inferProgress(component, "PLE table opened with ple_offload_backend=file")
	if info.Key != "ple-file" || info.Phase != "SSD PLE·ngram 연결" || info.Progress != .11 {
		t.Fatalf("unexpected PLE progress: %#v", info)
	}
	info = inferProgress(component, "Capture draft verify CUDA graph begin\nCapturing batches: 50%|xxxxx| 1/2 [00:01<00:01, 1.00s/it]")
	if info.Key != "cuda-graph-draft" || info.Phase != "MTP CUDA Graph 캡처" || info.Progress != .89 {
		t.Fatalf("unexpected MTP progress: %#v", info)
	}
}

func TestInferFlashNextBufferedWeightLoading(t *testing.T) {
	component := Component{ID: "flash-next", Name: "Qwen3.8 Flash-Next", ProgressKind: "sglang"}
	logs := "PLE table: file-backed mmap /ple/table.bin\nPLE table: resident set capped at 8.0 GiB\nusing attn output gate!"
	info := inferProgress(component, logs)
	if info.Key != "main-weights" || info.Progress != .16 {
		t.Fatalf("buffered tqdm must still report weight loading: %#v", info)
	}
}

func TestInferStructuredSGLangWeightProgress(t *testing.T) {
	component := Component{ID: "flash-next", Name: "Qwen3.8 Flash-Next", ProgressKind: "sglang"}
	logs := "Load weight begin.\nSGLANG_WEIGHT_PROGRESS current=104 total=206 elapsed_seconds=183.2 eta_seconds=179.7\n"
	info := inferProgress(component, logs)
	if info.Key != "main-weights" || info.Detail != "104/206 샤드 · NVFP4 본체 모델" || math.Abs(info.Progress-.3017475728) > .00001 || info.ETA != "08:46" {
		t.Fatalf("unexpected structured main progress: %#v", info)
	}

	logs += "Load weight end. elapsed=400.0 s\nLoad weight begin.\nSGLANG_WEIGHT_PROGRESS current=40 total=206 elapsed_seconds=8.1 eta_seconds=33.6\n"
	info = inferProgress(component, logs)
	if info.Key != "draft-weights" || info.Detail != "40/206 샤드 · 보조 예측 모델" || math.Abs(info.Progress-.5549514563) > .00001 || info.ETA != "01:11" {
		t.Fatalf("unexpected structured MTP progress: %#v", info)
	}
}

func TestFlashNextStructuredETAKeepsSlowQuantizationTail(t *testing.T) {
	component := Component{ID: "flash-next", Name: "Qwen3.8 Flash-Next", ProgressKind: "sglang"}
	logs := "Load weight begin.\nSGLANG_WEIGHT_PROGRESS current=192 total=206 elapsed_seconds=342.0 eta_seconds=24.9\n"
	info := inferProgress(component, logs)
	if info.ETA != "06:11" {
		t.Fatalf("192/206 must retain the measured NVFP4 tail, got %#v", info)
	}
	logs += "SGLANG_WEIGHT_PROGRESS current=196 total=206 elapsed_seconds=416.5 eta_seconds=21.2\n"
	info = inferProgress(component, logs)
	if info.ETA != "04:25" {
		t.Fatalf("196/206 must not claim only seconds remain, got %#v", info)
	}
}

func TestEstimateFlashNextBufferedWeightProgress(t *testing.T) {
	info := progressInfo{Key: "main-weights", Progress: .16}
	info = estimateFlashNextWeightProgress(info, 3*time.Minute)
	if math.Abs(info.Progress-.2666666667) > .00001 || info.ETA != "06:00" {
		t.Fatalf("unexpected estimated weight progress: %#v", info)
	}
}

func TestInferFlashNextWaitsForSGLangWarmup(t *testing.T) {
	component := Component{ID: "flash-next", Name: "Qwen3.8 Flash-Next", ProgressKind: "sglang"}
	info := inferProgress(component, "Application startup complete.\nUvicorn running on http://0.0.0.0:30000\nPrefill batch, #new-seq: 1, #new-token: 128")
	if info.Key != "draft-warmup" || info.Progress != .97 {
		t.Fatalf("Flash-Next must not become ready before its internal warmup: %#v", info)
	}
	info = inferProgress(component, "Application startup complete.\nThe server is fired up and ready to roll!")
	if info.Key != "api" || info.Progress != 1 {
		t.Fatalf("unexpected ready progress: %#v", info)
	}
}

func TestInferFlashNextDraftGraphMarkers(t *testing.T) {
	component := Component{ID: "flash-next", Name: "Qwen3.8 Flash-Next", ProgressKind: "sglang"}
	logs := "Capture target verify CUDA graph begin\nCapturing batches: 100%|xxxxx| 2/2 [00:29<00:00]\nCapture target verify CUDA graph end\nCapture draft extend CUDA graph begin\nCapturing batches: 50%|xxxxx| 1/2 [00:01<00:01]"
	info := inferProgress(component, logs)
	if info.Key != "cuda-graph-draft" || info.Phase != "MTP CUDA Graph 캡처" || info.Progress != .89 {
		t.Fatalf("unexpected Flash-Next draft graph progress: %#v", info)
	}
}

func TestInferGemmaDFlashProgress(t *testing.T) {
	component := Component{ID: "gemma31", ProgressKind: "sglang"}
	logs := "Load weight begin.\nMulti-thread loading shards: 100% Completed | 1/1 [00:24<00:00]\nLoad weight end. type=Gemma4ForConditionalGeneration\nLoad weight begin.\nMulti-thread loading shards: 50% Completed | 1/2 [00:08<00:08]"
	info := inferProgress(component, logs)
	if info.Key != "draft-weights" || info.Phase != "DFlash 가중치 적재" || info.Progress != .61 {
		t.Fatalf("unexpected DFlash progress: %#v", info)
	}
	info = inferProgress(component, "Prefill batch, #new-seq: 1, #new-token: 1, cuda graph: False")
	if info.Key != "draft-warmup" || info.Progress != .97 {
		t.Fatalf("unexpected DFlash warmup: %#v", info)
	}
}

func TestInferQwenDFlash2AndGraphProgress(t *testing.T) {
	component := Component{ID: "qwen27", ProgressKind: "sglang"}
	logs := "Capture target verify CUDA graph begin\nMAX_FUSED_QKV_SPLIT_DIM: <torch._dynamo warning potential risk>\nCapturing batches (bs=1):  50%|xxxxx| 1/2 [00:01<00:01, 1.00s/it]"
	info := inferProgress(component, logs)
	if info.Key != "cuda-graph-target" || info.Phase != "Target CUDA Graph 캡처" || info.Progress != .89 || info.ETA != "00:01" {
		t.Fatalf("unexpected Qwen graph progress: %#v", info)
	}
}

func TestInferQwenTorchCompileBeforeGraphCapture(t *testing.T) {
	component := Component{ID: "qwen27", ProgressKind: "sglang"}
	logs := "Capture target verify CUDA graph begin\nCapturing batches: 0%| | 0/2 [00:00<?, ?it/s]\ntorch._dynamo.utils.warn_once(msg)"
	info := inferProgress(component, logs)
	if info.Key != "torch-compile-target" || info.Phase != "Target CUDA Graph 커널 컴파일" || info.ETA != "" {
		t.Fatalf("unexpected Qwen compile progress: %#v", info)
	}
}

func TestInferDraftGraphUsesOnlyDraftSection(t *testing.T) {
	component := Component{ID: "gemma31", ProgressKind: "sglang"}
	logs := "Capture target verify CUDA graph begin\nCapturing batches: 100%|xxxxx| 2/2 [00:10<00:00, 1.00s/it]\nCapture target verify CUDA graph end\nCapture draft verify CUDA graph begin\nCapturing batches: 0%| | 0/2 [00:00<?, ?it/s]"
	info := inferProgress(component, logs)
	if info.Key != "cuda-graph-draft" || info.Phase != "DFlash CUDA Graph 캡처" || info.Progress != .84 || info.ETA != "" {
		t.Fatalf("unexpected draft graph progress: %#v", info)
	}
}

func TestETARejectsTorchAngleBracketWarnings(t *testing.T) {
	logs := "Capturing batches: 0%| | 0/2 [00:00<?, ?it/s]\nMAX_FUSED_QKV_SPLIT_DIM: <torch._dynamo warning potential risk>"
	if eta := etaForCurrentProgressLine(logs, barProgressPattern); eta != "" {
		t.Fatalf("warning text must not become an ETA: %q", eta)
	}
}

func TestImageWrapperStartupIsNotModelReady(t *testing.T) {
	component := Component{ID: "flux2", ProgressKind: "comfy"}
	info := inferProgress(component, "Application startup complete.\nUvicorn running on http://0.0.0.0:8691")
	if info.Key != "container" || info.Progress != .12 {
		t.Fatalf("wrapper startup must not report the image model ready: %#v", info)
	}
}

func TestStartupFailureDetectsCUDAOOM(t *testing.T) {
	logs := "torch.AcceleratorError: CUDA error: out of memory"
	if failure := startupFailure(logs); failure == "" {
		t.Fatal("CUDA OOM must stop startup polling")
	}
}
